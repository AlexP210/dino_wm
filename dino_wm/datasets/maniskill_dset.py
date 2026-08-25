"""Loading ManiSkill trajectory recordings, independent of which task produced them.

Everything here is task-agnostic: camera detection, the two storage layouts a recording can have
(gzipped straight out of tools/replay_trajectory.py, or contiguous after
tools/preprocess_data.py --memmap), $SLURM_TMPDIR staging, the frame-stack/frameskip windowing the
trajectory slicer asks for, and the normalization statistics.

A task module supplies only the two things that differ -- which recorded fields make up its flat
proprio and state vectors -- by subclassing `ManiSkillTrajDataset`; see pushcube_dset.py and
liftpeg_dset.py. Sharing rather than copying is deliberate: the copies this replaces had already
drifted from each other, one of them looking for its DINO features under a camera its images did
not come from, which does not fail loudly -- it silently yields observations with no features.
"""
import abc
import os
import shutil

import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional, Sequence
from .traj_dset import TrajDataset, get_train_val_sliced

RGB_TEMPLATE = "obs/sensor_data/{camera}/rgb"
DINO_TEMPLATE = "obs/sensor_data/{camera}/dino_patch_features"


def _detect_camera(traj: h5py.Group, camera: Optional[str], path: Path) -> str:
    """Which sensor's images this recording carries.

    Read off the file rather than hardcoded, because which camera a recording holds is a
    property of how it was replayed: `--camera-view wrist` writes `hand_camera` and drops the
    task's own camera, `focused`/`default` write `base_camera`. Deriving it here means the rgb
    and the DINO features can never be looked up under different cameras -- a mismatch that
    costs nothing at load time (both the dataset loader and the env wrapper silently skip
    observation paths they cannot resolve) and only surfaces much later as a missing key.
    """
    sensors = traj.get("obs/sensor_data")
    if sensors is None:
        raise ValueError(
            f"{path.name} has no obs/sensor_data: this is a state-only recording (obs_mode="
            "state, what tools/ppo_stages_fast.py writes). Render it to images first with "
            "tools/replay_trajectory.py -o rgb --camera-view <view> --save-traj."
        )
    cameras = sorted(name for name in sensors if "rgb" in sensors[name])
    if camera is not None:
        if camera not in cameras:
            raise ValueError(f"{path.name} has no rgb for camera {camera!r}; it has {cameras}")
        return camera
    if len(cameras) != 1:
        raise ValueError(
            f"{path.name} carries {len(cameras)} cameras with rgb ({cameras}); pass "
            "`camera=` to say which one the world model should see."
        )
    return cameras[0]


def _view_dataset(raw: np.memmap, dataset: h5py.Dataset) -> Optional[np.ndarray]:
    """
    Zero-copy view of a contiguous (uncompressed, unchunked) h5py.Dataset, backed by `raw`
    — a single np.memmap covering the whole source file. This reads no data itself: disk
    I/O only happens lazily, a page at a time, once the returned array is actually indexed.

    Returns None if `dataset` is chunked and/or compressed: such a dataset has no fixed
    file offset and cannot be memory-mapped, so the caller falls back to _H5View.
    """
    offset = dataset.id.get_offset()
    if offset is None:
        return None
    return np.ndarray(shape=dataset.shape, dtype=dataset.dtype, buffer=raw, offset=offset)


# One HDF5 handle per file per process, shared by every _H5View onto it. Keyed by pid
# because an h5py handle must not be shared across a fork: a DataLoader worker inherits
# the parent's entry, sees a pid that no longer matches, and opens its own.
_OPEN_FILES: dict = {}


def _open_file(path: str) -> h5py.File:
    key = (path, os.getpid())
    handle = _OPEN_FILES.get(key)
    if handle is None:
        handle = h5py.File(path, "r")
        _OPEN_FILES[key] = handle
    return handle


class _H5View:
    """
    Stand-in for the np.memmap view _view_dataset returns, for datasets that are chunked
    and/or compressed and so cannot be mapped. Indexes the same way, but reads (and
    decompresses) through h5py on every access rather than paging in from disk — correct,
    but with none of the OS page cache reuse the memmap path gets across epochs.

    Deliberately holds no open handle of its own, only the coordinates needed to reach the
    data. There is one view per trajectory, and HDF5 allocates a chunk cache per *open
    dataset* — around 99 KB apiece for this file, so keeping them open costs ~2.8 GB per
    process across 29,696 trajectories, which is enough to get DataLoader workers OOM-killed.
    The cache buys nothing here anyway: get_frames reads each trajectory straight through.
    """

    def __init__(self, path: str, name: str, shape, dtype):
        self.path = path
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, idx):
        dataset = _open_file(self.path)[self.name]
        # h5py's list-selection reads element by element, re-fetching (and re-inflating)
        # every chunk an index touches once per index. That is ruinous here: with a chunk
        # shape like (13, 56, 56, 1) a single frame spans 48 chunks, so a 4-frame fancy
        # read costs more than twice a 20-frame slice.
        #
        # get_frames asks for an ascending run of frames — contiguous at frameskip 1,
        # strided above it — so either way one slice over the enclosing span is cheaper.
        # The strided case then picks its frames out of that already-decompressed block,
        # which reads (num_frames - 1) * frameskip + 1 frames instead of the full window.
        if isinstance(idx, (list, np.ndarray)):
            idx = list(idx)
            if idx:
                lo, hi = idx[0], idx[-1]
                if idx == list(range(lo, hi + 1)):
                    return dataset[lo:hi + 1]
                if all(b > a for a, b in zip(idx, idx[1:])):
                    return dataset[lo:hi + 1][[i - lo for i in idx]]
        return dataset[idx]

    def __len__(self):
        return self.shape[0]


def _frame_view(raw: np.memmap, dataset: h5py.Dataset):
    """Memory-mapped view of `dataset` where the layout allows it, a lazy h5py reader otherwise."""
    view = _view_dataset(raw, dataset)
    if view is not None:
        return view
    return _H5View(dataset.file.filename, dataset.name, dataset.shape, dataset.dtype)


def _copy_with_progress(src: Path, dst: Path, size: int, chunk: int = 32 << 20) -> None:
    """Byte-for-byte copy of `src` to `dst`, ticking a progress bar as it goes.

    shutil.copyfile is marginally faster (it can hand the whole transfer to the kernel via
    copy_file_range/sendfile), but staging a multi-GB dataset off a networked filesystem runs
    for minutes, and a job that prints nothing for minutes is indistinguishable from a hung
    one. `mininterval` keeps the log readable when stderr is a file rather than a terminal,
    where tqdm cannot rewrite a line in place and every update becomes another line.
    """
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=size,
        desc=f"Copying {src.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=5.0,
    ) as bar:
        buf = bytearray(chunk)
        view = memoryview(buf)
        while True:
            n = fsrc.readinto(buf)
            if not n:
                break
            fdst.write(view[:n])
            bar.update(n)


def _stage_to_slurm_tmpdir(data_path: Path) -> Path:
    """
    Stage the dataset on node-local disk ($SLURM_TMPDIR) and return the local copy's path.

    home/project/scratch are networked filesystems, so every page the memmap views fault
    in during training crosses the network. One sequential copy up front turns the whole
    job's random reads into local ones.

    Staging is an optimization, never a requirement: whenever it isn't possible this falls
    back to reading `data_path` where it already lives, and prints why.
    """
    tmpdir = os.environ.get("SLURM_TMPDIR")
    if not tmpdir:
        return data_path

    local_dir = Path(tmpdir)
    size = data_path.stat().st_size

    # Apptainer/Singularity pass the host environment straight through but only mount the
    # paths they were told to, so inside a container $SLURM_TMPDIR is routinely set while
    # pointing at nothing. Don't create it: without a bind mount that would land in the
    # container's own writable layer, which is not the node-local disk we're after.
    if not local_dir.is_dir():
        print(
            f"Not staging data: $SLURM_TMPDIR={tmpdir} is not visible from here (in a "
            f"container, add '--bind $SLURM_TMPDIR'). Reading {data_path} directly.",
            flush=True,
        )
        return data_path

    local = local_dir / data_path.name
    if local.exists() and local.stat().st_size == size:
        print(f"Using dataset already staged at {local}", flush=True)
        return local

    free = shutil.disk_usage(local_dir).free
    if free < size:
        print(
            f"Not staging data: {data_path.name} needs {size / 1e9:.1f} GB but $SLURM_TMPDIR "
            f"has {free / 1e9:.1f} GB free. Reading {data_path} directly.",
            flush=True,
        )
        return data_path

    # Copy to a pid-unique name and rename into place, so a second process on the node
    # (another DDP rank, or a rerun) either sees the complete file or does its own copy —
    # never a half-written one.
    partial = local.with_name(f"{local.name}.{os.getpid()}.partial")
    print(
        f"Copying data: {data_path} -> {local} ({size / 1e9:.1f} GB), this may take a while...",
        flush=True,
    )
    try:
        _copy_with_progress(data_path, partial, size)
        os.replace(partial, local)
    except OSError as e:
        # Out of space, a vanished tmpdir, a concurrent rank filling the disk: none of these
        # are worth killing a GPU job over when the original file is still readable.
        partial.unlink(missing_ok=True)
        print(f"Copying data failed ({e}); reading {data_path} directly.", flush=True)
        return data_path
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print("Copying data: done", flush=True)
    return local


class ManiSkillTrajDataset(TrajDataset, abc.ABC):
    """One ManiSkill recording as fixed-length trajectories; subclass per task.

    ManiSkill trajectories store one extra observation frame per episode relative to the
    number of actions (obs_t, act_t -> obs_{t+1}). `get_seq_length()` is the action count, so
    `range(get_seq_length(idx))` naturally selects the leading obs_t frames and drops the
    trailing terminal one.
    """

    @property
    @abc.abstractmethod
    def proprio_keys(self) -> Sequence[str]:
        """Recorded fields concatenated into the flat proprio vector, in order."""

    @property
    @abc.abstractmethod
    def state_keys(self) -> Sequence[str]:
        """Recorded fields concatenated into the flat state vector, in order.

        The order is part of the contract, not an implementation detail: env wrappers slice
        these vectors back apart by offset (see env/pushcube/pushcube_wrapper.py).
        """

    def __init__(
        self,
        data_path,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        camera: Optional[str] = None,
    ):
        self.data_path = _stage_to_slurm_tmpdir(Path(data_path))
        self.transform = transform
        self.normalize_action = normalize_action

        # A single memmap covering the whole file; the per-trajectory RGB/DINO views
        # built below are cheap pointer arithmetic into it (see _view_dataset) and read
        # no data until get_frames() actually indexes them. The low-dim actions/proprio/
        # state fields stay eagerly loaded — they're small and needed up front for the
        # normalization stats and padding.
        self._raw_mmap = np.memmap(self.data_path, mode="r", dtype=np.uint8)

        with h5py.File(self.data_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            if n_rollout:
                traj_keys = traj_keys[:n_rollout]

            # All three of these are properties of how the file was written, so the first
            # trajectory settles them for the whole dataset. A file straight out of
            # replay_trajectory.py has gzipped rgb and no DINO features; the two
            # tools/preprocess_data.py flags add each independently, so all four combinations
            # occur and every one of them has to load.
            first_traj = f[traj_keys[0]]
            self.camera = _detect_camera(first_traj, camera, self.data_path)
            rgb_key = RGB_TEMPLATE.format(camera=self.camera)
            dino_key = DINO_TEMPLATE.format(camera=self.camera)

            has_dino = dino_key in first_traj
            if not has_dino:
                print(
                    f"No {dino_key!r} in {self.data_path.name} — observations will carry no "
                    "precomputed DINO features, so the encoder runs on raw images instead. "
                    f"tools/preprocess_data.py --dino-fp16 --camera {self.camera} adds them."
                )
            if first_traj[rgb_key].id.get_offset() is None:
                print(
                    f"{rgb_key!r} in {self.data_path.name} is chunked and/or compressed, so it "
                    "cannot be memory-mapped; falling back to reading through h5py. Expect "
                    "slower loading — tools/preprocess_data.py --memmap writes a mappable copy."
                )

            actions, states, proprios, seq_lengths = [], [], [], []
            rgb_views, dino_views = [], []
            for key in tqdm(traj_keys, desc="Mapping Data"):
                traj = f[key]
                actions.append(torch.from_numpy(traj["actions"][:]).float())
                proprios.append(torch.cat(
                    [torch.from_numpy(traj[k][:]).float() for k in self.proprio_keys], dim=-1
                ))
                states.append(torch.cat(
                    [torch.from_numpy(traj[k][:]).float() for k in self.state_keys], dim=-1
                ))
                seq_lengths.append(actions[-1].shape[0])
                rgb_views.append(_frame_view(self._raw_mmap, traj[rgb_key]))
                if has_dino:
                    dino_views.append(_frame_view(self._raw_mmap, traj[dino_key]))

        self.traj_keys = traj_keys
        self.rgb_views = rgb_views
        self.dino_views = dino_views if has_dino else None
        self.seq_lengths = torch.tensor(seq_lengths)

        self.actions = self._pad_stack(actions)
        self.states = self._pad_stack(states)
        self.proprios = self._pad_stack(proprios)
        self.actions = self.actions / action_scale  # scaled back up in env

        n = len(self.traj_keys)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _pad_stack(seqs):
        max_t = max(s.shape[0] for s in seqs)
        padded = [
            torch.cat([s, s.new_zeros(max_t - s.shape[0], s.shape[-1])])
            if s.shape[0] < max_t else s
            for s in seqs
        ]
        return torch.stack(padded)

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames, action_frames=None):
        """
        `frames` selects the observation/state frames to return. `action_frames`
        selects the action frames independently; the trajectory slicer passes the
        dense window there while striding `frames`, so actions can be concatenated
        across a frameskip window without paying to read the frames in between.
        Defaults to `frames`.
        """
        frames = list(frames)
        action_frames = frames if action_frames is None else list(action_frames)
        # Fancy-indexing an np.memmap view allocates a fresh, contiguous, writable array
        # and is the point where only the touched pages are read from disk (OS page cache
        # serves repeat touches, e.g. across epochs, at close to RAM speed).
        image = torch.from_numpy(self.rgb_views[idx][frames])   # THWC uint8
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, action_frames]
        state = self.states[idx, frames]

        image = image.float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": proprio,
        }
        if self.dino_views is not None:
            # T P D precomputed features
            obs["dino_patch_features"] = torch.from_numpy(
                self.dino_views[idx][frames].astype(np.float32)
            )
        return obs, act, state, {} # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_maniskill_slice_train_val(
    dataset_cls,
    transform,
    data_path,
    n_rollout=None,
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    camera=None,
):
    """Build `dataset_cls` over `data_path` and slice it into train/valid windows."""
    dset = dataset_cls(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        camera=camera,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
