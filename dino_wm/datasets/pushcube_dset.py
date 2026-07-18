import h5py
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
from typing import Optional, Callable, Any

# ManiSkill PushCube-v1 trajectories store one extra observation frame per
# episode relative to the number of actions (obs_t, act_t -> obs_{t+1}).
# get_seq_length() is set to the action count, so range(get_seq_length(idx))
# naturally selects the leading obs_t frames and drops the trailing terminal one.
PROPRIO_KEYS = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/cube",
    "env_states/actors/goal_region",
    "env_states/actors/table-workspace",
]
RGB_KEY = "obs/sensor_data/base_camera/rgb"
DINO_KEY = "obs/sensor_data/base_camera/dino_patch_features"


class PushBlockDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "/data/AlexPleava/datasets/maniskill/PushCube-v1/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        with h5py.File(self.data_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            if n_rollout:
                traj_keys = traj_keys[:n_rollout]

            actions, states, proprios, seq_lengths = [], [], [], []
            for key in traj_keys:
                traj = f[key]
                actions.append(torch.from_numpy(traj["actions"][:]).float())
                proprios.append(torch.cat(
                    [torch.from_numpy(traj[k][:]).float() for k in PROPRIO_KEYS], dim=-1
                ))
                states.append(torch.cat(
                    [torch.from_numpy(traj[k][:]).float() for k in STATE_KEYS], dim=-1
                ))
                seq_lengths.append(actions[-1].shape[0])

        self.traj_keys = traj_keys
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

    def get_frames(self, idx, frames):
        frames = list(frames)
        with h5py.File(self.data_path, "r") as f:
            image = f[self.traj_keys[idx]][RGB_KEY][frames]  # THWC uint8
            dino = f[self.traj_keys[idx]][DINO_KEY][frames]  # T P D precomputed features
        image = torch.from_numpy(image)
        dino = torch.from_numpy(dino).float()
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        image = image.float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": proprio,
            "dino_patch_features": dino,
        }
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

def load_pushcube_slice_train_val(
    transform,
    n_rollout=None,
    data_path="/data/AlexPleava/datasets/maniskill/PushCube-v1/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = PushBlockDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
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
