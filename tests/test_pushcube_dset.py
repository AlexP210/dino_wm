"""ManiSkillTrajDataset against both storage layouts: `python tests/test_pushcube_dset.py`.

The fixtures are written here rather than pointing at a recording, so this runs anywhere and
covers what a real dataset directory usually cannot: the gzipped layout that comes straight out
of tools/replay_trajectory.py and the contiguous one tools/preprocess_data.py --memmap writes,
side by side, plus DINO features which only exist after a --dino run.
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dino_wm"))

from datasets.maniskill_dset import ManiSkillTrajDataset, _H5View  # noqa: E402
from datasets.pushcube_dset import PushBlockDataset  # noqa: E402

T, RES, PATCHES, DIM = 12, 8, 4, 6   # steps, image size, dino patches, dino width
FIXTURES = tempfile.mkdtemp(prefix="pushcube_dset_")


PUSHCUBE_ACTORS = ("cube", "goal_region", "table-workspace")
LIFTPEG_ACTORS = ("peg", "table-workspace")
PLACESPHERE_ACTORS = ("sphere", "bin", "table-workspace")


def _write(path, camera="hand_camera", compress=False, dino=False, images=True, episodes=2,
           actors=PUSHCUBE_ACTORS):
    """One h5 in the schema replay_trajectory.py produces, in either storage layout."""
    rng = np.random.RandomState(0)
    with h5py.File(path, "w") as f:
        for e in range(episodes):
            g = f.create_group(f"traj_{e}")
            g.create_dataset("actions", data=rng.rand(T, 4).astype(np.float32))
            g.create_dataset("rewards", data=rng.rand(T).astype(np.float32))
            for name in ("terminated", "truncated", "success"):
                g.create_dataset(name, data=np.zeros(T, dtype=bool))
            g.create_dataset("obs/agent/qpos", data=rng.rand(T + 1, 9).astype(np.float32))
            g.create_dataset("obs/agent/qvel", data=rng.rand(T + 1, 9).astype(np.float32))
            g.create_dataset("obs/extra/tcp_pose", data=rng.rand(T + 1, 7).astype(np.float32))
            for actor in actors:
                g.create_dataset(f"env_states/actors/{actor}",
                                 data=rng.rand(T + 1, 13).astype(np.float32))
            g.create_dataset("env_states/articulations/panda",
                             data=rng.rand(T + 1, 31).astype(np.float32))
            if images:
                # frame t is filled with the value t, so a returned frame identifies itself
                rgb = np.tile(
                    np.arange(T + 1, dtype=np.uint8).reshape(-1, 1, 1, 1), (1, RES, RES, 3))
                kwargs = {"compression": "gzip", "chunks": (1, RES, RES, 3)} if compress else {}
                g.create_dataset(f"obs/sensor_data/{camera}/rgb", data=rgb, **kwargs)
                if dino:
                    g.create_dataset(
                        f"obs/sensor_data/{camera}/dino_patch_features",
                        data=np.full((T + 1, PATCHES, DIM), 0.5, dtype=np.float16))
    return path


def test_both_layouts_load_and_agree():
    """The point of the exercise: gzipped and contiguous files are interchangeable."""
    gz = _write(os.path.join(FIXTURES, "gzipped.h5"), compress=True)
    mm = _write(os.path.join(FIXTURES, "contiguous.h5"), compress=False)

    a = PushBlockDataset(data_path=gz, normalize_action=True)
    b = PushBlockDataset(data_path=mm, normalize_action=True)

    # both code paths really were taken, rather than both falling back to the same one.
    # The mapped view is a plain ndarray over the whole-file memmap, not an np.memmap itself.
    assert isinstance(a.rgb_views[0], _H5View), type(a.rgb_views[0])
    assert not isinstance(b.rgb_views[0], _H5View), type(b.rgb_views[0])
    assert isinstance(b.rgb_views[0].base, np.memmap), type(b.rgb_views[0].base)

    for frames in ([0, 1, 2, 3], [0, 4, 8], [T - 1]):
        obs_a, act_a, state_a, _ = a.get_frames(0, frames)
        obs_b, act_b, state_b, _ = b.get_frames(0, frames)
        assert torch.equal(obs_a["visual"], obs_b["visual"]), frames
        assert torch.equal(obs_a["proprio"], obs_b["proprio"]), frames
        assert torch.equal(act_a, act_b) and torch.equal(state_a, state_b), frames
        # each frame is filled with its own index, so this pins the frames actually returned
        got = [int(round(float(v) * 255)) for v in obs_a["visual"][:, 0, 0, 0]]
        assert got == list(frames), (got, frames)


def test_strided_reads_and_independent_action_frames():
    """What TrajSlicerDataset asks for under frameskip: strided obs, dense actions."""
    ds = PushBlockDataset(data_path=_write(os.path.join(FIXTURES, "strided.h5"), compress=True))
    obs, act, state, _ = ds.get_frames(0, range(0, 9, 4), action_frames=range(0, 9))
    assert obs["visual"].shape == (3, 3, RES, RES), obs["visual"].shape
    assert act.shape == (9, 4), act.shape
    assert [int(round(float(v) * 255)) for v in obs["visual"][:, 0, 0, 0]] == [0, 4, 8]


def test_camera_is_detected_and_can_be_overridden():
    base = _write(os.path.join(FIXTURES, "base_cam.h5"), camera="base_camera")
    assert PushBlockDataset(data_path=base).camera == "base_camera"
    wrist = _write(os.path.join(FIXTURES, "wrist_cam.h5"), camera="hand_camera")
    assert PushBlockDataset(data_path=wrist).camera == "hand_camera"
    assert PushBlockDataset(data_path=wrist, camera="hand_camera").camera == "hand_camera"

    try:
        PushBlockDataset(data_path=wrist, camera="base_camera")
    except ValueError as error:
        assert "no rgb for camera 'base_camera'" in str(error), error
    else:
        raise AssertionError("a camera the file does not have must be rejected")


def test_dino_features_come_from_the_same_camera_as_the_images():
    """The bug this replaces: rgb read from hand_camera, features looked for on base_camera."""
    path = _write(os.path.join(FIXTURES, "with_dino.h5"), camera="hand_camera", dino=True)
    ds = PushBlockDataset(data_path=path)
    assert ds.dino_views is not None, "features under the detected camera must be found"
    obs, _, _, _ = ds.get_frames(0, [0, 2])
    assert obs["dino_patch_features"].shape == (2, PATCHES, DIM)
    # stored fp16, handed out fp32, which is what the encoder consumes
    assert obs["dino_patch_features"].dtype == torch.float32

    without = PushBlockDataset(data_path=_write(os.path.join(FIXTURES, "no_dino.h5")))
    assert without.dino_views is None
    obs, _, _, _ = without.get_frames(0, [0, 2])
    assert "dino_patch_features" not in obs


def test_each_task_overrides_only_its_key_lists():
    """Every task shares the machinery; they differ in their actor set alone."""
    from datasets.liftpeg_dset import LiftPegDataset
    from datasets.placesphere_dset import PlaceSphereDataset

    cases = [
        (PushBlockDataset, PUSHCUBE_ACTORS, "pc_state.h5", 31 + 13 * 3),   # panda cube goal table
        (LiftPegDataset, LIFTPEG_ACTORS, "lp_state.h5", 31 + 13 * 2),      # panda peg table
        (PlaceSphereDataset, PLACESPHERE_ACTORS, "ps_state.h5", 31 + 13 * 3),  # panda sphere bin table
    ]
    loaded = {}
    for cls, actors, name, expected_state in cases:
        ds = cls(data_path=_write(os.path.join(FIXTURES, name), actors=actors))
        assert ds.state_dim == expected_state, (cls.__name__, ds.state_dim)
        assert ds.proprio_dim == 9 + 9 + 7 and ds.action_dim == 4, cls.__name__
        # the machinery is genuinely shared, not re-implemented per task
        assert type(ds).get_frames is PushBlockDataset.get_frames, cls.__name__
        obs, act, state, _ = ds.get_frames(0, [0, 2])
        assert obs["visual"].shape == (2, 3, RES, RES)
        assert state.shape == (2, ds.state_dim)
        loaded[cls] = ds

    # PlaceSphere's state happens to be as wide as PushCube's; the keys still must not cross,
    # or a mismatched recording would be mis-sliced instead of rejected
    assert loaded[PlaceSphereDataset].state_dim == loaded[PushBlockDataset].state_dim
    for cls, other_file in ((PushBlockDataset, "ps_state.h5"), (PlaceSphereDataset, "pc_state.h5")):
        try:
            cls(data_path=os.path.join(FIXTURES, other_file))
        except KeyError:
            pass
        else:
            raise AssertionError(f"{cls.__name__} must reject {other_file}")


def test_a_task_module_must_declare_both_key_lists():
    """The base class is abstract in the enforced sense, not just by convention."""
    for cls, missing in ((ManiSkillTrajDataset, "proprio_keys"), (_HalfDeclared, "state_keys")):
        try:
            cls(data_path=os.path.join(FIXTURES, "gzipped.h5"))
        except TypeError as error:
            assert missing in str(error), (cls.__name__, error)
        else:
            raise AssertionError(f"{cls.__name__} should not be instantiable")


class _HalfDeclared(ManiSkillTrajDataset):
    proprio_keys = ["obs/agent/qpos"]     # and no state_keys


def test_state_only_recording_is_rejected_with_a_useful_message():
    path = _write(os.path.join(FIXTURES, "state_only.h5"), images=False)
    try:
        PushBlockDataset(data_path=path)
    except ValueError as error:
        assert "state-only recording" in str(error) and "replay_trajectory" in str(error), error
    else:
        raise AssertionError("a recording with no images must be rejected")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
