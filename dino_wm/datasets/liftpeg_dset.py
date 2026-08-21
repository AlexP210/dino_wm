"""LiftPegUpright-v1.1 trajectories.

Only the flat-vector layout is here; everything about how a ManiSkill recording is read lives in
maniskill_dset.py.
"""
from .maniskill_dset import ManiSkillTrajDataset, load_maniskill_slice_train_val

# The arm and its tcp are the same as PushCube's, so proprio is unchanged (9 + 9 + 7 = 25).
PROPRIO_KEYS = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]

# LiftPegUpright's scene holds a peg instead of a cube and, unlike PushCube, no goal_region:
# success is the peg standing upright rather than reaching a marked area, so there is no goal
# actor to record. The flat state is therefore 31 + 13 + 13 = 57 wide, not PushCube's 70.
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/peg",
    "env_states/actors/table-workspace",
]


class LiftPegDataset(ManiSkillTrajDataset):
    """A LiftPegUpright recording: the panda, the peg and the table (57-wide state)."""

    proprio_keys = PROPRIO_KEYS
    state_keys = STATE_KEYS


def load_liftpeg_slice_train_val(
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
    return load_maniskill_slice_train_val(
        LiftPegDataset,
        transform=transform,
        data_path=data_path,
        n_rollout=n_rollout,
        normalize_action=normalize_action,
        split_ratio=split_ratio,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        camera=camera,
    )
