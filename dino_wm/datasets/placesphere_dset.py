"""PlaceSphere-v1.1 trajectories.

Only the flat-vector layout is here; everything about how a ManiSkill recording is read lives in
maniskill_dset.py.
"""
from .maniskill_dset import ManiSkillTrajDataset, load_maniskill_slice_train_val

# The arm and its tcp are the same as PushCube's, so proprio is unchanged (9 + 9 + 7 = 25).
#
# PlaceSphere records two `obs/extra` fields the other tasks do not, both deliberately left out:
# `bin_pos` is the goal location (already in the state vector as env_states/actors/bin, and giving
# the policy the goal for free is a decision, not a default), and `is_grasped` is privileged
# contact information rather than something the arm's own sensors report. Adding either also needs
# care -- `is_grasped` is a scalar per step, shape (T,), which the base class concatenates along
# the last axis and would have to be unsqueezed first.
PROPRIO_KEYS = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]

# PlaceSphere's scene holds the sphere to be placed and the bin it goes into, so unlike
# LiftPegUpright there is a goal actor. 31 + 13 + 13 + 13 = 70 wide, the same width as PushCube's
# state but not the same contents -- loading one task's recording with another's keys raises
# rather than silently mis-slicing, since the actor names differ.
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/sphere",
    "env_states/actors/bin",
    "env_states/actors/table-workspace",
]


class PlaceSphereDataset(ManiSkillTrajDataset):
    """A PlaceSphere recording: the panda, the sphere, its bin and the table (70-wide state)."""

    proprio_keys = PROPRIO_KEYS
    state_keys = STATE_KEYS


def load_placesphere_slice_train_val(
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
        PlaceSphereDataset,
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
