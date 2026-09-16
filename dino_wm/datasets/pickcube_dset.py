"""PickCube-v1 / PickCube-v1.1 trajectories.

Only the flat-vector layout is here; everything about how a ManiSkill recording is read lives in
maniskill_dset.py.
"""
from .maniskill_dset import ManiSkillTrajDataset, load_maniskill_slice_train_val

# The arm and its tcp are the same as PushCube's, plus `goal_pos` -- 9 + 9 + 7 + 3 = 28, the one
# task here whose proprio is not the plain 25.
#
# PickCube's goal marker is hidden from the cameras (ManiSkill appends `goal_site` to
# `_hidden_objects`, so it is invisible in every observation), and under the wrist view a grasped
# cube looks nearly the same wherever the arm is. `goal_pos` is the recorded goal_site position, so
# carrying it in proprio is what keeps the target available to the model at all.
#
# Note what this does NOT fix: goal_pos is exactly constant within an episode, and every
# `goal_source` here draws init and goal from the same episode, so these 3 dims are identical in the
# current and goal observations and contribute no gradient to the planning objective. They matter
# for anything predicting reward/success/value from observations, not for goal-image MPC.
PROPRIO_KEYS = [
    "obs/agent/qpos",
    "obs/agent/qvel",
    "obs/extra/tcp_pose",
    "obs/extra/goal_pos",
]

# These MUST stay in sync with env/pickcube/pickcube_wrapper.py, which slices the same vectors
# back apart by offset.
#
# PickCube's goal actor is `goal_site` -- a sphere floating at the target position -- where
# PushCube's is a flat `goal_region` decal on the table. The state is therefore as wide as
# PushCube's (31 + 13 * 3 = 70) while naming a different actor, so the two recordings are
# interchangeable by width and distinguishable only by these keys: getting them wrong mis-slices
# a recording rather than failing, which is why the loader looks the fields up by name.
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/cube",
    "env_states/actors/goal_site",
    "env_states/actors/table-workspace",
]


class PickCubeDataset(ManiSkillTrajDataset):
    """A PickCube recording: the panda, the cube, its goal site and the table (70-wide state).

    Proprio is 28-wide here, not the 25 the other tasks use; see PROPRIO_KEYS.
    """

    proprio_keys = PROPRIO_KEYS
    state_keys = STATE_KEYS


def load_pickcube_slice_train_val(
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
        PickCubeDataset,
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
