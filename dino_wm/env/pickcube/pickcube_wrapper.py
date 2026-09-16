"""PickCube-v1.1 behind DINO-WM's planning interface.

Only the task-specific parts are here; see env/maniskill_wrapper.py for everything else.
"""
import numpy as np

from ..maniskill_wrapper import POSITION, ManiSkillPlanningWrapper

# ManiSkill's own numbers for PickCube-v1 with the panda (see PICK_CUBE_CONFIGS["panda"] and the
# `is_static(0.2)` in PickCubeEnv.evaluate()).
GOAL_THRESH = 0.025
STATIC_QVEL_THRESHOLD = 0.2

# The panda articulation row is [root_p(3), root_q(4), lin_vel(3), ang_vel(3), qpos(dof),
# qvel(dof)] (Articulation.get_state), so qvel starts 13 + dof in. The two gripper finger joints
# are excluded, matching Panda.is_static's `get_qvel()[..., :-2]`.
ARTICULATION_ROOT_WIDTH = 13
GRIPPER_JOINTS = 2


class PickCubeWrapper(ManiSkillPlanningWrapper):
    """Pick the cube up and hold it, still, at the goal position."""

    task_id = "PickCube-v1.1"
    """The project's own PickCube: the stock task with early termination removed, which is what
    every dataset here is now collected under. It differs from `PickCube-v1` in `terminated`
    alone -- same scene, dynamics, reward and success predicate -- so a recording made under
    either id restores and replays identically through this wrapper."""

    # These MUST stay in sync with datasets/pickcube_dset.py: the `state` vectors handed to this
    # wrapper (init_state, goal_state) are concatenations of these fields in exactly this order.
    # 28-wide, not the 25 the other tasks use: PickCube's goal marker is hidden from every camera,
    # so `goal_pos` rides along in proprio to keep the target observable. See pickcube_dset.py.
    proprio_keys = [
        "obs/agent/qpos",
        "obs/agent/qvel",
        "obs/extra/tcp_pose",
        "obs/extra/goal_pos",
    ]
    state_keys = [
        "env_states/articulations/panda",
        "env_states/actors/cube",
        "env_states/actors/goal_site",
        "env_states/actors/table-workspace",
    ]

    def evaluate_states(self, cur, goal):
        """PickCube's test: cube within goal_thresh of the goal site, arm essentially stopped.

        Unlike PushCube this is a 3D distance, not an xy one, and there is no resting condition:
        the goal site is spawned up to 0.3 m above the table, so success generally means holding
        the cube in the air rather than putting it down.

        The goal site is read from `goal` rather than `cur`, for the same reason PushCube reads
        its goal region there: it is a goal *marker* whose position defines the task, so scoring
        against the one the goal state carries is what makes `sample_random_init_goal_states`
        report success exactly when the sampled goal has been reached. Within one episode the two
        are identical anyway.

        The static term is what makes this criterion more than a distance check, and it is
        evaluated on `cur`: ManiSkill will not call an episode solved while the arm is still
        moving, so neither will this.
        """
        cube = self.actor_state(cur, "env_states/actors/cube")[..., POSITION]
        site = self.actor_state(goal, "env_states/actors/goal_site")[..., POSITION]
        cube_dist = np.linalg.norm(cube - site, axis=-1)

        panda = self.actor_state(cur, "env_states/articulations/panda")
        dof = (panda.shape[-1] - ARTICULATION_ROOT_WIDTH) // 2
        qvel = panda[..., ARTICULATION_ROOT_WIDTH + dof:][..., :dof - GRIPPER_JOINTS]
        max_qvel = np.abs(qvel).max(axis=-1)

        success = (cube_dist <= GOAL_THRESH) & (max_qvel <= STATIC_QVEL_THRESHOLD)
        return success, {"cube_dist": cube_dist, "max_qvel": max_qvel}
