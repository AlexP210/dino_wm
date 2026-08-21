"""PushCube-v1.1 behind DINO-WM's planning interface.

Only the task-specific parts are here; see env/maniskill_wrapper.py for everything else.
"""
import numpy as np

from ..maniskill_wrapper import POSITION, ManiSkillPlanningWrapper

# ManiSkill's own numbers for PushCube-v1 (see PushCubeEnv: goal_radius, cube_half_size).
GOAL_RADIUS = 0.1
CUBE_HALF_SIZE = 0.02


class PushCubeWrapper(ManiSkillPlanningWrapper):
    """Push the cube into the goal region."""

    task_id = "PushCube-v1.1"
    """The project's own PushCube: the stock task with early termination removed, which is what
    every dataset here is now collected under. It differs from `PushCube-v1` in `terminated`
    alone -- same scene, dynamics, reward and success predicate -- so a recording made under
    either id restores and replays identically through this wrapper, and DINO-WM's planning
    never reads `done` anyway (`rollout` discards what `step_multiple` returns)."""

    # These MUST stay in sync with datasets/pushcube_dset.py: the `state` vectors handed to this
    # wrapper (init_state, goal_state) are concatenations of these fields in exactly this order.
    proprio_keys = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
    state_keys = [
        "env_states/articulations/panda",
        "env_states/actors/cube",
        "env_states/actors/goal_region",
        "env_states/actors/table-workspace",
    ]

    def evaluate_states(self, cur, goal):
        """PushCube's test: cube inside the goal region in xy, still resting on the table.

        The goal region is read from `goal` rather than `cur`, because it is a goal *marker*
        whose position defines the task: scoring a rollout against the region its goal state
        carries is what makes `sample_random_init_goal_states` report success exactly when the
        sampled goal has been reached. Within one episode the two are identical anyway.
        """
        cube = self.actor_state(cur, "env_states/actors/cube")[..., POSITION]
        region = self.actor_state(goal, "env_states/actors/goal_region")[..., POSITION]

        cube_dist = np.linalg.norm(cube[..., :2] - region[..., :2], axis=-1)
        resting = cube[..., 2] < CUBE_HALF_SIZE + 5e-3
        return (cube_dist < GOAL_RADIUS) & resting, {"cube_dist": cube_dist}
