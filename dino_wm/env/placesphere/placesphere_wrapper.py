"""PlaceSphere-v1.1 behind DINO-WM's planning interface.

Only the task-specific parts are here; see env/maniskill_wrapper.py for everything else.
"""
import numpy as np

from ..maniskill_wrapper import (
    ANGULAR_VELOCITY,
    LINEAR_VELOCITY,
    POSITION,
    ManiSkillPlanningWrapper,
)

# ManiSkill's own numbers for PlaceSphere-v1 (see PlaceSphereEnv: radius, block_half_size[0] =
# short_side_half_size, the 0.005 m tolerances in its evaluate(), and the is_static thresholds).
SPHERE_RADIUS = 0.02
BIN_FLOOR_HALF_THICKNESS = 0.0025
XY_TOLERANCE = 0.005
Z_TOLERANCE = 0.005
LINEAR_VELOCITY_THRESHOLD = 1e-2
ANGULAR_VELOCITY_THRESHOLD = 0.5


class PlaceSphereWrapper(ManiSkillPlanningWrapper):
    """Place the sphere into the shallow bin."""

    task_id = "PlaceSphere-v1.1"

    # These MUST stay in sync with datasets/placesphere_dset.py.
    proprio_keys = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
    state_keys = [
        "env_states/articulations/panda",
        "env_states/actors/sphere",
        "env_states/actors/bin",
        "env_states/actors/table-workspace",
    ]

    def evaluate_states(self, cur, goal):
        """PlaceSphere's test: the sphere resting in the bin and no longer moving.

        Both the sphere and the bin come from `cur`, matching `PlaceSphereEnv.evaluate`, which
        reads the live bin: the bin is part of the scene the robot must not disturb rather than a
        goal marker, so `goal` is unused.

        One term of ManiSkill's predicate is dropped, and cannot be recovered here:
        `~is_grasped` needs contact forces, which a recorded state vector does not carry. It
        makes this marginally permissive in principle -- a sphere held motionless in the bin
        would count. Measured on the recordings this is moot: across 450 steps spanning the
        random, mid-training and expert stages, this criterion agrees with the file's own
        per-step `success` flag on every step (a gripped sphere is not static in practice).
        """
        sphere = self.actor_state(cur, "env_states/actors/sphere")
        bin_ = self.actor_state(cur, "env_states/actors/bin")

        offset = sphere[..., POSITION] - bin_[..., POSITION]
        xy_dist = np.linalg.norm(offset[..., :2], axis=-1)
        z_error = np.abs(offset[..., 2] - SPHERE_RADIUS - BIN_FLOOR_HALF_THICKNESS)
        in_bin = (xy_dist <= XY_TOLERANCE) & (z_error <= Z_TOLERANCE)

        static = (
            (np.linalg.norm(sphere[..., LINEAR_VELOCITY], axis=-1) <= LINEAR_VELOCITY_THRESHOLD)
            & (np.linalg.norm(sphere[..., ANGULAR_VELOCITY], axis=-1)
               <= ANGULAR_VELOCITY_THRESHOLD)
        )
        return in_bin & static, {"sphere_xy_dist": xy_dist, "sphere_z_error": z_error}
