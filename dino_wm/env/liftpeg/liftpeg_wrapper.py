"""LiftPegUpright-v1.1 behind DINO-WM's planning interface.

Only the task-specific parts are here; see env/maniskill_wrapper.py for everything else.
"""
import numpy as np
import torch
from mani_skill.utils.geometry import rotation_conversions

from ..maniskill_wrapper import POSITION, QUATERNION, ManiSkillPlanningWrapper

# ManiSkill's own numbers for LiftPegUpright-v1 (see LiftPegUprightEnv: peg_half_length, and the
# 0.08 rad / 0.005 m tolerances in its evaluate()).
PEG_HALF_LENGTH = 0.12
UPRIGHT_TOLERANCE = 0.08
HEIGHT_TOLERANCE = 0.005


class LiftPegUprightWrapper(ManiSkillPlanningWrapper):
    """Stand the peg upright on the table."""

    task_id = "LiftPegUpright-v1.1"

    # These MUST stay in sync with datasets/liftpeg_dset.py.
    proprio_keys = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
    state_keys = [
        "env_states/articulations/panda",
        "env_states/actors/peg",
        "env_states/actors/table-workspace",
    ]

    def evaluate_states(self, cur, goal):
        """LiftPegUpright's test: the peg standing on end, its centre a peg-half-length up.

        `goal` is unused: unlike PushCube there is no goal actor to score against -- standing the
        peg upright is a property of the peg alone, and it is the same criterion whichever goal
        state the planner was aiming at.

        The euler conversion goes through ManiSkill's own `rotation_conversions` on the recorded
        quaternion, so the criterion cannot drift from `LiftPegUprightEnv.evaluate` through a
        hand-rolled quaternion-to-angle formula.
        """
        peg = self.actor_state(cur, "env_states/actors/peg")

        quaternion = torch.as_tensor(np.ascontiguousarray(peg[..., QUATERNION]),
                                     dtype=torch.float32)
        euler = rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(quaternion), "XYZ"
        ).numpy()
        tilt = np.abs(np.abs(euler[..., 2]) - np.pi / 2)
        height_error = np.abs(peg[..., POSITION][..., 2] - PEG_HALF_LENGTH)

        success = (tilt < UPRIGHT_TOLERANCE) & (height_error < HEIGHT_TOLERANCE)
        return success, {"peg_tilt": tilt, "peg_height_error": height_error}
