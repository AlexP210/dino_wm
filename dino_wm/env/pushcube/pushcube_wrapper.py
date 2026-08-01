import numpy as np
import gym
import gymnasium
import torch

import mani_skill.envs  # noqa: F401  (registers PushCube-v1 with gymnasium)
from mani_skill.utils import sapien_utils

from utils import aggregate_dct

# These MUST stay in sync with datasets/pushcube_dset.py: `state` and `proprio` vectors
# handed to this wrapper (init_state, goal_state) are concatenations of the recorded h5
# fields in exactly this order, so the wrapper has to slice them back apart the same way.
PROPRIO_KEYS = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/cube",
    "env_states/actors/goal_region",
    "env_states/actors/table-workspace",
]

# Camera used to record the dataset (see the `sensor_configs` block of the trajectory
# .json, and tsd/tasks/maniskill_task.py::make_env). The DINO features the world model
# was trained on were rendered from *this* viewpoint at *this* resolution — planning
# against a differently-posed camera silently puts every goal image and every encoder
# input off-distribution, so these are not free parameters.
CAMERA_EYE = [0.3, 0, 0.9]
CAMERA_TARGET = [-0.1, 0, -0.3]
CAMERA_FOV = np.pi / 5
CAMERA_RESOLUTION = 224

# ManiSkill's own success criterion for PushCube-v1 (see PushCubeEnv.evaluate).
GOAL_RADIUS = 0.1
CUBE_HALF_SIZE = 0.02


def _get_nested(d, path):
    for key in path.split("/"):
        d = d[key]
    return d


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class PushCubeWrapper(gym.Env):
    """
    Adapts ManiSkill's PushCube-v1 to the planning interface DINO-WM expects
    (`prepare` / `step_multiple` / `rollout` / `eval_state` /
    `sample_random_init_goal_states` / `update_env`), matching PushTWrapper and
    PointMazeWrapper.

    Two impedance mismatches this bridges:

    - ManiSkill is a *gymnasium* env returning batched torch tensors (leading dim
      num_envs) and a 5-tuple from step(); DINO-WM's planning stack is old-style *gym*,
      expects a 4-tuple, and expects unbatched numpy. Everything is squeezed and
      converted at this boundary.
    - DINO-WM addresses simulator state as one flat vector, because that is what
      `eval_state` diffs and what `prepare` restores. ManiSkill addresses it as a nested
      {actors, articulations} dict. The flat<->nested mapping is derived at __init__ from
      the live env's own state dict (rather than hardcoding the 31/13/13/13 split), so it
      stays correct if the task's actor set ever changes — but the *order* is pinned to
      STATE_KEYS, since that is the order the dataset concatenated them in.

    Args:
        sim_backend: "physx_cpu" (default) or "physx_cuda". Two reasons for the default,
            neither of them "match the recording" — the dataset was in fact recorded on
            physx_cuda. First, planning instantiates n_evals *separate* single-env
            instances in one process (see plan.py), and ManiSkill's GPU sim is not built
            to host several independent scenes that way. Second, measured: replaying
            recorded actions from a restored state tracks the recorded cube trajectory
            ~10x more closely on physx_cpu (final cube xy error 0.019 vs 0.206 over 3
            episodes) — the recording ran at num_envs=4096, and physx_cuda at num_envs=1
            does not reproduce that batched solver's behavior. State restoration is exact
            (~1e-8) and rendering matches the recording on either backend, so this
            affects dynamics only.
        reconfiguration_freq: 0 (default) builds the scene once and only re-initializes
            on reset. The dataset used 1, but PushCube randomizes nothing at reconfigure
            time (only object *poses*, at episode init), so 0 is equivalent here and much
            faster per reset.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        sim_backend: str = "physx_cpu",
        reconfiguration_freq: int = 0,
        control_mode: str = "pd_ee_delta_pos",
        obs_mode: str = "rgb",
        reward_mode: str = "normalized_dense",
        render_size: int = CAMERA_RESOLUTION,
    ):
        camera_pose = sapien_utils.look_at(eye=CAMERA_EYE, target=CAMERA_TARGET)
        self._env = gymnasium.make(
            "PushCube-v1",
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            sim_backend=sim_backend,
            reconfiguration_freq=reconfiguration_freq,
            render_mode="rgb_array",
            sensor_configs=dict(
                width=render_size,
                height=render_size,
                fov=CAMERA_FOV,
                pose=camera_pose,
            ),
        )
        self._base = self._env.unwrapped
        self.render_size = render_size
        self._seed = None

        # A reset is required before the sim state dict is populated, and we need that
        # dict to derive the flat-state layout below.
        self._env.reset(seed=0)

        self._state_slices = {}
        offset = 0
        state_dict = self._base.get_state_dict()
        for key in STATE_KEYS:
            width = _get_nested(state_dict, key.removeprefix("env_states/")).shape[-1]
            self._state_slices[key] = (offset, offset + width)
            offset += width
        self.state_dim = offset

        # single_action_space is the per-env action space regardless of num_envs; the
        # batched `action_space` collapses to the same thing at num_envs=1, so reading it
        # instead would silently break if this ever ran batched.
        ms_action_space = self._base.single_action_space
        self.action_dim = int(np.prod(ms_action_space.shape))
        self.action_space = gym.spaces.Box(
            low=ms_action_space.low.reshape(-1),
            high=ms_action_space.high.reshape(-1),
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        # Advertised for gym's benefit; the planning stack consumes the dict returned by
        # _get_obs() directly and never samples from this.
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8
        )

    # ------------------------------------------------------------------ #
    # flat state vector <-> ManiSkill nested state dict
    # ------------------------------------------------------------------ #

    def _get_state(self):
        state_dict = self._base.get_state_dict()
        parts = [
            _to_numpy(_get_nested(state_dict, key.removeprefix("env_states/")))[0]
            for key in STATE_KEYS
        ]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        nested = {"actors": {}, "articulations": {}}
        for key, (start, end) in self._state_slices.items():
            path = key.removeprefix("env_states/")
            group, name = path.split("/")
            nested[group][name] = torch.as_tensor(
                state[start:end], device=self._base.device
            ).unsqueeze(0)
        self._base.set_state_dict(nested)

    def _get_obs(self):
        """
        Re-reads observations from the current sim state. Called after _set_state, where
        it matters that ManiSkill's get_obs() re-runs update_render()/capture, so the rgb
        reflects the state we just wrote rather than the pre-set-state frame.
        """
        obs = self._base.get_obs()
        visual = _to_numpy(obs["sensor_data"]["base_camera"]["rgb"])[0]  # (H, W, C) uint8
        proprio = np.concatenate(
            [_to_numpy(_get_nested(obs, key.removeprefix("obs/")))[0] for key in PROPRIO_KEYS],
            axis=-1,
        ).astype(np.float32)
        return {"visual": visual, "proprio": proprio}

    # ------------------------------------------------------------------ #
    # DINO-WM planning interface
    # ------------------------------------------------------------------ #

    def seed(self, seed=None):
        self._seed = None if seed is None else int(seed)
        return [self._seed]

    def update_env(self, env_info):
        """
        No-op: PushBlockDataset returns an empty env_info dict, because unlike PushT
        (whose block shape varies per trajectory) every PushCube-v1 trajectory shares one
        scene configuration. Everything that does vary is carried in the state vector.
        """
        pass

    def sample_random_init_goal_states(self, seed):
        """
        Return two states: one initial, one goal.

        The initial state is drawn from the task's own reset distribution. The goal is
        that same state with only the cube translated to a uniformly random point inside
        the goal region — which is exactly the set of states PushCube counts as success.
        Sampling the goal from the *reset* distribution instead (the obvious analogue of
        what PushTWrapper does) would produce goals where the cube has not been pushed at
        all, which are not meaningful targets for this task.
        """
        rs = np.random.RandomState(seed)
        self._env.reset(seed=int(seed))
        init_state = self._get_state()

        cube_start, _ = self._state_slices["env_states/actors/cube"]
        goal_start, _ = self._state_slices["env_states/actors/goal_region"]

        goal_state = init_state.copy()
        radius, angle = GOAL_RADIUS * np.sqrt(rs.uniform()), rs.uniform(0, 2 * np.pi)
        goal_state[cube_start] = init_state[goal_start] + radius * np.cos(angle)
        goal_state[cube_start + 1] = init_state[goal_start + 1] + radius * np.sin(angle)
        goal_state[cube_start + 2] = CUBE_HALF_SIZE
        # A resting cube: zero out the 6 linear/angular velocity components.
        goal_state[cube_start + 7 : cube_start + 13] = 0.0
        return init_state, goal_state

    def eval_state(self, goal_state, cur_state):
        """
        ManiSkill's PushCube-v1 success condition (cube within goal_radius of the target
        in xy, and still resting on the table), evaluated against the *goal's* target
        position so it stays meaningful for goals sampled by this wrapper.
        """
        cube_start, _ = self._state_slices["env_states/actors/cube"]
        goal_start, _ = self._state_slices["env_states/actors/goal_region"]

        cube_xy = cur_state[cube_start : cube_start + 2]
        target_xy = goal_state[goal_start : goal_start + 2]
        cube_z = cur_state[cube_start + 2]

        cube_dist = float(np.linalg.norm(cube_xy - target_xy))
        success = bool(cube_dist < GOAL_RADIUS and cube_z < CUBE_HALF_SIZE + 5e-3)
        return {
            "success": success,
            "state_dist": float(np.linalg.norm(goal_state - cur_state)),
            # The full-state distance above is dominated by the panda's 31 joint dims and
            # says little about the task; this is the metric to actually read.
            "cube_dist": cube_dist,
        }

    def reset(self):
        obs, _ = self._env.reset(seed=self._seed)
        return self._get_obs(), self._get_state()

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state.
        obs: dict of (H W C) visual and (D,) proprio
        state: (state_dim,)
        """
        self.seed(seed)
        self._env.reset(seed=self._seed)
        self._set_state(init_state)
        return self._get_obs(), self._get_state()

    def step(self, action):
        action = torch.as_tensor(
            np.asarray(action, dtype=np.float32), device=self._base.device
        ).unsqueeze(0)
        _, reward, terminated, _, info = self._env.step(action)
        obs = self._get_obs()
        state = self._get_state()
        return (
            obs,
            float(_to_numpy(reward).reshape(-1)[0]),
            bool(_to_numpy(terminated).reshape(-1)[0]),
            {"state": state},
        )

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T+1, H, W, C)
        states: (T+1, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states

    def render(self, mode="rgb_array"):
        return _to_numpy(self._env.render())[0]

    def close(self):
        self._env.close()
