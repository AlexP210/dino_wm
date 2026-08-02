import h5py
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


def _cube_success(cube_xy, cube_z, target_xy):
    """
    PushCube-v1's success test: cube inside the goal region in xy, still resting on the
    table. Shared by eval_state and the demo-goal index below so the criterion the goals
    are selected by and the criterion they are scored by cannot drift apart.

    Broadcasts over a leading time axis, so it takes either single frames or whole tracks.
    Returns (success, cube_dist).
    """
    cube_dist = np.linalg.norm(np.asarray(cube_xy) - np.asarray(target_xy), axis=-1)
    success = (cube_dist < GOAL_RADIUS) & (np.asarray(cube_z) < CUBE_HALF_SIZE + 5e-3)
    return success, cube_dist


def _split_episode_indices(n, split, split_ratio, split_seed):
    """
    Reproduce the train/valid partition that datasets/traj_dset.py::split_traj_datasets
    applies to this same file, so goals can be drawn from episodes the world model was
    not trained on. Pinned to that implementation: a torch.randperm over the
    integer-sorted trajectory order, first int(split_ratio * n) to train.

    split=None returns every episode.
    """
    if split is None:
        return list(range(n))
    if split not in ("train", "valid"):
        raise ValueError(f"split must be 'train', 'valid' or None, got {split!r}")
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    cut = int(split_ratio * n)
    return perm[:cut] if split == "train" else perm[cut:]


# Cache of the per-episode goal index, keyed by everything that determines its contents.
# Planning builds n_evals separate wrapper instances inside one process (see
# SerialVectorEnv in plan.py), and without this every one of them would re-scan the same
# ~20k-episode file.
_DEMO_GOAL_INDEX_CACHE = {}


def _build_demo_goal_index(
    data_path, state_dim, cube_start, goal_start, split, split_ratio, split_seed
):
    """
    Scan the recorded trajectories once and keep, for each episode that satisfies
    PushCube's success condition at some frame after the first, that episode's initial
    state and the stack of its success-frame states.

    Only those two things are retained (not the full state tracks), so the index stays a
    few MB regardless of how much of the file qualifies.

    Every recorded state frame is a candidate goal, including the trailing terminal one
    that datasets/pushcube_dset.py drops — that frame is dropped there because it has no
    successor *action*, which a goal does not need.
    """
    with h5py.File(data_path, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        episodes = []
        for i in _split_episode_indices(
            len(traj_keys), split, split_ratio, split_seed
        ):
            traj = f[traj_keys[i]]
            state = np.concatenate(
                [traj[k][:] for k in STATE_KEYS], axis=-1
            ).astype(np.float32)
            if state.shape[-1] != state_dim:
                raise ValueError(
                    f"{data_path}: episode {traj_keys[i]} has state width "
                    f"{state.shape[-1]}, but this env's flat state is {state_dim} wide. "
                    "The recording and the live env disagree on STATE_KEYS."
                )
            ok, _ = _cube_success(
                state[:, cube_start : cube_start + 2],
                state[:, cube_start + 2],
                state[:, goal_start : goal_start + 2],
            )
            # A goal equal to the start would be trivially satisfied. In practice the cube
            # never spawns inside the goal region, so this only guards against a future
            # change to the reset distribution.
            ok[0] = False
            if not ok.any():
                continue
            episodes.append((state[0].copy(), state[ok].copy()))

    if not episodes:
        raise ValueError(
            f"{data_path}: no episode in the {split or 'full'} split ever reaches "
            "PushCube's success condition, so there are no demonstrated goals to sample."
        )
    return episodes


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
        goal_data_path: recorded-trajectory .h5 that `sample_random_init_goal_states`
            draws its init/goal pairs from — i.e. the same file the world model trained
            on. Only read under `goal_source: 'random_state'`, and only on first use, so
            leaving it None costs nothing for `goal_source: 'dset'` planning.
        goal_split / goal_split_ratio / goal_split_seed: which partition of that file
            goals may come from. Defaults match datasets/pushcube_dset.py's own split, so
            goals land on held-out episodes. Pass goal_split=None to use every episode.
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
        goal_data_path: str = None,
        goal_split: str = "valid",
        goal_split_ratio: float = 0.8,
        goal_split_seed: int = 42,
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

        self._goal_data_path = goal_data_path
        self._goal_split = goal_split
        self._goal_split_ratio = goal_split_ratio
        self._goal_split_seed = goal_split_seed

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

    def _demo_goal_index(self):
        """Lazily built, process-wide cached; see _build_demo_goal_index."""
        if self._goal_data_path is None:
            raise ValueError(
                "PushCubeWrapper.sample_random_init_goal_states needs goal_data_path, "
                "the recorded .h5 it draws demonstrated goals from. Planning passes it "
                "through from env.dataset.data_path (see plan.py); set it in "
                "conf/env/push_cube.yaml under `kwargs` if you are constructing the env "
                "yourself."
            )
        cube_start, _ = self._state_slices["env_states/actors/cube"]
        goal_start, _ = self._state_slices["env_states/actors/goal_region"]
        key = (
            str(self._goal_data_path),
            self._goal_split,
            self._goal_split_ratio,
            self._goal_split_seed,
        )
        if key not in _DEMO_GOAL_INDEX_CACHE:
            _DEMO_GOAL_INDEX_CACHE[key] = _build_demo_goal_index(
                self._goal_data_path,
                self.state_dim,
                cube_start,
                goal_start,
                self._goal_split,
                self._goal_split_ratio,
                self._goal_split_seed,
            )
        return _DEMO_GOAL_INDEX_CACHE[key]

    def sample_random_init_goal_states(self, seed):
        """
        Return two states: one initial, one goal, both lifted from a single recorded
        demonstration that actually solves the task.

        Draws a uniformly random episode among those that reach PushCube's success
        condition at some point, takes that episode's first state as the initial state,
        and a uniformly random one of its success frames as the goal.

        Two properties this buys over synthesizing the goal (which is what this used to
        do — copy the init state and teleport the cube to a random point inside the goal
        region):

        - The goal is a state the simulator actually produced, with the arm wherever the
          demonstrator left it. A synthetic goal pairs a pushed cube with an arm still in
          its reset pose, which is off the data manifold the DINO encoder was trained on,
          so its embedding is a poor target for the planner to descend toward.
        - Init and goal come from the same episode, hence share a goal region, so
          `eval_state` — which measures the cube against the *goal state's* goal region —
          reports success exactly when the goal has been reached. Under
          `goal_source: 'dset'` those two criteria come apart, because the goal is
          whatever the demo happened to be doing goal_H world-model steps in.

        Note this makes the goal a solved state rather than a fixed horizon away: the
        distance from frame 0 to a success frame varies per episode, so difficulty varies
        across evals and is not controlled by goal_H.
        """
        index = self._demo_goal_index()
        rs = np.random.RandomState(seed)
        init_state, goal_states = index[rs.randint(len(index))]
        goal_state = goal_states[rs.randint(len(goal_states))]
        return init_state.copy(), goal_state.copy()

    def eval_state(self, goal_state, cur_state):
        """
        ManiSkill's PushCube-v1 success condition (cube within goal_radius of the target
        in xy, and still resting on the table), evaluated against the *goal's* target
        position so it stays meaningful for goals sampled by this wrapper.
        """
        cube_start, _ = self._state_slices["env_states/actors/cube"]
        goal_start, _ = self._state_slices["env_states/actors/goal_region"]

        success, cube_dist = _cube_success(
            cur_state[cube_start : cube_start + 2],
            cur_state[cube_start + 2],
            goal_state[goal_start : goal_start + 2],
        )
        success, cube_dist = bool(success), float(cube_dist)
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
