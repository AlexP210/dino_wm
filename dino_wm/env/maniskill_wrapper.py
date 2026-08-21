"""Adapting a ManiSkill task to the planning interface DINO-WM expects, task-agnostically.

Everything here is shared by every ManiSkill task: building the env through
`custom_maniskill_tasks.make_env`, the flat-state-vector <-> nested-state-dict mapping,
observation extraction, the gym/gymnasium impedance match, and the demonstrated-goal index.

A task module supplies three things by subclassing `ManiSkillPlanningWrapper`: its task id, the
recorded fields making up its flat state and proprio vectors, and its success criterion. See
env/pushcube/, env/liftpeg/, env/placesphere/.

Two impedance mismatches this bridges:

- ManiSkill is a *gymnasium* env returning batched torch tensors (leading dim num_envs) and a
  5-tuple from step(); DINO-WM's planning stack is old-style *gym*, expects a 4-tuple, and expects
  unbatched numpy. Everything is squeezed and converted at this boundary.
- DINO-WM addresses simulator state as one flat vector, because that is what `eval_state` diffs
  and what `prepare` restores. ManiSkill addresses it as a nested {actors, articulations} dict.
  The flat<->nested mapping is derived at __init__ from the live env's own state dict (rather than
  hardcoding widths), so it stays correct if a task's actor set ever changes -- but the *order* is
  pinned to `state_keys`, since that is the order the dataset concatenated them in.
"""
import abc

import h5py
import numpy as np
import gym
import torch

# The env is built by the project's central definition (environments/custom_maniskill_tasks),
# which also registers the -v1.1 task ids and is what the dataset recorder, the trajectory
# converter and TSD's online task build their envs with. Importing it registers those ids.
from custom_maniskill_tasks import (
    DEFAULT_CAMERA_RESOLUTION,
    FOCUSED_CAMERA_UID,
    WRIST_CAMERA_UID,
    make_env,
)

from utils import aggregate_dct

CAMERA_VIEW = "wrist"
"""Which of the central camera views the observations come from; the dataset must have been
recorded through the same one. The DINO features the world model was trained on were rendered
from that viewpoint at that resolution, so planning against a differently-posed camera silently
puts every goal image and every encoder input off-distribution -- this is not a free parameter,
it is a property of the checkpoint being planned with."""

OBSERVATION_CAMERA = {
    "wrist": WRIST_CAMERA_UID,
    "focused": FOCUSED_CAMERA_UID,
    "default": FOCUSED_CAMERA_UID,
    "standard": FOCUSED_CAMERA_UID,
}
"""Which sensor `_get_obs` reads for each view. The wrist view drops the task's own camera, so
this cannot be hardcoded to base_camera."""

CAMERA_RESOLUTION = DEFAULT_CAMERA_RESOLUTION

# Actor and articulation state rows are [p(3), q(4), linear_velocity(3), angular_velocity(3)],
# so a task's success criterion can read pose and velocity straight out of the flat vector.
POSITION = slice(0, 3)
QUATERNION = slice(3, 7)
LINEAR_VELOCITY = slice(7, 10)
ANGULAR_VELOCITY = slice(10, 13)


def _get_nested(d, path):
    for key in path.split("/"):
        d = d[key]
    return d


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


class ManiSkillPlanningWrapper(gym.Env, abc.ABC):
    """One ManiSkill task behind DINO-WM's planning interface; subclass per task.

    Provides `prepare` / `step_multiple` / `rollout` / `eval_state` /
    `sample_random_init_goal_states` / `update_env`, matching PushTWrapper and PointMazeWrapper.

    Args:
        task_id: which registered task to build; defaults to the subclass's `task_id`.
        camera_view: which camera the observations come from; see `CAMERA_VIEW`. Must match the
            recording the world model was trained on.
        sim_backend: "physx_cpu" (default) or "physx_cuda". Two reasons for the default,
            neither of them "match the recording" -- the datasets were in fact recorded on
            physx_cuda. First, planning instantiates n_evals *separate* single-env instances in
            one process (see plan.py), and ManiSkill's GPU sim is not built to host several
            independent scenes that way. Second, measured on PushCube: replaying recorded actions
            from a restored state tracks the recorded cube trajectory ~10x more closely on
            physx_cpu (final cube xy error 0.019 vs 0.206 over 3 episodes) -- the recording ran
            at num_envs=4096, and physx_cuda at num_envs=1 does not reproduce that batched
            solver's behavior. State restoration is exact (~1e-8) and rendering matches the
            recording on either backend, so this affects dynamics only.
        reconfiguration_freq: 0 (default) builds the scene once and only re-initializes on reset.
            The datasets used 1, but these tasks randomize nothing at reconfigure time (only
            object *poses*, at episode init), so 0 is equivalent here and much faster per reset.
        goal_data_path: recorded-trajectory .h5 that `sample_random_init_goal_states` draws its
            init/goal pairs from -- i.e. the same file the world model trained on. Only read
            under `goal_source: 'random_state'`, and only on first use, so leaving it None costs
            nothing for `goal_source: 'dset'` planning.
        goal_split / goal_split_ratio / goal_split_seed: which partition of that file goals may
            come from. Defaults match the dataset modules' own split, so goals land on held-out
            episodes. Pass goal_split=None to use every episode.
    """

    metadata = {"render.modes": ["rgb_array"]}

    @property
    @abc.abstractmethod
    def task_id(self) -> str:
        """The registered ManiSkill task id this wrapper drives, e.g. "PushCube-v1.1"."""

    @property
    @abc.abstractmethod
    def state_keys(self):
        """Recorded fields concatenated into the flat state vector, in order.

        Must match the task's dataset module (datasets/*_dset.py), since the init/goal state
        vectors handed to this wrapper are concatenations of these fields in exactly this order.
        """

    @property
    @abc.abstractmethod
    def proprio_keys(self):
        """Recorded observation fields concatenated into the flat proprio vector, in order."""

    @abc.abstractmethod
    def evaluate_states(self, cur, goal):
        """The task's success criterion, evaluated on flat state vectors.

        Both arguments are (..., state_dim); implementations broadcast over the leading axis so
        one call serves a single frame (`eval_state`) or a whole episode (the goal index).

        Returns (success, metrics): a boolean array, and a dict of task-specific scalars to
        report alongside it -- the distance a human should actually read, since the full-state
        L2 `eval_state` also reports is dominated by the panda's 31 joint dimensions.

        Implementations read the *current* state for the objects the robot moves. Whether the
        goal geometry comes from `cur` or `goal` is per-task and documented there: it matters
        only for goals drawn from a different episode than the rollout.
        """

    def __init__(
        self,
        sim_backend: str = "physx_cpu",
        reconfiguration_freq: int = 0,
        control_mode: str = "pd_ee_delta_pos",
        obs_mode: str = "rgb",
        reward_mode: str = "normalized_dense",
        render_size: int = CAMERA_RESOLUTION,
        task_id: str = None,
        camera_view: str = CAMERA_VIEW,
        goal_data_path: str = None,
        goal_split: str = "valid",
        goal_split_ratio: float = 0.8,
        goal_split_seed: int = 42,
    ):
        # frame_skip=1 and no frame stacking on purpose: DINO-WM does its own chunking, in the
        # planner (`rearrange(actions, "b (t f) d -> b t (f d)")`) and in TrajSlicerDataset, and
        # flattens back to primitive actions before calling step_multiple. The wrappers
        # `make_env` can add would double-apply that.
        self._env = make_env(
            task_id or self.task_id,
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            camera_view=camera_view,
            camera_resolution=render_size,
            sim_backend=sim_backend,
            reward_mode=reward_mode,
            reconfiguration_freq=reconfiguration_freq,
            frame_skip=1,
            n_frames=None,
        )
        self._base = self._env.unwrapped
        self.render_size = render_size
        self._camera_uid = OBSERVATION_CAMERA[camera_view]
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
        for key in self.state_keys:
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

    def actor_state(self, states, key):
        """The rows of `states` belonging to one actor/articulation, as (..., width).

        Use with the POSITION/QUATERNION/LINEAR_VELOCITY/ANGULAR_VELOCITY slices to write a
        success criterion against a flat state vector.
        """
        start, end = self._state_slices[key]
        return np.asarray(states)[..., start:end]

    def _get_state(self):
        state_dict = self._base.get_state_dict()
        parts = [
            _to_numpy(_get_nested(state_dict, key.removeprefix("env_states/")))[0]
            for key in self.state_keys
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
        visual = _to_numpy(obs["sensor_data"][self._camera_uid]["rgb"])[0]  # (H, W, C) uint8
        proprio = np.concatenate(
            [_to_numpy(_get_nested(obs, key.removeprefix("obs/")))[0]
             for key in self.proprio_keys],
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
        No-op: these dataset modules return an empty env_info dict, because unlike PushT
        (whose block shape varies per trajectory) every trajectory of a given ManiSkill task
        shares one scene configuration. Everything that does vary is carried in the state vector.
        """
        pass

    def _build_demo_goal_index(self, data_path):
        """
        Scan the recorded trajectories once and keep, for each episode that satisfies this
        task's success condition at some frame after the first, that episode's initial state
        and the stack of its success-frame states.

        Only those two things are retained (not the full state tracks), so the index stays a
        few MB regardless of how much of the file qualifies.

        Every recorded state frame is a candidate goal, including the trailing terminal one
        that the dataset modules drop -- that frame is dropped there because it has no successor
        *action*, which a goal does not need.

        The criterion is recomputed from the recorded states rather than read from the file's
        own per-step `success` flag. Verified equivalent: over 450 sampled steps spanning the
        random, mid-training and expert stages of the PushCube, LiftPegUpright and PlaceSphere
        recordings, the two agree on every step.
        """
        with h5py.File(data_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            episodes = []
            for i in _split_episode_indices(
                len(traj_keys), self._goal_split, self._goal_split_ratio,
                self._goal_split_seed,
            ):
                traj = f[traj_keys[i]]
                state = np.concatenate(
                    [traj[k][:] for k in self.state_keys], axis=-1
                ).astype(np.float32)
                if state.shape[-1] != self.state_dim:
                    raise ValueError(
                        f"{data_path}: episode {traj_keys[i]} has state width "
                        f"{state.shape[-1]}, but this env's flat state is {self.state_dim} "
                        "wide. The recording and the live env disagree on state_keys."
                    )
                ok, _ = self.evaluate_states(state, state)
                ok = np.asarray(ok).copy()
                # A goal equal to the start would be trivially satisfied. In practice no task
                # here spawns already-solved, so this only guards against a future change to
                # the reset distribution.
                ok[0] = False
                if not ok.any():
                    continue
                episodes.append((state[0].copy(), state[ok].copy()))

        if not episodes:
            raise ValueError(
                f"{data_path}: no episode in the {self._goal_split or 'full'} split ever "
                f"reaches {self.task_id}'s success condition, so there are no demonstrated "
                "goals to sample."
            )
        return episodes

    def _demo_goal_index(self):
        """Lazily built, process-wide cached; see _build_demo_goal_index."""
        if self._goal_data_path is None:
            raise ValueError(
                f"{type(self).__name__}.sample_random_init_goal_states needs goal_data_path, "
                "the recorded .h5 it draws demonstrated goals from. Planning passes it "
                "through from env.dataset.data_path (see plan.py); set it in the task's "
                "conf/env/*.yaml under `kwargs` if you are constructing the env yourself."
            )
        key = (
            type(self).__name__,
            str(self._goal_data_path),
            self._goal_split,
            self._goal_split_ratio,
            self._goal_split_seed,
        )
        if key not in _DEMO_GOAL_INDEX_CACHE:
            _DEMO_GOAL_INDEX_CACHE[key] = self._build_demo_goal_index(self._goal_data_path)
        return _DEMO_GOAL_INDEX_CACHE[key]

    def sample_random_init_goal_states(self, seed):
        """
        Return two states: one initial, one goal, both lifted from a single recorded
        demonstration that actually solves the task.

        Draws a uniformly random episode among those that reach the task's success condition at
        some point, takes that episode's first state as the initial state, and a uniformly
        random one of its success frames as the goal.

        Two properties this buys over synthesizing the goal:

        - The goal is a state the simulator actually produced, with the arm wherever the
          demonstrator left it. A synthetic goal pairs a solved object configuration with an arm
          still in its reset pose, which is off the data manifold the DINO encoder was trained
          on, so its embedding is a poor target for the planner to descend toward.
        - Init and goal come from the same episode, hence share the task's goal geometry, so
          `eval_state` reports success exactly when the goal has been reached. Under
          `goal_source: 'dset'` those two criteria come apart, because the goal is whatever the
          demo happened to be doing goal_H world-model steps in.

        Note this makes the goal a solved state rather than a fixed horizon away: the distance
        from frame 0 to a success frame varies per episode, so difficulty varies across evals
        and is not controlled by goal_H.
        """
        index = self._demo_goal_index()
        rs = np.random.RandomState(seed)
        init_state, goal_states = index[rs.randint(len(index))]
        goal_state = goal_states[rs.randint(len(goal_states))]
        return init_state.copy(), goal_state.copy()

    def eval_state(self, goal_state, cur_state):
        """The task's success condition plus the distances worth reading, for one frame."""
        success, metrics = self.evaluate_states(cur_state, goal_state)
        return {
            "success": bool(np.asarray(success).reshape(-1)[0]),
            # The full-state distance is dominated by the panda's 31 joint dims and says little
            # about the task; the task-specific metrics beside it are the ones to read.
            "state_dist": float(np.linalg.norm(goal_state - cur_state)),
            **{name: float(np.asarray(value).reshape(-1)[0]) for name, value in metrics.items()},
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
