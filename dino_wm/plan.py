import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # 'dset_success' picks its segments deliberately (see
            # sample_success_boundary_segments); 'dset' and 'random_action' leave the
            # trajectory and offset to the uniform draw inside the sampler.
            segments = (
                self.sample_success_boundary_segments()
                if self.goal_source == "dset_success"
                else None
            )
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(
                    traj_len=self.frameskip * self.goal_H + 1, segments=segments
                )
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions
            if self.goal_source == "dset_success":
                self.report_goal_is_success()

    def sample_traj_segment_from_dset(self, traj_len, segments=None):
        """
        Args:
            traj_len: number of observation/state frames each segment must contain.
            segments: optional list of (traj_id, offset) of length n_evals, chosen by the
                caller. When None, both are drawn uniformly at random, which is the
                'dset' / 'random_action' behavior. 'dset_success' passes the pairs its
                own boundary search selected.
        """
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough. Skipped when the caller chose the
        # segments: it already established their trajectories are long enough, and this
        # check reads every validation trajectory's images in full (through
        # self.dset[i]) to look at nothing but a frame count -- tens of GB of
        # decompression on a recording like PushCube's.
        if segments is None:
            valid_traj = [
                self.dset[i][0]["visual"].shape[0]
                for i in range(len(self.dset))
                if self.dset[i][0]["visual"].shape[0] >= traj_len
            ]
            if len(valid_traj) == 0:
                raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            if segments is None:
                max_offset = -1
                while max_offset < 0:  # filter out traj that are not long enough
                    traj_id = random.randint(0, len(self.dset) - 1)
                    obs, act, state, e_info = self.dset[traj_id]
                    max_offset = obs["visual"].shape[0] - traj_len
                offset = random.randint(0, max_offset)
            else:
                traj_id, offset = segments[i]
                obs, act, state, e_info = self.dset[traj_id]
                if obs["visual"].shape[0] < offset + traj_len:
                    raise ValueError(
                        f"segment ({traj_id}, {offset}) needs {traj_len} frames but "
                        f"trajectory {traj_id} has {obs['visual'].shape[0]}."
                    )
            state = state.numpy()
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def _traj_state_track(self, traj_id):
        """Raw (T, state_dim) state track of one trajectory, without reading its images.

        Going through self.dset[traj_id] would decompress and transform that trajectory's
        entire RGB track, which is ruinous when scanning every validation trajectory.
        `states` is a padded (N, T, D) tensor already resident on the underlying dataset,
        and -- unlike actions and proprios -- it is left unnormalized, which is what the
        env's success predicate expects. TrajSubset forwards the attribute but not the
        index remapping, hence the explicit indices lookup.
        """
        return self.dset.states[
            self.dset.indices[traj_id], : int(self.dset.get_seq_length(traj_id))
        ].numpy()

    def sample_success_boundary_segments(self):
        """
        Choose (traj_id, offset) pairs whose segment straddles the task's success
        boundary: its first frame fails the success predicate and its last frame passes
        it, so the crossing falls strictly inside the segment.

        This is the one regime where both properties hold at once:

        - The goal is exactly frameskip * goal_H steps away, reachable by construction,
          with `gt_actions` a known solution -- the short-horizon guarantee 'dset' gives.
        - The goal is a *solved* state, so eval_state's success flag measures whether the
          goal was reached. Under plain 'dset' the goal is wherever the demo happened to
          be goal_H steps in, which is usually not solved, and the two criteria come apart
          (see ManiSkillPlanningWrapper.evaluate_states).

        Episodes are weighted equally rather than by how many crossings each contains,
        matching how sample_random_init_goal_states draws an episode and then a frame.

        Note this draws through `random`, so unlike the init/goal pair under
        'random_state' -- keyed to eval_seed, which is 1 regardless of cfg.seed at
        n_evals=1 -- the segment here does respond to cfg.seed.
        """
        if not hasattr(self.env, "success_track"):
            raise ValueError(
                "goal_source='dset_success' needs an env exposing a state-space success "
                "predicate (SerialVectorEnv.success_track over the ManiSkill wrappers' "
                f"evaluate_states); {type(self.env).__name__} for '{self.env_name}' has "
                "none, so there is no success boundary to straddle."
            )

        seg_len = self.frameskip * self.goal_H  # steps between the segment's two ends
        offsets_per_traj = {}
        n_crossings = 0
        for traj_id in range(len(self.dset)):
            ok = self.env.success_track(self._traj_state_track(traj_id))
            end = np.arange(seg_len, len(ok))
            crossing = end[ok[end] & ~ok[end - seg_len]]
            if len(crossing) > 0:
                offsets_per_traj[traj_id] = (crossing - seg_len).tolist()
                n_crossings += len(crossing)

        if not offsets_per_traj:
            raise ValueError(
                f"No trajectory in the {len(self.dset)}-episode split contains a segment "
                f"of {seg_len} steps that starts unsolved and ends solved, so nothing "
                f"straddles {self.env_name}'s success boundary. goal_H={self.goal_H} at "
                f"frameskip={self.frameskip} is likely too long or too short for this task."
            )

        traj_ids = sorted(offsets_per_traj)
        print(
            f"dset_success: {n_crossings} boundary-straddling segments of {seg_len} steps "
            f"across {len(traj_ids)}/{len(self.dset)} episodes"
        )
        segments = []
        for _ in range(self.n_evals):
            traj_id = traj_ids[random.randint(0, len(traj_ids) - 1)]
            offsets = offsets_per_traj[traj_id]
            segments.append((traj_id, offsets[random.randint(0, len(offsets) - 1)]))
        print(f"dset_success: planning (traj_id, offset) = {segments}")
        return segments

    def report_goal_is_success(self):
        """
        Check the *replayed* goal still passes the predicate its segment was chosen for.

        The segment is selected on the recorded state track, but the goal handed to the
        planner is the last frame of replaying the recorded actions in the live sim (see
        prepare_targets), and the two differ by the sim's replay error. That matters
        precisely here: a segment ending on its episode's first success frame sits right
        at the predicate's threshold, so a small drift can push the goal back outside it
        -- quietly restoring the goal/success mismatch this mode exists to remove. Later
        crossing frames have the object further inside and are not at risk.

        Reported rather than resampled: the segment is only unusable if the drift actually
        crossed back, and which evals that happened to is what a reader needs to see.
        """
        eval_g = self.env.eval_state(self.state_g, self.state_g)
        success = np.asarray(eval_g["success"]).reshape(-1)
        for i in range(self.n_evals):
            # state_dist is the goal against itself here, i.e. identically 0; the task's
            # own metrics say how far inside the threshold the goal actually sits.
            detail = ", ".join(
                f"{name}={np.asarray(value).reshape(-1)[i]:.4f}"
                for name, value in eval_g.items()
                if name not in ("success", "state_dist")
            )
            print(f"dset_success: eval {i} goal solved={bool(success[i])}  {detail}")
        if not success.all():
            print(
                f"dset_success: WARNING -- {int((~success).sum())}/{self.n_evals} replayed "
                "goals fail the success predicate, so for those evals the final success "
                "flag no longer measures goal-reaching. Replay drift moved the object back "
                "across the threshold; re-run with a different seed."
            )

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device(cfg_dict["device"] if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    env_kwargs = dict(model_cfg.env.kwargs)
    if model_cfg.env.name == "push_cube":
        # push_cube draws goal_source='random_state' goals from the recorded trajectories
        # rather than synthesizing them. Checkpoints trained before goal_data_path existed
        # have no such entry in their frozen hydra.yaml, and it is always the file the
        # model trained on, so default it rather than making old checkpoints unplannable.
        env_kwargs.setdefault("goal_data_path", model_cfg.env.dataset.data_path)

    # use dummy vector env for wall and deformable envs, and for push_cube: SubprocVectorEnv
    # forks, and by this point the parent has already initialized CUDA (the world model is
    # on the GPU), which a forked SAPIEN/torch CUDA context cannot survive.
    if model_cfg.env.name in ("wall", "deformable_env", "push_cube"):
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(model_cfg.env.name, *model_cfg.env.args, **env_kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **env_kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
