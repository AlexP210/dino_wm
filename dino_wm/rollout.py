"""Roll one validation trajectory out in the simulator and in the world model, side by side.

Three things come from the config (conf/rollout.yaml): which conf/env config the simulator
is built from, which conf/dataset config the trajectories are read from, and which
checkpoint holds the world model. Everything about how the model is *assembled* --
num_hist, num_pred, frameskip, concat_dim, which encoder/decoder/predictor -- is read back
from the hydra.yaml frozen beside the checkpoint at training time, because rebuilding those
from current config defaults would quietly produce a different model than the weights were
trained for.

The output is one tiled PNG. The top row is the simulator replaying the trajectory's
recorded actions from its recorded initial state; the bottom row is the world model
imagining those same actions forward. Both rows are `horizon + 1` frames and their columns
line up frame for frame, so the bottom row's drift from the top is the model's error.

The world model is seeded from the *simulator's* first `num_hist` frames rather than the
dataset's, so the two rows start from an identical observation and every later difference
between them is world-model error alone. Sim-replay drift -- the simulator not exactly
reproducing the recording -- is reported separately rather than folded into that
comparison; on ManiSkill tasks it is the reason to prefer the physx_cpu backend.

Usage (inside the TSD container, see jobs/run_dino_wm_plan.sh for the invocation pattern):

    python rollout.py env=push_cube dataset=pushcube \
        ckpt_path=/data/.../checkpoints/model_latest.pth \
        data_dir=$DATA_DIR checkpoint_dir=$CHECKPOINT_DIR output_dir=$OUTPUT_DIR
"""

import os
import random
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import utils as vutils

import env  # noqa: F401  -- importing registers the gym ids the env configs name
from env.serial_vector_env import SerialVectorEnv
from plan import load_model
from preprocessor import Preprocessor
from utils import move_to_device, seed as set_seed


def load_train_cfg(ckpt_path):
    """The config a checkpoint was trained under, from <run_dir>/hydra.yaml."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    # <run_dir>/checkpoints/model_<epoch>.pth  ->  <run_dir>/hydra.yaml
    train_cfg_path = ckpt_path.parent.parent / "hydra.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} has no training config at {train_cfg_path}. The model is rebuilt "
            "from it, so a checkpoint without one cannot be rolled out."
        )
    return OmegaConf.load(train_cfg_path)


def check_cfg_agrees_with_checkpoint(cfg, train_cfg):
    """Fail loudly on the config values shared between this run and training.

    Both are inputs to the dataset configs here -- `img_size` to the image transform,
    `normalize_action` to the action/proprio statistics -- so a disagreement doesn't error,
    it silently feeds the model differently scaled inputs than it was trained on.
    """
    for key in ("img_size", "normalize_action"):
        here, trained = cfg[key], train_cfg[key]
        if here != trained:
            raise ValueError(
                f"{key}={here} here but the checkpoint trained with {key}={trained}. The "
                "dataset config interpolates this into the observation transform and the "
                "normalization statistics, so the model would see differently scaled "
                "inputs than it was trained on."
            )


def pick_trajectory(cfg, dset, frameskip, num_hist, start):
    """Choose a validation trajectory long enough for the requested rollout.

    Returns (traj_id, horizon). Selection goes through `get_seq_length`, which is a lookup
    into an already-resident tensor; reading candidates through `dset[i]` instead would
    decompress every validation episode's images to look at nothing but a frame count.
    """
    # A trajectory of T frames supports frames start .. start + horizon * frameskip, so
    # horizon is bounded by (T - 1 - start) // frameskip.
    def max_horizon(i):
        return (int(dset.get_seq_length(i)) - 1 - start) // frameskip

    # num_hist frames go to seeding the model, so a shorter rollout predicts nothing.
    required = num_hist if cfg.horizon is None else cfg.horizon
    candidates = [i for i in range(len(dset)) if max_horizon(i) >= required]
    if not candidates:
        longest = max(max_horizon(i) for i in range(len(dset)))
        raise ValueError(
            f"No validation trajectory supports a {required}-step rollout from frame "
            f"{start} at frameskip {frameskip}; the longest supports {longest}."
        )

    if cfg.traj_id is None:
        traj_id = random.choice(candidates)
    else:
        traj_id = cfg.traj_id
        if not 0 <= traj_id < len(dset):
            raise ValueError(
                f"traj_id={traj_id} is out of range for the {len(dset)}-episode "
                "validation split."
            )
        if max_horizon(traj_id) < required:
            raise ValueError(
                f"traj_id={traj_id} supports only a {max_horizon(traj_id)}-step rollout "
                f"from frame {start} at frameskip {frameskip}, not the {required} asked for."
            )

    horizon = max_horizon(traj_id) if cfg.horizon is None else cfg.horizon
    return traj_id, horizon


def rollout_in_sim(cfg, envs, preprocessor, init_state, dense_actions, frameskip):
    """Replay the recorded actions from the recorded initial state.

    Returns the observations and states at world-model resolution: the env is stepped once
    per recorded action, then subsampled at `frameskip` so its frames land on the same
    instants the world model predicts.
    """
    # The dataset stores actions normalized; the env expects them in their original units.
    exec_actions = preprocessor.denormalize_actions(dense_actions.unsqueeze(0)).numpy()
    sim_obses, sim_states = envs.rollout([cfg.seed], init_state[None], exec_actions)
    # rollout returns the reset frame plus one per action: horizon * frameskip + 1 frames.
    sim_obses = {k: v[:, ::frameskip] for k, v in sim_obses.items()}
    return sim_obses, sim_states[:, ::frameskip]


def rollout_in_wm(model, sim_obs, dense_actions, frameskip, num_hist, device):
    """Imagine the same actions forward from the simulator's first `num_hist` frames.

    `VWorldModel.rollout` returns one frame per action plus one, matching the subsampled
    simulator rollout, with the leading `num_hist` frames being encodings of the context
    rather than predictions.
    """
    context = {k: v[:, :num_hist] for k, v in sim_obs.items()}
    # The model consumes one action token per predicted frame, with a frameskip window's
    # worth of env actions concatenated into each.
    wm_actions = rearrange(dense_actions, "(t f) d -> t (f d)", f=frameskip)
    wm_actions = wm_actions.unsqueeze(0).to(device)
    with torch.no_grad():
        z_obses, _ = model.rollout(obs_0=context, act=wm_actions)
        visual = model.decode_obs(z_obses)[0]["visual"]
    return visual


def report_divergence(dset_visual, sim_visual, wm_visual, dset_states, sim_states, num_hist):
    """Print how far the two rows are from each other, and from the recording.

    Two different quantities, worth keeping apart: replay drift is the simulator failing to
    reproduce the recording (a property of the sim backend and the state reset), while the
    world-model error is measured against the simulator, which is what the top row actually
    shows.
    """
    replay_drift = (dset_visual - sim_visual).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    wm_error = (sim_visual - wm_visual).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    state_drift = np.linalg.norm(dset_states - sim_states, axis=-1)

    print(
        "\nPer-frame divergence (pixels in the model's [-1, 1] space, mean |difference|):\n"
        "  frame  replay drift  |state| drift   wm error"
    )
    for t in range(len(wm_error)):
        # The world model is still being fed observations over the context frames, so its
        # error there is reconstruction error, not prediction error.
        tag = " (context)" if t < num_hist else ""
        print(
            f"  {t:>5}  {replay_drift[t]:>12.4f}  {state_drift[t]:>13.4f}  "
            f"{wm_error[t]:>9.4f}{tag}"
        )
    print(
        f"  mean   {replay_drift.mean():>12.4f}  {state_drift.mean():>13.4f}  "
        f"{wm_error.mean():>9.4f}"
    )
    print(f"  predicted frames only: wm error {wm_error[num_hist:].mean():.4f}")


def save_comparison(sim_visual, wm_visual, filename):
    """One tiled image: simulator on the top row, world model on the bottom."""
    # save_image fills rows left to right, so concatenating the two rollouts along time and
    # setting nrow to the rollout length puts each on its own row with columns aligned.
    imgs = torch.cat([sim_visual.cpu(), wm_visual.cpu()], dim=0)
    vutils.save_image(
        imgs,
        filename,
        nrow=sim_visual.shape[0],
        normalize=True,
        value_range=(-1, 1),
    )
    return os.path.abspath(filename)


@hydra.main(config_path="conf", config_name="rollout")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt_path = Path(cfg.ckpt_path)
    train_cfg = load_train_cfg(ckpt_path)
    check_cfg_agrees_with_checkpoint(cfg, train_cfg)
    frameskip, num_hist = train_cfg.frameskip, train_cfg.num_hist

    if cfg.dataset.data_path != train_cfg.env.dataset.data_path:
        print(
            "WARNING: rolling out on a different recording than the checkpoint trained on.\n"
            f"  dataset={cfg.dataset._target_}: {cfg.dataset.data_path}\n"
            f"  checkpoint trained on:          {train_cfg.env.dataset.data_path}\n"
            "  Action and proprio normalization statistics are computed from the recording "
            "loaded here, so they no longer match the ones the model was trained with."
        )

    # The trajectory datasets, not the sliced windows the trainer consumes: this replays
    # whole episodes, and the split is the same deterministic one training used.
    _, traj_dsets = hydra.utils.call(
        cfg.dataset,
        num_hist=num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=frameskip,
    )
    dset = traj_dsets["valid"]

    model = load_model(ckpt_path, train_cfg, train_cfg.num_action_repeat, device=device)
    model.eval()
    if model.decoder is None:
        raise ValueError(
            f"{ckpt_path} has no decoder, so its rollouts cannot be turned back into "
            "images. Train with has_decoder=True, or point env.decoder_path at one."
        )

    preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )

    start = 0 if cfg.start is None else cfg.start
    traj_id, horizon = pick_trajectory(cfg, dset, frameskip, num_hist, start)
    n_steps = horizon * frameskip  # env steps, before subsampling to model resolution
    print(
        f"Rolling out validation trajectory {traj_id} of {len(dset)}: frames "
        f"{start}..{start + n_steps} at frameskip {frameskip} "
        f"({horizon} world-model steps, {num_hist} of them context)"
    )

    obs, act, state, env_info = dset[traj_id]
    dset_visual = obs["visual"][start : start + n_steps + 1 : frameskip].to(device)
    dense_actions = act[start : start + n_steps]
    state = state.numpy()
    dset_states = state[start : start + n_steps + 1 : frameskip]

    # One env, driven serially. SubprocVectorEnv forks, and CUDA is already initialized in
    # this process by the world model, which a forked SAPIEN/torch CUDA context cannot
    # survive -- the same reason plan.py routes several of these envs through SerialVectorEnv.
    envs = SerialVectorEnv([gym.make(cfg.env.name, *cfg.env.args, **dict(cfg.env.kwargs))])
    envs.update_env([env_info])

    sim_obses, sim_states = rollout_in_sim(
        cfg, envs, preprocessor, state[start], dense_actions, frameskip
    )
    sim_obs = move_to_device(preprocessor.transform_obs(sim_obses), device)
    wm_visual = rollout_in_wm(model, sim_obs, dense_actions, frameskip, num_hist, device)

    report_divergence(
        dset_visual, sim_obs["visual"][0], wm_visual[0], dset_states, sim_states[0], num_hist
    )
    path = save_comparison(
        sim_obs["visual"][0],
        wm_visual[0],
        f"rollout_traj{traj_id}_start{start}_h{horizon}.png",
    )
    print(f"\nTop row: simulator. Bottom row: world model.\nWrote {path}")
    for e in envs.envs:
        e.close()


if __name__ == "__main__":
    main()
