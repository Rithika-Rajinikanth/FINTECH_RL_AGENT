"""
CreditLens — PPO Training Script
Trains a PPO agent using Stable-Baselines3 on the CreditLens gymnasium environment.
100% free — no cloud services, no API keys.

Usage:
    python -m creditlens.rl.train_ppo --task easy --timesteps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from creditlens.env.engine import CreditLensGymEnv

MODELS_DIR = Path("artifacts/rl_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("artifacts/tensorboard_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def train(
    task_id: str = "easy",
    timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    clip_range: float = 0.2,
    seed: int = 42,
) -> PPO:
    """Train PPO agent on CreditLens task."""

    logger.info(f"Training PPO | task={task_id} | timesteps={timesteps}")

    # Vectorised environment (4 parallel envs for faster training)
    def make_env():
        env = CreditLensGymEnv(task_id=task_id)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=4, seed=seed)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 100)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
            activation_fn=__import__("torch").nn.ReLU,
        ),
        tensorboard_log=str(LOGS_DIR / task_id),
        verbose=1,
        seed=seed,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / task_id),
        log_path=str(MODELS_DIR / task_id),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODELS_DIR / task_id / "checkpoints"),
        name_prefix="ppo_creditlens",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(str(MODELS_DIR / task_id / "final_model"))
    logger.info(f"Model saved to {MODELS_DIR / task_id / 'final_model'}")

    return model


# ─────────────────────────────────────────────
# Optuna Hyperparameter Search
# ─────────────────────────────────────────────

def hyperopt(task_id: str = "easy", n_trials: int = 20, timesteps: int = 50_000):
    """Run Optuna hyperparameter search."""

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

        model = train(
            task_id=task_id,
            timesteps=timesteps,
            learning_rate=lr,
            n_steps=n_steps,
            clip_range=clip_range,
        )

        # Evaluate
        eval_env = CreditLensGymEnv(task_id=task_id)
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward

        return total_reward

    study = optuna.create_study(direction="maximize", study_name=f"creditlens_{task_id}")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best reward: {study.best_value:.4f}")
    return study.best_params


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditLens PPO Training")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--hyperopt", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hyperopt:
        best = hyperopt(task_id=args.task, n_trials=args.trials, timesteps=50_000)
        print(f"\nBest hyperparameters: {best}")
    else:
        train(task_id=args.task, timesteps=args.timesteps, seed=args.seed)
