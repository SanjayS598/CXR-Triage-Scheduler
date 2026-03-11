import os
import yaml
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from rl_environment.envs.gym_env import HospitalGymEnv


def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg   = cfg.get("simulation", {})
    train_cfg = cfg.get("training", {})

    seed = sim_cfg.get("random_seed", 42)
    env = Monitor(HospitalGymEnv.from_yaml(config_path), filename="runs/train_monitor")
    eval_env = Monitor(HospitalGymEnv.from_yaml(config_path), filename="runs/eval_monitor")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.get("n_steps", 2048),
        save_path="checkpoints/",
        name_prefix="ppo_hospital",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/best/",
        log_path="runs/eval/",
        eval_freq=train_cfg.get("n_steps", 2048) * 5,  # evaluate every 5 rollouts
        n_eval_episodes=5,
        deterministic=True,
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        n_steps=train_cfg.get("n_steps", 2048),
        batch_size=train_cfg.get("batch_size", 64),
        n_epochs=train_cfg.get("n_epochs", 10),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_range=train_cfg.get("clip_range", 0.2),
        tensorboard_log="runs/tensorboard/",
        seed=seed,
        verbose=1,
    )

    print(f"Training PPO for {train_cfg.get('total_timesteps', 100000)} timesteps...")
    print(f"Config: {config_path}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model.learn(
        total_timesteps=train_cfg.get("total_timesteps", 100000),
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_hospital",
    )

    model.save("checkpoints/ppo_hospital_final")
    print("Training complete. Model saved to checkpoints/ppo_hospital_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/poc.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(args.config)
