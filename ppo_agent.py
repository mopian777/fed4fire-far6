from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from collections import OrderedDict
import torch


def build_ppo(env, seed=1):
    return PPO(
        "MlpPolicy",
        Monitor(env),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
        seed=seed,
    )


def get_state_dict(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.policy.state_dict().items())


def set_state_dict(model, state):
    device = model.device
    moved = OrderedDict((k, v.to(device)) for k, v in state.items())
    model.policy.load_state_dict(moved, strict=True)
