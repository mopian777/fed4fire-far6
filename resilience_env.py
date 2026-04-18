import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ResilienceEnv(gym.Env):
    def __init__(
        self,
        csv_path,
        latency_sla=30.0,
        loss_sla=0.10,
        throughput_min=2.0,
        episode_len=200,
        disturbance_prob=0.10,
        disturbance_strength=0.15,
        seed=1,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.latency_sla = latency_sla
        self.loss_sla = loss_sla
        self.throughput_min = throughput_min
        self.episode_len = episode_len
        self.disturbance_prob = disturbance_prob
        self.disturbance_strength = disturbance_strength
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5,), dtype=np.float32)

        self.ptr = 0
        self.step_count = 0
        self.offered_load = 0.7
        self.queue_level = 0.2

    def _get_row(self, idx):
        row = self.df.iloc[idx % len(self.df)]
        return float(row["latency_ms"]), float(row["packet_loss"]), float(row["throughput_mbps"])

    def _obs(self, latency, loss, throughput):
        return np.array([
            latency / max(self.latency_sla, 1e-6),
            loss / max(self.loss_sla, 1e-6),
            throughput / max(self.throughput_min, 1e-6),
            self.offered_load,
            self.queue_level
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_hi = max(1, len(self.df) - self.episode_len)
        self.ptr = int(self.rng.integers(0, start_hi))
        self.step_count = 0
        self.offered_load = 0.7
        self.queue_level = 0.2
        l, p, t = self._get_row(self.ptr)
        return self._obs(l, p, t), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        a1, a2, a3 = [float(x) for x in action]

        latency, loss, throughput = self._get_row(self.ptr)

        load_noise = self.rng.normal(0.0, 0.03)
        self.offered_load = float(np.clip(0.7 + load_noise, 0.4, 1.0))
        offered_load_mbps = self.offered_load * 10.0

        # a1: scheduler_weight
        latency *= (1.0 - 0.15 * max(a1, 0.0) + 0.05 * max(-a1, 0.0))

        # a2: redundancy_ratio
        loss *= (1.0 - 0.20 * max(a2, 0.0) + 0.08 * max(-a2, 0.0))
        throughput *= (1.0 - 0.05 * max(a2, 0.0))

        # a3: resource_share
        throughput *= (1.0 + 0.25 * max(a3, 0.0) - 0.10 * max(-a3, 0.0))
        latency *= (1.0 - 0.10 * max(a3, 0.0))

        if self.rng.random() < self.disturbance_prob:
            latency *= (1.0 + self.disturbance_strength)
            loss *= (1.0 + self.disturbance_strength)
            throughput *= (1.0 - self.disturbance_strength)

        throughput = max(0.0, throughput)
        served = min(throughput, offered_load_mbps)
        deficit = max(0.0, offered_load_mbps - served)
        self.queue_level = float(np.clip(0.8 * self.queue_level + 0.1 * deficit / 10.0, 0.0, 1.0))

        d_latency = max(0.0, (latency - self.latency_sla) / self.latency_sla)
        d_loss = max(0.0, (loss - self.loss_sla) / self.loss_sla)
        d_throughput = max(0.0, (self.throughput_min - served) / self.throughput_min)

        violation = float((d_latency > 0) or (d_loss > 0) or (d_throughput > 0))
        throughput_violation = float(served < self.throughput_min)

        wL, wP, wT = 1.0, 1.0, 2.0
        beta = 0.01
        lambda_c = 3.0

        reward = -(
            wL * d_latency +
            wP * d_loss +
            wT * d_throughput +
            beta * float(np.sum(np.square(action))) +
            lambda_c * throughput_violation
        )

        self.ptr += 1
        self.step_count += 1
        done = self.step_count >= self.episode_len

        info = {
            "latency_ms": float(latency),
            "packet_loss": float(loss),
            "throughput_mbps": float(served),
            "offered_load": float(offered_load_mbps),
            "queue_level": float(self.queue_level),
            "violation": float(violation),
            "throughput_violation": float(throughput_violation),
            "a1": a1,
            "a2": a2,
            "a3": a3,
        }

        return self._obs(latency, loss, served), float(reward), done, False, info
