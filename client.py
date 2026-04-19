import os
import time
import socket
import argparse
import numpy as np

from resilience_env import ResilienceEnv
from ppo_agent import build_ppo
from common import (
    now_ms,
    save_csv_row,
    send_msg,
    recv_msg,
    get_state_dict,
    set_state_dict,
)


def make_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def build_file_prefix(algo_tag: str, run_id: str, client_id: int) -> str:
    return f"{algo_tag}_{run_id}_client_{client_id}"


def zero_action(env):
    if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
        return np.zeros(env.action_space.shape, dtype=np.float32)
    a = env.action_space.sample()
    try:
        return np.zeros_like(a, dtype=np.float32)
    except Exception:
        return a


def action_triplet(action):
    arr = np.array(action).reshape(-1)
    if len(arr) >= 3:
        return float(arr[0]), float(arr[1]), float(arr[2])
    if len(arr) == 2:
        return float(arr[0]), float(arr[1]), 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0, 0.0
    return 0.0, 0.0, 0.0


def evaluate(model, env, client_id, round_idx, out_csv, max_steps=200, fixed_action=None):
    obs, _ = env.reset()
    rows = []

    for step in range(max_steps):
        if fixed_action is None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = fixed_action

        obs, reward, done, trunc, info = env.step(action)
        a1, a2, a3 = action_triplet(action)

        row = {
            "time_ms": now_ms(),
            "client_id": client_id,
            "round_idx": round_idx,
            "step": step,
            "latency_ms": info["latency_ms"],
            "packet_loss": info["packet_loss"],
            "throughput_mbps": info["throughput_mbps"],
            "offered_load": info.get("offered_load", 0.0),
            "queue_level": info.get("queue_level", 0.0),
            "reward": reward,
            "violation": info.get("violation", 0),
            "throughput_violation": info.get("throughput_violation", 0),
            "a1": info.get("a1", a1),
            "a2": info.get("a2", a2),
            "a3": info.get("a3", a3),
        }
        rows.append(row)
        if done or trunc:
            break

    for row in rows:
        save_csv_row(out_csv, list(row.keys()), row)


def run_baseline(args):
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = build_file_prefix(args.algo_tag, args.run_id, args.client_id)
    client_log = os.path.join(args.out_dir, f"{prefix}_round_log.csv")
    eval_log = os.path.join(args.out_dir, f"{prefix}_eval.csv")

    env = ResilienceEnv(csv_path=args.trace_csv, seed=args.seed + args.client_id)
    fixed_action = zero_action(env)

    for round_idx in range(args.rounds):
        t0 = now_ms()
        evaluate(
            model=None,
            env=env,
            client_id=args.client_id,
            round_idx=round_idx,
            out_csv=eval_log,
            max_steps=args.eval_steps,
            fixed_action=fixed_action,
        )
        t1 = now_ms()

        save_csv_row(
            client_log,
            [
                "time_ms", "client_id", "round_idx", "base_version", "acked_global_version",
                "pull_ms", "local_train_ms", "push_ms"
            ],
            {
                "time_ms": now_ms(),
                "client_id": args.client_id,
                "round_idx": round_idx,
                "base_version": 0,
                "acked_global_version": 0,
                "pull_ms": 0.0,
                "local_train_ms": t1 - t0,
                "push_ms": 0.0,
            }
        )
        time.sleep(args.sleep_between_rounds)


def run_local_ppo(args):
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = build_file_prefix(args.algo_tag, args.run_id, args.client_id)
    client_log = os.path.join(args.out_dir, f"{prefix}_round_log.csv")
    eval_log = os.path.join(args.out_dir, f"{prefix}_eval.csv")

    env = ResilienceEnv(csv_path=args.trace_csv, seed=args.seed + args.client_id)
    model = build_ppo(env, seed=args.seed + args.client_id)

    for round_idx in range(args.rounds):
        train_start = now_ms()
        model.learn(total_timesteps=args.local_timesteps, reset_num_timesteps=False)
        train_end = now_ms()

        save_csv_row(
            client_log,
            [
                "time_ms", "client_id", "round_idx", "base_version", "acked_global_version",
                "pull_ms", "local_train_ms", "push_ms"
            ],
            {
                "time_ms": now_ms(),
                "client_id": args.client_id,
                "round_idx": round_idx,
                "base_version": 0,
                "acked_global_version": 0,
                "pull_ms": 0.0,
                "local_train_ms": train_end - train_start,
                "push_ms": 0.0,
            }
        )

        evaluate(model, env, args.client_id, round_idx, eval_log, max_steps=args.eval_steps)
        time.sleep(args.sleep_between_rounds)


def run_federated(args):
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = build_file_prefix(args.algo_tag, args.run_id, args.client_id)
    client_log = os.path.join(args.out_dir, f"{prefix}_round_log.csv")
    eval_log = os.path.join(args.out_dir, f"{prefix}_eval.csv")

    env = ResilienceEnv(csv_path=args.trace_csv, seed=args.seed + args.client_id)
    model = build_ppo(env, seed=args.seed + args.client_id)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))

    send_msg(sock, {"type": "hello", "client_id": args.client_id})
    send_msg(sock, {"type": "init_state", "state": get_state_dict(model)})

    for round_idx in range(args.rounds):
        t0 = now_ms()

        send_msg(sock, {"type": "pull"})
        resp = recv_msg(sock)
        if resp is None:
            raise RuntimeError("Server closed connection before sending global state.")
        global_state = resp["global_state"]
        global_version = resp["global_version"]

        set_state_dict(model, global_state)

        train_start = now_ms()
        model.learn(total_timesteps=args.local_timesteps, reset_num_timesteps=False)
        train_end = now_ms()

        local_state = get_state_dict(model)

        push_start = now_ms()
        send_msg(sock, {
            "type": "push_update",
            "client_id": args.client_id,
            "base_version": global_version,
            "state": local_state,
        })
        ack = recv_msg(sock)
        if ack is None:
            raise RuntimeError("Server closed connection before sending ack.")
        push_end = now_ms()

        save_csv_row(
            client_log,
            [
                "time_ms", "client_id", "round_idx", "base_version", "acked_global_version",
                "pull_ms", "local_train_ms", "push_ms"
            ],
            {
                "time_ms": now_ms(),
                "client_id": args.client_id,
                "round_idx": round_idx,
                "base_version": global_version,
                "acked_global_version": ack["global_version"],
                "pull_ms": train_start - t0,
                "local_train_ms": train_end - train_start,
                "push_ms": push_end - push_start,
            }
        )

        evaluate(model, env, args.client_id, round_idx, eval_log, max_steps=args.eval_steps)
        time.sleep(args.sleep_between_rounds)

    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="")
    parser.add_argument("--server_port", type=int, default=7000)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--trace_csv", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep_between_rounds", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument(
        "--run_mode",
        type=str,
        default="federated",
        choices=["baseline", "local_ppo", "federated"]
    )
    parser.add_argument("--algo_tag", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    if not args.run_id:
        args.run_id = make_run_id()

    if not args.algo_tag:
        if args.run_mode == "baseline":
            args.algo_tag = "baseline"
        elif args.run_mode == "local_ppo":
            args.algo_tag = "localppo"
        else:
            args.algo_tag = "federated"

    if args.run_mode == "baseline":
        run_baseline(args)
    elif args.run_mode == "local_ppo":
        run_local_ppo(args)
    else:
        if not args.server_ip:
            raise ValueError("server_ip is required in federated mode")
        run_federated(args)
