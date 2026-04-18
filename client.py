import socket
import argparse
import time
import os
import numpy as np

from common import send_msg, recv_msg, save_csv_row, now_ms
from resilience_env import ResilienceEnv
from ppo_agent import build_ppo, get_state_dict, set_state_dict


def evaluate(model, env, client_id, round_idx, out_csv, max_steps=200):
    obs, _ = env.reset()
    rows = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        row = {
            "time_ms": now_ms(),
            "client_id": client_id,
            "round_idx": round_idx,
            "step": step,
            "latency_ms": info["latency_ms"],
            "packet_loss": info["packet_loss"],
            "throughput_mbps": info["throughput_mbps"],
            "offered_load": info["offered_load"],
            "queue_level": info["queue_level"],
            "reward": reward,
            "violation": info["violation"],
            "throughput_violation": info["throughput_violation"],
            "a1": info["a1"],
            "a2": info["a2"],
            "a3": info["a3"],
        }
        rows.append(row)
        if done or trunc:
            break

    for row in rows:
        save_csv_row(
            out_csv,
            list(row.keys()),
            row
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, required=True)
    parser.add_argument("--server_port", type=int, default=7000)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--trace_csv", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep_between_rounds", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    client_log = f"results/client_{args.client_id}_round_log.csv"
    eval_log = f"results/client_{args.client_id}_eval.csv"

    env = ResilienceEnv(csv_path=args.trace_csv, seed=args.seed + args.client_id)
    model = build_ppo(env, seed=args.seed + args.client_id)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))

    send_msg(sock, {"type": "hello", "client_id": args.client_id})
    send_msg(sock, {"type": "init_state", "state": get_state_dict(model)})

    for round_idx in range(args.rounds):
        # pull global
        t0 = now_ms()
        send_msg(sock, {"type": "pull"})
        resp = recv_msg(sock)
        global_state = resp["global_state"]
        global_version = resp["global_version"]

        set_state_dict(model, global_state)

        # local train
        train_start = now_ms()
        model.learn(total_timesteps=args.local_timesteps, reset_num_timesteps=False)
        train_end = now_ms()

        local_state = get_state_dict(model)

        # push update
        push_start = now_ms()
        send_msg(sock, {
            "type": "push_update",
            "client_id": args.client_id,
            "base_version": global_version,
            "state": local_state,
        })
        ack = recv_msg(sock)
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

        evaluate(model, env, args.client_id, round_idx, eval_log, max_steps=200)
        time.sleep(args.sleep_between_rounds)

    sock.close()
