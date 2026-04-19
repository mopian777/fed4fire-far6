import os
import time
import socket
import threading
import argparse
from copy import deepcopy

import torch

from common import now_ms, save_csv_row, send_msg, recv_msg


def make_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


class FLServer:
    def __init__(self, host, port, num_clients, mode, alpha, server_log_path):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.mode = mode.lower()
        self.alpha = float(alpha)
        self.server_log_path = server_log_path

        self.lock = threading.Lock()

        self.global_state = None
        self.global_version = 0

        self.server_sock = None
        self.is_running = False

    def _clone_state(self, state):
        out = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = deepcopy(v)
        return out

    def _mix_states(self, global_state, local_state, beta):
        new_state = {}
        for k in global_state.keys():
            gv = global_state[k]
            lv = local_state[k]

            if torch.is_tensor(gv) and torch.is_tensor(lv):
                gv_cpu = gv.detach().cpu()
                lv_cpu = lv.detach().cpu()

                if torch.is_floating_point(gv_cpu):
                    new_state[k] = (1.0 - beta) * gv_cpu + beta * lv_cpu
                else:
                    # 非浮点参数（如计数器）直接采用本地值
                    new_state[k] = lv_cpu.clone()
            else:
                new_state[k] = deepcopy(lv)
        return new_state

    def _compute_beta(self, staleness):
        if self.mode == "fedavg":
            return 1.0
        elif self.mode == "fedasync":
            return float((staleness + 1) ** (-self.alpha))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _log_server_event(self, client_id, base_version, global_before, global_after, staleness, beta):
        save_csv_row(
            self.server_log_path,
            [
                "time_ms",
                "client_id",
                "mode",
                "base_version",
                "global_before",
                "global_after",
                "staleness",
                "beta",
            ],
            {
                "time_ms": now_ms(),
                "client_id": client_id,
                "mode": self.mode,
                "base_version": base_version,
                "global_before": global_before,
                "global_after": global_after,
                "staleness": staleness,
                "beta": beta,
            },
        )

    def _handle_init_state(self, msg, client_id):
        state = msg["state"]
        with self.lock:
            if self.global_state is None:
                self.global_state = self._clone_state(state)
                self.global_version = 0
                print(f"[SERVER] initialized global state from first client")

    def _handle_pull(self, conn):
        with self.lock:
            if self.global_state is None:
                raise RuntimeError("Global state is None when processing pull.")
            resp = {
                "type": "pull_resp",
                "global_state": self._clone_state(self.global_state),
                "global_version": self.global_version,
            }
        send_msg(conn, resp)

    def _handle_push_update(self, conn, msg):
        client_id = msg["client_id"]
        base_version = int(msg["base_version"])
        local_state = msg["state"]

        with self.lock:
            if self.global_state is None:
                raise RuntimeError("Global state is None when processing push_update.")

            global_before = int(self.global_version)
            staleness = max(1, global_before - base_version + 1)
            beta = self._compute_beta(staleness)

            self.global_state = self._mix_states(self.global_state, local_state, beta)
            self.global_version += 1
            global_after = int(self.global_version)

            self._log_server_event(
                client_id=client_id,
                base_version=base_version,
                global_before=global_before,
                global_after=global_after,
                staleness=staleness,
                beta=beta,
            )

            ack = {
                "type": "ack",
                "global_version": self.global_version,
            }

        send_msg(conn, ack)

    def handle_client(self, conn, addr):
        client_id = None
        try:
            print(f"[SERVER] client connected from {addr}")

            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print(f"[SERVER] client {client_id} disconnected")
                    break

                msg_type = msg.get("type", "")

                if msg_type == "hello":
                    client_id = msg.get("client_id", None)
                    print(f"[SERVER] hello from client {client_id}")

                elif msg_type == "init_state":
                    self._handle_init_state(msg, client_id)

                elif msg_type == "pull":
                    self._handle_pull(conn)

                elif msg_type == "push_update":
                    self._handle_push_update(conn, msg)

                else:
                    raise ValueError(f"Unknown message type: {msg_type}")

        except Exception as e:
            print(f"[SERVER] client handler error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def serve_forever(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(64)
        self.is_running = True

        print(
            f"[SERVER] listening on {self.host}:{self.port} | "
            f"mode={self.mode} | num_clients={self.num_clients} | "
            f"log={self.server_log_path}"
        )

        try:
            while self.is_running:
                conn, addr = self.server_sock.accept()
                th = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                th.start()
        finally:
            self.close()

    def close(self):
        self.is_running = False
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--mode", type=str, default="fedavg", choices=["fedavg", "fedasync"])
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--algo_tag", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    if not args.run_id:
        args.run_id = make_run_id()

    if not args.algo_tag:
        args.algo_tag = args.mode.lower()

    os.makedirs(args.out_dir, exist_ok=True)
    server_log_path = os.path.join(
        args.out_dir,
        f"server_{args.algo_tag}_{args.run_id}.csv"
    )

    server = FLServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        mode=args.mode,
        alpha=args.alpha,
        server_log_path=server_log_path,
    )
    server.serve_forever()
