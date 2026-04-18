import socket
import threading
import argparse
from collections import OrderedDict

from common import send_msg, recv_msg, save_csv_row, now_ms


class FedServer:
    def __init__(self, host, port, num_clients, mode="fedasync", alpha=1.0):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.mode = mode
        self.alpha = alpha

        self.global_state = None
        self.global_version = 0
        self.client_socks = {}
        self.lock = threading.Lock()

        self.pending_sync_updates = []
        self.server_log = "results/server_log.csv"

    def staleness_beta(self, stale):
        return 1.0 / (1.0 + self.alpha * max(stale, 0))

    def aggregate_async(self, local_state, base_version, client_id):
        stale = self.global_version - base_version
        beta = self.staleness_beta(stale)

        if self.global_state is None:
            self.global_state = local_state
        else:
            new_state = OrderedDict()
            for k in self.global_state.keys():
                new_state[k] = (1.0 - beta) * self.global_state[k] + beta * local_state[k]
            self.global_state = new_state

        before = self.global_version
        self.global_version += 1

        save_csv_row(
            self.server_log,
            ["time_ms", "client_id", "mode", "base_version", "global_before", "global_after", "staleness", "beta"],
            {
                "time_ms": now_ms(),
                "client_id": client_id,
                "mode": self.mode,
                "base_version": base_version,
                "global_before": before,
                "global_after": self.global_version,
                "staleness": stale,
                "beta": beta,
            },
        )

    def aggregate_sync_if_ready(self):
        if len(self.pending_sync_updates) < self.num_clients:
            return

        updates = self.pending_sync_updates
        self.pending_sync_updates = []

        avg_state = OrderedDict()
        for k in updates[0]["state"].keys():
            avg_state[k] = sum(u["state"][k] for u in updates) / len(updates)

        self.global_state = avg_state
        before = self.global_version
        self.global_version += 1

        for upd in updates:
            stale = self.global_version - upd["base_version"]
            save_csv_row(
                self.server_log,
                ["time_ms", "client_id", "mode", "base_version", "global_before", "global_after", "staleness", "beta"],
                {
                    "time_ms": now_ms(),
                    "client_id": upd["client_id"],
                    "mode": self.mode,
                    "base_version": upd["base_version"],
                    "global_before": before,
                    "global_after": self.global_version,
                    "staleness": stale,
                    "beta": 1.0 / len(updates),
                },
            )

    def handle_client(self, conn, addr):
        client_id = None
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break

                msg_type = msg.get("type")

                if msg_type == "hello":
                    client_id = msg["client_id"]
                    with self.lock:
                        self.client_socks[client_id] = conn
                    print(f"[SERVER] client {client_id} connected from {addr}")

                elif msg_type == "init_state":
                    with self.lock:
                        if self.global_state is None:
                            self.global_state = msg["state"]
                            print("[SERVER] initialized global state from first client")

                elif msg_type == "pull":
                    with self.lock:
                        send_msg(conn, {
                            "type": "global_model",
                            "global_state": self.global_state,
                            "global_version": self.global_version,
                        })

                elif msg_type == "push_update":
                    local_state = msg["state"]
                    base_version = msg["base_version"]

                    with self.lock:
                        if self.mode == "fedasync":
                            self.aggregate_async(local_state, base_version, client_id)
                        else:
                            self.pending_sync_updates.append({
                                "client_id": client_id,
                                "base_version": base_version,
                                "state": local_state,
                            })
                            self.aggregate_sync_if_ready()

                        send_msg(conn, {
                            "type": "ack",
                            "global_version": self.global_version,
                        })

                else:
                    print(f"[SERVER] unknown msg type: {msg_type}")

        except Exception as e:
            print(f"[SERVER] client handler error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            if client_id is not None:
                print(f"[SERVER] client {client_id} disconnected")

    def serve_forever(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(self.num_clients)
        print(f"[SERVER] listening on {self.host}:{self.port} mode={self.mode}")

        while True:
            conn, addr = srv.accept()
            th = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            th.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--mode", type=str, default="fedasync", choices=["fedasync", "fedavg"])
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    server = FedServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        mode=args.mode,
        alpha=args.alpha,
    )
    server.serve_forever()
