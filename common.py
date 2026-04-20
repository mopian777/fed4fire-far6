import socket
import json
import base64
import pickle
import struct
import time
import csv
import os
import torch


def now_ms():
    return time.time() * 1000.0


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock):
    header = recv_all(sock, 4)
    if header is None:
        return None
    length = struct.unpack("!I", header)[0]
    body = recv_all(sock, length)
    if body is None:
        return None
    return pickle.loads(body)


def save_csv_row(path, fieldnames, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_state_dict(model):
    state = model.policy.state_dict()
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def set_state_dict(model, state_dict):
    current = model.policy.state_dict()
    loaded = {}

    for k, v in current.items():
        if k in state_dict:
            sv = state_dict[k]
            if torch.is_tensor(v) and torch.is_tensor(sv):
                loaded[k] = sv.to(v.device).type(v.dtype)
            else:
                loaded[k] = sv
        else:
            loaded[k] = v

    model.policy.load_state_dict(loaded, strict=False)
