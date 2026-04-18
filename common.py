import socket
import json
import base64
import pickle
import struct
import time
import csv
import os


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
