"""Send a single speed command (0-100) to the ESP32 motor controller over WiFi.

Usage:
    py tools/send_motor_speed.py 50     # set speed to 50%
    py tools/send_motor_speed.py 0      # stop
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from realtime_inspection import MOTOR_HOST, MOTOR_PORT

if len(sys.argv) != 2:
    raise SystemExit("Usage: py tools/send_motor_speed.py <speed 0-100>")

speed = sys.argv[1]

print(f"Connecting to {MOTOR_HOST}:{MOTOR_PORT} ...")
s = socket.create_connection((MOTOR_HOST, MOTOR_PORT), timeout=5)
s.sendall(f"{speed}\n".encode())
s.close()
print(f"Sent speed {speed}.")
