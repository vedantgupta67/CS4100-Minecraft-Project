"""
Wrapper that restarts log_training.py indefinitely.
Press Enter (or type anything + Enter) to finish the current episode and stop.
"""

import os
import sys
import subprocess
import threading

STOP_FLAG = "STOP_REQUESTED"
SCRIPT = os.path.join(os.path.dirname(__file__), "log_training.py")


def listen_for_stop(stop_event: threading.Event):
    print("[WRAPPER] Press Enter to stop after the current episode...")
    sys.stdin.readline()
    print("[WRAPPER] Stop requested — will exit after this episode finishes.")
    open(STOP_FLAG, "w").close()
    stop_event.set()


def main():
    # Clean up any leftover flag from a previous run
    if os.path.exists(STOP_FLAG):
        os.remove(STOP_FLAG)

    stop_event = threading.Event()
    listener = threading.Thread(target=listen_for_stop, args=(stop_event,), daemon=True)
    listener.start()

    run_number = 0
    while not stop_event.is_set():
        run_number += 1
        print(f"\n{'═' * 50}")
        print(f"[WRAPPER] Starting run #{run_number}")
        print(f"{'═' * 50}\n")

        result = subprocess.run([sys.executable, SCRIPT])

        if result.returncode != 0:
            print(f"[WRAPPER] Run #{run_number} exited with code {result.returncode}")

        if os.path.exists(STOP_FLAG):
            print("[WRAPPER] Stop flag found — not restarting.")
            break

    # Clean up flag file
    if os.path.exists(STOP_FLAG):
        os.remove(STOP_FLAG)

    print(f"[WRAPPER] Done after {run_number} run(s).")


if __name__ == "__main__":
    main()
