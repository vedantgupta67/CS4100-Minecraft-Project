"""Compatibility wrapper for block identification data collection.

This entry point preserves the historical behavior of skipping empty labels,
while delegating implementation to block_identification_train.py.
"""

from block_identification_train import *  # noqa: F401,F403
from block_identification_train import run as _run


def run(episodes: int = 1, max_steps: int = 50000):
    return _run(episodes=episodes, max_steps=max_steps, include_empty_labels=False)


if __name__ == "__main__":
    run()
