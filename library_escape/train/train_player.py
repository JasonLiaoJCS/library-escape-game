"""Train an RL player against a scripted enemy."""

from __future__ import annotations

from .single_agent_runner import run_single_agent


def main() -> None:
    run_single_agent("player")


if __name__ == "__main__":
    main()
