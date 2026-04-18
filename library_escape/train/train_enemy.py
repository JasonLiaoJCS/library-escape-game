"""Train an RL enemy against scripted player behaviors."""

from __future__ import annotations

from .single_agent_runner import run_single_agent


def main() -> None:
    run_single_agent("enemy")


if __name__ == "__main__":
    main()
