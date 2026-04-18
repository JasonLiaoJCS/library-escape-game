"""Simple opponent pool for alternating self-play."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable


class OpponentPool:
    def __init__(self, max_size: int, seed: int | None = None) -> None:
        self.max_size = max_size
        self._rng = random.Random(seed)
        self._entries: deque[Callable[[], object]] = deque(maxlen=max_size)

    def add(self, factory: Callable[[], object]) -> None:
        self._entries.append(factory)

    def sample(self, include_latest: bool = True) -> object:
        if not self._entries:
            raise RuntimeError("OpponentPool is empty.")
        entries = list(self._entries)
        if not include_latest and len(entries) > 1:
            entries = entries[:-1]
        factory = self._rng.choice(entries)
        return factory()

    def latest(self) -> object:
        if not self._entries:
            raise RuntimeError("OpponentPool is empty.")
        return self._entries[-1]()

    def __len__(self) -> int:
        return len(self._entries)


class PooledController:
    def __init__(
        self,
        pool: OpponentPool | None,
        fallback_factory,
        latest_weight: float = 0.5,
        historical_weight: float = 0.5,
    ):
        self.pool = pool
        self.fallback_factory = fallback_factory
        self.latest_weight = max(0.0, float(latest_weight))
        self.historical_weight = max(0.0, float(historical_weight))
        self._rng = random.Random()
        self.current = fallback_factory()

    def reset(self) -> None:
        if self.pool is not None and len(self.pool) > 0:
            if len(self.pool) == 1:
                self.current = self.pool.latest()
            else:
                total = self.latest_weight + self.historical_weight
                if total <= 1e-8:
                    self.current = self.pool.sample()
                elif self._rng.random() < (self.latest_weight / total):
                    self.current = self.pool.latest()
                else:
                    self.current = self.pool.sample(include_latest=False)
        else:
            self.current = self.fallback_factory()
        if hasattr(self.current, "reset"):
            self.current.reset()

    def act(self, world):
        return self.current.act(world)


class WeightedControllerChooser:
    def __init__(self, weighted_factories: list[tuple[Callable[[], object], float]], seed: int | None = None):
        self._rng = random.Random(seed)
        self._weighted_factories = [(factory, max(0.0, float(weight))) for factory, weight in weighted_factories if weight > 0.0]
        if not self._weighted_factories:
            raise ValueError("WeightedControllerChooser requires at least one positive-weight factory.")
        self.current = self._weighted_factories[0][0]()

    def _sample_factory(self) -> Callable[[], object]:
        factories = [item[0] for item in self._weighted_factories]
        weights = [item[1] for item in self._weighted_factories]
        return self._rng.choices(factories, weights=weights, k=1)[0]

    def reset(self) -> None:
        self.current = self._sample_factory()()
        if hasattr(self.current, "reset"):
            self.current.reset()

    def act(self, world):
        return self.current.act(world)
