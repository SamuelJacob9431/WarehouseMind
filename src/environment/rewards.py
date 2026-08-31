from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    step: float = -1.0
    collision: float = -5.0
    package: float = 20.0
    delivery: float = 100.0
    completion: float = 200.0