from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward values in the warehouse environment."""

    # Per-step cost.
    step: float = -1.0

    # Additional penalty when movement is blocked.
    collision: float = -5.0

    # Reward for picking up a package.
    package: float = 20.0

    # Reward for successfully delivering a package.
    delivery: float = 100.0

    # Bonus for delivering every package.
    completion: float = 200.0

    # Reward multiplier for reducing Manhattan distance
    # to the current target.
    distance: float = 0.5