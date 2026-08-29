from dataclasses import dataclass


@dataclass
class WarehouseConfig:
    """Configuration for a warehouse episode."""

    grid_size: int = 10

    # Number of packages the robot must deliver.
    num_packages: int = 3

    # Number of randomly generated obstacles for ze map.
    obstacle_count: int = 8

    # Maximum number of actions in one episode(one Run)
    max_steps: int = 200

    # Reward configs
    step_reward: float = -1.0
    collision_reward: float = -5.0
    package_reward: float = 20.0
    delivery_reward: float = 100.0
    completion_reward: float = 200.0

    def __post_init__(self):
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5.")

        if self.num_packages < 1:
            raise ValueError("num_packages must be at least 1.")

        if self.obstacle_count < 0:
            raise ValueError("obstacle_count cannot be negative.")

        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")