from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import WarehouseConfig
from .entities import (
    DeliveryZone,
    Package,
    Position,
    Robot,
)
from .generator import WarehouseGenerator
from .rewards import RewardConfig


class WarehouseEnv(gym.Env):
    """
    Custom warehouse reinforcement-learning environment.

    Workflow:
        1. Find an undelivered package.
        2. Pick it up.
        3. Carry it to the delivery zone.
        4. Deliver it.
        5. Repeat until all packages are delivered.

    The robot carries exactly one package at a time.
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_NAMES = {
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
    }

    DIRECTIONS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.config = config or WarehouseConfig()

        if render_mode not in (None, "ansi"):
            raise ValueError(
                "render_mode must be None or 'ansi'."
            )

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        # Observation space (12 continuous values in [-1, 1]):
        # [0] robot row, [1] robot col, [2] target row, [3] target col,
        # [4] delivery row, [5] delivery col, [6] carrying state,
        # [7-10] blocked UP/DOWN/LEFT/RIGHT, [11] remaining packages
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.reward_config = RewardConfig(
            step=self.config.step_reward,
            collision=self.config.collision_reward,
            package=self.config.package_reward,
            delivery=self.config.delivery_reward,
            completion=self.config.completion_reward,
        )

        self.generator = WarehouseGenerator(
            grid_size=self.config.grid_size,
            obstacle_count=self.config.obstacle_count,
            num_packages=self.config.num_packages,
        )

        self.robot: Robot | None = None
        self.delivery_zone: DeliveryZone | None = None
        self.packages: list[Package] = []
        self.obstacles: set[Position] = set()
        self.carried_package: Package | None = None
        self.steps = 0
        self.total_reward = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        self.steps = 0
        self.total_reward = 0.0
        self.carried_package = None

        (
            robot_position,
            delivery_position,
            obstacles,
            package_positions,
        ) = self.generator.generate(self.np_random)

        self.robot = Robot(position=robot_position, carrying=False)
        self.delivery_zone = DeliveryZone(position=delivery_position)
        self.obstacles = obstacles
        self.packages = [
            Package(position=pos) for pos in package_positions
        ]

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        """Execute one environment step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = self.reward_config.step

        old_distance = self._distance_to_target()
        moved = self._move(action)

        if not moved:
            reward += (
                self.reward_config.collision
                - self.reward_config.step
            )

        new_distance = self._distance_to_target()

        if moved:
            reward += (
                old_distance - new_distance
            ) * self.reward_config.distance

        reward += self._handle_package_interaction()

        all_delivered = self._all_packages_delivered()
        terminated = all_delivered
        truncated = self.steps >= self.config.max_steps

        if all_delivered:
            reward += self.reward_config.completion

        self.total_reward += reward

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def _move(self, action: int) -> bool:
        """Attempt to move the robot. Returns True on success."""
        assert self.robot is not None

        row_delta, col_delta = self.DIRECTIONS[action]
        row, col = self.robot.position
        new_position = (row + row_delta, col + col_delta)

        if not self._is_valid_position(new_position):
            return False

        self.robot.position = new_position
        return True

    def _is_valid_position(self, position: Position) -> bool:
        """Check if position is within bounds and free of obstacles."""
        row, col = position

        if row <= 0 or row >= self.config.grid_size - 1:
            return False
        if col <= 0 or col >= self.config.grid_size - 1:
            return False
        if position in self.obstacles:
            return False

        return True

    def _handle_package_interaction(self) -> float:
        """Handle package pickup and delivery logic."""
        assert self.robot is not None
        assert self.delivery_zone is not None

        reward = 0.0

        if not self.robot.carrying:
            for package in self.packages:
                if package.delivered or package.position != self.robot.position:
                    continue

                self.robot.carrying = True
                self.carried_package = package
                reward += self.reward_config.package
                break
        else:
            if self.robot.position == self.delivery_zone.position:
                if self.carried_package is None:
                    return reward

                self.carried_package.delivered = True
                self.carried_package = None
                self.robot.carrying = False
                reward += self.reward_config.delivery

        return reward

    def _all_packages_delivered(self) -> bool:
        """Return True if all packages have been delivered."""
        return all(p.delivered for p in self.packages)

    def _distance_to_target(self) -> int:
        """Return Manhattan distance to target (package or delivery zone)."""
        assert self.robot is not None
        assert self.delivery_zone is not None

        robot_row, robot_col = self.robot.position

        if self.robot.carrying:
            target_row, target_col = self.delivery_zone.position
        else:
            package = self._get_nearest_undelivered_package()
            if package is None:
                target_row, target_col = self.delivery_zone.position
            else:
                target_row, target_col = package.position

        return abs(target_row - robot_row) + abs(target_col - robot_col)

    def _get_observation(self) -> np.ndarray:
        """Construct the 12-dimensional observation vector."""
        assert self.robot is not None
        assert self.delivery_zone is not None

        robot_row, robot_col = self.robot.position
        nearest_pkg = self._get_nearest_undelivered_package()

        if nearest_pkg is None:
            target_row, target_col = self.delivery_zone.position
        else:
            target_row, target_col = nearest_pkg.position

        max_coord = self.config.grid_size - 1

        values = [
            self._normalize(robot_row, max_coord),
            self._normalize(robot_col, max_coord),
            self._normalize(target_row, max_coord),
            self._normalize(target_col, max_coord),
            self._normalize(self.delivery_zone.position[0], max_coord),
            self._normalize(self.delivery_zone.position[1], max_coord),
            1.0 if self.robot.carrying else -1.0,
            float(self._is_blocked(self.robot.position, self.UP)),
            float(self._is_blocked(self.robot.position, self.DOWN)),
            float(self._is_blocked(self.robot.position, self.LEFT)),
            float(self._is_blocked(self.robot.position, self.RIGHT)),
            self._normalize(self._remaining_packages(), self.config.num_packages),
        ]

        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _normalize(value: int | float, maximum: int | float) -> float:
        """Normalize value into approximately [-1, 1]."""
        if maximum == 0:
            return 0.0
        return float((2.0 * value / maximum) - 1.0)

    def _get_nearest_undelivered_package(self) -> Package | None:
        """Return the closest undelivered package via Manhattan distance."""
        assert self.robot is not None

        available = [p for p in self.packages if not p.delivered]
        if not available:
            return None

        robot_row, robot_col = self.robot.position
        return min(
            available,
            key=lambda p: abs(p.position[0] - robot_row) + abs(p.position[1] - robot_col),
        )

    def _remaining_packages(self) -> int:
        """Return count of undelivered packages."""
        return sum(not p.delivered for p in self.packages)

    def _is_blocked(self, position: Position, action: int) -> bool:
        """Check if an action leads to an invalid position."""
        row_delta, col_delta = self.DIRECTIONS[action]
        row, col = position
        next_position = (row + row_delta, col + col_delta)
        return not self._is_valid_position(next_position)

    def _get_info(self) -> dict[str, Any]:
        """Return environment diagnostic information."""
        assert self.robot is not None
        assert self.delivery_zone is not None

        return {
            "robot_position": self.robot.position,
            "delivery_position": self.delivery_zone.position,
            "carrying": self.robot.carrying,
            "carried_package": None if self.carried_package is None else self.carried_package.position,
            "packages_remaining": self._remaining_packages(),
            "packages_delivered": self.config.num_packages - self._remaining_packages(),
            "steps": self.steps,
            "total_reward": self.total_reward,
            "distance_to_target": self._distance_to_target(),
        }

    def render(self):
        """Render the warehouse ASCII grid."""
        assert self.robot is not None
        assert self.delivery_zone is not None

        grid = [[" " for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)]

        # Border
        for row in range(self.config.grid_size):
            grid[row][0] = "#"
            grid[row][-1] = "#"
        for col in range(self.config.grid_size):
            grid[0][col] = "#"
            grid[-1][col] = "#"

        # Obstacles
        for row, col in self.obstacles:
            grid[row][col] = "█"

        # Packages
        for package in self.packages:
            if package.delivered or self.carried_package is package:
                continue
            row, col = package.position
            grid[row][col] = "P"

        # Delivery zone & Robot
        d_row, d_col = self.delivery_zone.position
        grid[d_row][d_col] = "D"

        r_row, r_col = self.robot.position
        grid[r_row][r_col] = "R"

        output = "\n".join("".join(row) for row in grid)

        if self.render_mode == "ansi":
            return output

        print(output)

    def close(self):
        """Clean up environment resources."""
        pass