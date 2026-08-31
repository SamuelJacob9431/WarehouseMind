from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import WarehouseConfig
from .entities import DeliveryZone, Package, Position, Robot
from .generator import WarehouseGenerator
from .rewards import RewardConfig


class WarehouseEnv(gym.Env):
    """
    Custom warehouse environment for reinforcement learning.

    The robot must collect every package and bring each package
    to the delivery zone.

    The robot can carry one package at a time.
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    # Actions
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

        # Action space
     

        self.action_space = spaces.Discrete(4)

       
        # Observation
       

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

       
        # Environment components
     

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

        # State
       

        self.robot: Robot | None = None
        self.delivery_zone: DeliveryZone | None = None

        self.packages: list[Package] = []
        self.obstacles: set[Position] = set()

        self.steps = 0
        self.total_reward = 0.0

    # Gymnasium API
  

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.steps = 0
        self.total_reward = 0.0

        (
            robot_position,
            delivery_position,
            obstacles,
            package_positions,
        ) = self.generator.generate(self.np_random)

        self.robot = Robot(
            position=robot_position,
            carrying=False,
        )

        self.delivery_zone = DeliveryZone(
            position=delivery_position,
        )

        self.obstacles = obstacles

        self.packages = [
            Package(position=position)
            for position in package_positions
        ]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        self.steps += 1

        reward = self.reward_config.step

        
        # Movement
       

        moved = self._move(action)

        if not moved:
            reward += (
                self.reward_config.collision
                - self.reward_config.step
            )

   
        # Automatic package interaction
        

        reward += self._handle_package_interaction()


        # Episode status
      

        all_delivered = self._all_packages_delivered()

        terminated = all_delivered

        truncated = (
            self.steps >= self.config.max_steps
        )

        if all_delivered:
            reward += self.reward_config.completion

        self.total_reward += reward

        observation = self._get_observation()
        info = self._get_info()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # Movement
    

    def _move(self, action: int) -> bool:

        assert self.robot is not None

        row_delta, col_delta = self.DIRECTIONS[action]

        row, col = self.robot.position

        new_position = (
            row + row_delta,
            col + col_delta,
        )

        if not self._is_valid_position(new_position):
            return False

        self.robot.position = new_position

        return True

    def _is_valid_position(
        self,
        position: Position,
    ) -> bool:

        row, col = position

        if row <= 0 or row >= self.config.grid_size - 1:
            return False

        if col <= 0 or col >= self.config.grid_size - 1:
            return False

        if position in self.obstacles:
            return False

        return True

    
    # Package logic
    

    def _handle_package_interaction(self) -> float:

        assert self.robot is not None
        assert self.delivery_zone is not None

        reward = 0.0

        # Pick up package
       
        if not self.robot.carrying:

            for package in self.packages:

                if package.delivered:
                    continue

                if package.position == self.robot.position:

                    self.robot.carrying = True

                    reward += self.reward_config.package

                    break

       
        # Deliver package
       
        else:

            if (
                self.robot.position
                == self.delivery_zone.position
            ):

                self.robot.carrying = False

                for package in self.packages:

                    if (
                        not package.delivered
                        and package.position
                        == self.robot.position
                    ):
                        package.delivered = True
                        break

                reward += self.reward_config.delivery

        return reward

    def _all_packages_delivered(self) -> bool:

        return all(
            package.delivered
            for package in self.packages
        )


    # Observation
    

    def _get_observation(self) -> np.ndarray:

        assert self.robot is not None
        assert self.delivery_zone is not None

        robot_row, robot_col = self.robot.position

        nearest_package = (
            self._get_nearest_undelivered_package()
        )

        if nearest_package is None:
            target_row, target_col = (
                self.delivery_zone.position
            )
        else:
            target_row, target_col = (
                nearest_package.position
            )

        max_coordinate = self.config.grid_size - 1

        values = [
            self._normalize(robot_row, max_coordinate),
            self._normalize(robot_col, max_coordinate),

            self._normalize(target_row, max_coordinate),
            self._normalize(target_col, max_coordinate),

            self._normalize(
                self.delivery_zone.position[0],
                max_coordinate,
            ),
            self._normalize(
                self.delivery_zone.position[1],
                max_coordinate,
            ),

            1.0 if self.robot.carrying else -1.0,

            float(
                self._is_blocked(
                    self.robot.position,
                    self.UP,
                )
            ),

            float(
                self._is_blocked(
                    self.robot.position,
                    self.DOWN,
                )
            ),

            float(
                self._is_blocked(
                    self.robot.position,
                    self.LEFT,
                )
            ),

            float(
                self._is_blocked(
                    self.robot.position,
                    self.RIGHT,
                )
            ),

            self._normalize(
                self._remaining_packages(),
                self.config.num_packages,
            ),
        ]

        return np.asarray(
            values,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize(
        value: int | float,
        maximum: int | float,
    ) -> float:

        if maximum == 0:
            return 0.0

        return float(
            (2.0 * value / maximum) - 1.0
        )

   
    # Observation helpers
   
    def _get_nearest_undelivered_package(
        self,
    ) -> Package | None:

        assert self.robot is not None

        available = [
            package
            for package in self.packages
            if not package.delivered
        ]

        if not available:
            return None

        robot_row, robot_col = self.robot.position

        return min(
            available,
            key=lambda package: (
                abs(package.position[0] - robot_row)
                + abs(package.position[1] - robot_col)
            ),
        )

    def _remaining_packages(self) -> int:

        return sum(
            not package.delivered
            for package in self.packages
        )

    def _is_blocked(
        self,
        position: Position,
        action: int,
    ) -> bool:

        row_delta, col_delta = self.DIRECTIONS[action]

        row, col = position

        next_position = (
            row + row_delta,
            col + col_delta,
        )

        return not self._is_valid_position(
            next_position
        )

    
    # Info
    

    def _get_info(self) -> dict[str, Any]:

        assert self.robot is not None
        assert self.delivery_zone is not None

        return {
            "robot_position": self.robot.position,
            "delivery_position": (
                self.delivery_zone.position
            ),
            "carrying": self.robot.carrying,
            "packages_remaining": (
                self._remaining_packages()
            ),
            "packages_delivered": (
                self.config.num_packages
                - self._remaining_packages()
            ),
            "steps": self.steps,
            "total_reward": self.total_reward,
        }


    # Rendering
   

    def render(self):

        assert self.robot is not None
        assert self.delivery_zone is not None

        grid = [
            [" " for _ in range(self.config.grid_size)]
            for _ in range(self.config.grid_size)
        ]

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

            if not package.delivered:
                row, col = package.position
                grid[row][col] = "P"

        # Delivery zone
        row, col = self.delivery_zone.position
        grid[row][col] = "D"

        # Robot
        row, col = self.robot.position

        if self.robot.carrying:
            grid[row][col] = "R"
        else:
            grid[row][col] = "R"

        output = "\n".join(
            "".join(row)
            for row in grid
        )

        if self.render_mode == "ansi":
            return output

        print(output)

    def close(self):
        pass