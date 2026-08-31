from __future__ import annotations

from collections import deque

import numpy as np

from .entities import Position


class WarehouseGenerator:
    """Generates valid random warehouse layouts."""

    def __init__(
        self,
        grid_size: int,
        obstacle_count: int,
        num_packages: int,
    ):
        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.num_packages = num_packages

    def generate(
        self,
        rng: np.random.Generator,
    ) -> tuple[
        Position,
        Position,
        set[Position],
        list[Position],
    ]:
        """
        Generate:

        robot position
        delivery position
        obstacle positions
        package positions
        """

        robot = (1, 1)

        delivery = (
            self.grid_size - 2,
            self.grid_size - 2,
        )

        forbidden = {
            robot,
            delivery,
        }

        obstacles = self._generate_obstacles(
            rng,
            forbidden,
        )

        packages = self._generate_packages(
            rng,
            forbidden | obstacles,
        )

        # Ensure every important location is reachable.
        if not self._layout_is_reachable(
            robot,
            delivery,
            obstacles,
            packages,
        ):
            return self.generate(rng)

        return robot, delivery, obstacles, packages

    def _generate_obstacles(
        self,
        rng: np.random.Generator,
        forbidden: set[Position],
    ) -> set[Position]:

        obstacles: set[Position] = set()

        available = [
            (row, col)
            for row in range(1, self.grid_size - 1)
            for col in range(1, self.grid_size - 1)
            if (row, col) not in forbidden
        ]

        if self.obstacle_count >= len(available):
            raise ValueError(
                "Too many obstacles for this warehouse size."
            )

        selected_indices = rng.choice(
            len(available),
            size=self.obstacle_count,
            replace=False,
        )

        for index in selected_indices:
            obstacles.add(available[int(index)])

        return obstacles

    def _generate_packages(
        self,
        rng: np.random.Generator,
        forbidden: set[Position],
    ) -> list[Position]:

        available = [
            (row, col)
            for row in range(1, self.grid_size - 1)
            for col in range(1, self.grid_size - 1)
            if (row, col) not in forbidden
        ]

        if self.num_packages >= len(available):
            raise ValueError(
                "Too many packages for this warehouse size."
            )

        indices = rng.choice(
            len(available),
            size=self.num_packages,
            replace=False,
        )

        return [
            available[int(index)]
            for index in indices
        ]

    def _layout_is_reachable(
        self,
        robot: Position,
        delivery: Position,
        obstacles: set[Position],
        packages: list[Position],
    ) -> bool:
        """
        Check whether all important positions are reachable
        from the robot.
        """

        reachable = self._flood_fill(
            robot,
            obstacles,
        )

        if delivery not in reachable:
            return False

        return all(
            package in reachable
            for package in packages
        )

    def _flood_fill(
        self,
        start: Position,
        obstacles: set[Position],
    ) -> set[Position]:

        queue = deque([start])
        visited = {start}

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        while queue:
            row, col = queue.popleft()

            for row_delta, col_delta in directions:

                next_position = (
                    row + row_delta,
                    col + col_delta,
                )

                if not self._is_valid_position(
                    next_position,
                    obstacles,
                ):
                    continue

                if next_position in visited:
                    continue

                visited.add(next_position)
                queue.append(next_position)

        return visited

    def _is_valid_position(
        self,
        position: Position,
        obstacles: set[Position],
    ) -> bool:

        row, col = position

        if row <= 0 or row >= self.grid_size - 1:
            return False

        if col <= 0 or col >= self.grid_size - 1:
            return False

        if position in obstacles:
            return False

        return True