import sys

import numpy as np
import pygame
import torch

from stable_baselines3 import PPO

from environment import WarehouseEnv, WarehouseConfig


CELL_SIZE = 60
SIDEBAR_WIDTH = 320
FPS = 6

ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
]


def draw_text(
    screen,
    text,
    position,
    font,
):

    surface = font.render(
        text,
        True,
        (230, 230, 230),
    )

    screen.blit(
        surface,
        position,
    )


def get_action_probabilities(
    model,
    observation,
):

    observation_tensor, _ = (
        model.policy.obs_to_tensor(
            observation
        )
    )

    with torch.no_grad():

        distribution = (
            model.policy.get_distribution(
                observation_tensor
            )
        )

        probabilities = (
            distribution.distribution.probs
        )

    return probabilities.cpu().numpy()[0]


def main():

    pygame.init()

    config = WarehouseConfig(
        grid_size=10,
        num_packages=2,
        obstacle_count=3,
        max_steps=150,
    )

    env = WarehouseEnv(
        config=config
    )

    model = PPO.load(
        "models/ppo_warehouse",
        env=env,
    )

    width = (
        config.grid_size * CELL_SIZE
        + SIDEBAR_WIDTH
    )

    height = (
        config.grid_size * CELL_SIZE
    )

    screen = pygame.display.set_mode(
        (width, height)
    )

    pygame.display.set_caption(
        "WarehouseMind — PPO Agent"
    )

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(
        "Arial",
        24,
    )

    font = pygame.font.SysFont(
        "Arial",
        20,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        17,
    )

    observation, info = env.reset(
        seed=42
    )

    running = True
    paused = False

    probabilities = np.zeros(4)

    last_action = 0
    last_reward = 0.0

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:

                    observation, info = (
                        env.reset()
                    )

                    last_reward = 0.0

        if not paused:

            probabilities = (
                get_action_probabilities(
                    model,
                    observation,
                )
            )

            action = int(
                np.argmax(probabilities)
            )

            last_action = action

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            last_reward = reward

            if terminated or truncated:

                pygame.time.wait(1000)

                observation, info = (
                    env.reset()
                )

       
        # Drawin!
        

        screen.fill(
            (25, 25, 30)
        )

        # Grid
        for row in range(
            config.grid_size
        ):

            for col in range(
                config.grid_size
            ):

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )

                pygame.draw.rect(
                    screen,
                    (45, 45, 50),
                    rect,
                )

                pygame.draw.rect(
                    screen,
                    (70, 70, 75),
                    rect,
                    1,
                )

        # Obstacles
        for row, col in env.obstacles:

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            pygame.draw.rect(
                screen,
                (90, 90, 95),
                rect,
            )

        # Packages
        for package in env.packages:

            if package.delivered:
                continue

            row, col = package.position

            center = (
                col * CELL_SIZE
                + CELL_SIZE // 2,
                row * CELL_SIZE
                + CELL_SIZE // 2,
            )

            pygame.draw.circle(
                screen,
                (240, 190, 60),
                center,
                17,
            )

            draw_text(
                screen,
                "P",
                (
                    center[0] - 7,
                    center[1] - 11,
                ),
                font,
            )

        # Delivery zone
        row, col = (
            env.delivery_zone.position
        )

        rect = pygame.Rect(
            col * CELL_SIZE + 8,
            row * CELL_SIZE + 8,
            CELL_SIZE - 16,
            CELL_SIZE - 16,
        )

        pygame.draw.rect(
            screen,
            (70, 150, 90),
            rect,
        )

        draw_text(
            screen,
            "D",
            (
                col * CELL_SIZE + 20,
                row * CELL_SIZE + 14,
            ),
            font,
        )

        # Robot
        row, col = env.robot.position

        center = (
            col * CELL_SIZE
            + CELL_SIZE // 2,
            row * CELL_SIZE
            + CELL_SIZE // 2,
        )

        pygame.draw.circle(
            screen,
            (80, 150, 240),
            center,
            20,
        )

        draw_text(
            screen,
            "R",
            (
                center[0] - 7,
                center[1] - 12,
            ),
            font,
        )

       #Sidebaaro
        sidebar_x = (
            config.grid_size
            * CELL_SIZE
            + 20
        )

        draw_text(
            screen,
            "WarehouseMind",
            (sidebar_x, 20),
            title_font,
        )

        draw_text(
            screen,
            "PPO Agent",
            (sidebar_x, 55),
            font,
        )

        draw_text(
            screen,
            f"Step: {env.steps}",
            (sidebar_x, 100),
            small_font,
        )

        draw_text(
            screen,
            f"Reward: {env.total_reward:.1f}",
            (sidebar_x, 125),
            small_font,
        )

        draw_text(
            screen,
            f"Packages: {env._remaining_packages()}",
            (sidebar_x, 150),
            small_font,
        )

        draw_text(
            screen,
            f"Carrying: {env.robot.carrying}",
            (sidebar_x, 175),
            small_font,
        )

        # --------------------------------------------------
        # ACTION PROBABILITIES
        # --------------------------------------------------

        draw_text(
            screen,
            "ACTION PROBABILITIES",
            (sidebar_x, 220),
            font,
        )

        for index, name in enumerate(
            ACTION_NAMES
        ):

            probability = (
                probabilities[index]
            )

            y = 260 + index * 45

            draw_text(
                screen,
                f"{name}: {probability * 100:.1f}%",
                (sidebar_x, y),
                small_font,
            )

            # Probability bar
            bar_x = sidebar_x
            bar_y = y + 23

            bar_width = 240
            bar_height = 8

            pygame.draw.rect(
                screen,
                (60, 60, 65),
                (
                    bar_x,
                    bar_y,
                    bar_width,
                    bar_height,
                ),
            )

            pygame.draw.rect(
                screen,
                (100, 160, 230),
                (
                    bar_x,
                    bar_y,
                    int(
                        bar_width
                        * probability
                    ),
                    bar_height,
                ),
            )

        # --------------------------------------------------
        # DECISION
        # --------------------------------------------------

        draw_text(
            screen,
            "PPO DECISION",
            (sidebar_x, 470),
            font,
        )

        draw_text(
            screen,
            f"Selected: {ACTION_NAMES[last_action]}",
            (sidebar_x, 505),
            small_font,
        )

        draw_text(
            screen,
            f"Reward: {last_reward:+.1f}",
            (sidebar_x, 530),
            small_font,
        )

        draw_text(
            screen,
            "SPACE = Pause",
            (sidebar_x, 570),
            small_font,
        )

        draw_text(
            screen,
            "R = Reset",
            (sidebar_x, 595),
            small_font,
        )

        draw_text(
            screen,
            "ESC = Quit",
            (sidebar_x, 620),
            small_font,
        )

        pygame.display.flip()

        clock.tick(FPS)

    env.close()

    pygame.quit()

    sys.exit()


if __name__ == "__main__":
    main()