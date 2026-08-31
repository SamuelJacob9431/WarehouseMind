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
        num_packages=3,
        obstacle_count=8,
        max_steps=300,
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