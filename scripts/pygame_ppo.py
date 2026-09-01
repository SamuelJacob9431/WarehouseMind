import sys
import math

import numpy as np
import pygame
import torch

from stable_baselines3 import PPO

from environment import WarehouseEnv, WarehouseConfig


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900

SIDEBAR_WIDTH = 370
TOPBAR_HEIGHT = 72
STATUSBAR_HEIGHT = 42

FPS = 8

MODEL_PATH = "models/ppo_warehouse"


ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
]

ACTION_SYMBOLS = [
    "↑",
    "↓",
    "←",
    "→",
]


# ============================================================
# COLORS
# ============================================================

BG = (13, 15, 20)
PANEL = (18, 21, 28)
PANEL_LIGHT = (24, 28, 36)

GRID = (48, 52, 61)
GRID_BORDER = (63, 68, 78)

TEXT = (235, 238, 245)
TEXT_DIM = (155, 162, 175)

BLUE = (65, 145, 240)
BLUE_LIGHT = (85, 170, 255)

GREEN = (70, 180, 105)
GREEN_LIGHT = (100, 210, 130)

YELLOW = (240, 185, 55)
YELLOW_LIGHT = (255, 205, 80)

RED = (235, 80, 90)

OBSTACLE = (67, 71, 80)

BAR_BACKGROUND = (42, 46, 54)


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(
    screen,
    text,
    position,
    font,
    color=TEXT,
):
    surface = font.render(
        str(text),
        True,
        color,
    )

    screen.blit(
        surface,
        position,
    )


def draw_centered_text(
    screen,
    text,
    center,
    font,
    color=TEXT,
):
    surface = font.render(
        str(text),
        True,
        color,
    )

    rect = surface.get_rect(
        center=center
    )

    screen.blit(
        surface,
        rect,
    )


def rounded_rect(
    screen,
    color,
    rect,
    radius=10,
    border_color=None,
    border_width=1,
):
    pygame.draw.rect(
        screen,
        color,
        rect,
        border_radius=radius,
    )

    if border_color is not None:
        pygame.draw.rect(
            screen,
            border_color,
            rect,
            width=border_width,
            border_radius=radius,
        )


# ============================================================
# ICONS
# ============================================================

def draw_robot_icon(
    screen,
    center,
    size,
):
    """
    Draw a simple PN warehouse robot icon.
    """

    x, y = center

    rect_size = int(size * 0.82)

    rect = pygame.Rect(
        x - rect_size // 2,
        y - rect_size // 2,
        rect_size,
        rect_size,
    )

    rounded_rect(
        screen,
        BLUE,
        rect,
        radius=max(6, size // 7),
    )

    # Slight inner highlight
    inner = rect.inflate(
        -max(4, size // 10),
        -max(4, size // 10),
    )

    pygame.draw.rect(
        screen,
        BLUE_LIGHT,
        inner,
        width=2,
        border_radius=max(4, size // 8),
    )

    font_size = max(
        12,
        int(size * 0.34),
    )

    font = pygame.font.SysFont(
        "Arial",
        font_size,
        bold=True,
    )

    draw_centered_text(
        screen,
        "PN",
        (x, y),
        font,
        (255, 255, 255),
    )


def draw_package_icon(
    screen,
    center,
    size,
):
    """
    Draw a cardboard package using polygons.
    """

    x, y = center

    box = size * 0.48

    # Front face
    front = [
        (x - box, y - box * 0.55),
        (x, y - box * 0.25),
        (x, y + box),
        (x - box, y + box * 0.55),
    ]

    # Right face
    right = [
        (x, y - box * 0.25),
        (x + box, y - box * 0.55),
        (x + box, y + box * 0.55),
        (x, y + box),
    ]

    # Top face
    top = [
        (x - box, y - box * 0.55),
        (x, y - box * 0.90),
        (x + box, y - box * 0.55),
        (x, y - box * 0.25),
    ]

    pygame.draw.polygon(
        screen,
        (194, 139, 55),
        front,
    )

    pygame.draw.polygon(
        screen,
        (164, 111, 38),
        right,
    )

    pygame.draw.polygon(
        screen,
        (238, 190, 95),
        top,
    )

    # Tape
    tape_width = max(
        4,
        int(size * 0.10),
    )

    pygame.draw.line(
        screen,
        (245, 220, 155),
        (x, y - box * 0.88),
        (x, y + box * 0.88),
        tape_width,
    )

    # Outline
    pygame.draw.lines(
        screen,
        (105, 75, 35),
        True,
        front,
        2,
    )

    pygame.draw.lines(
        screen,
        (105, 75, 35),
        True,
        right,
        2,
    )

    pygame.draw.lines(
        screen,
        (105, 75, 35),
        True,
        top,
        2,
    )


def draw_delivery_icon(
    screen,
    center,
    size,
):
    x, y = center

    rect_size = int(size * 0.72)

    rect = pygame.Rect(
        x - rect_size // 2,
        y - rect_size // 2,
        rect_size,
        rect_size,
    )

    rounded_rect(
        screen,
        GREEN,
        rect,
        radius=7,
    )

    font = pygame.font.SysFont(
        "Arial",
        max(14, int(size * 0.34)),
        bold=True,
    )

    draw_centered_text(
        screen,
        "D",
        (x, y),
        font,
        (255, 255, 255),
    )


# ============================================================
# PPO
# ============================================================

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


# ============================================================
# MAIN VISUALIZER
# ============================================================

def main():

    pygame.init()

    pygame.display.set_caption(
        "WarehouseMind — PPO Agent"
    )

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,
    )

    clock = pygame.time.Clock()

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    title_font = pygame.font.SysFont(
        "Arial",
        26,
        bold=True,
    )

    heading_font = pygame.font.SysFont(
        "Arial",
        19,
        bold=True,
    )

    normal_font = pygame.font.SysFont(
        "Arial",
        17,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        14,
    )

    tiny_font = pygame.font.SysFont(
        "Arial",
        12,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    config = WarehouseConfig(
        grid_size=10,
        num_packages=2,
        obstacle_count=3,
        max_steps=150,
    )

    env = WarehouseEnv(
        config=config
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PPO.load(
        MODEL_PATH,
        env=env,
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    episode = 1

    observation, info = env.reset(
        seed=42
    )

    probabilities = np.zeros(
        4,
        dtype=np.float32,
    )

    last_action = 0
    last_reward = 0.0

    paused = False
    fullscreen = False

    running = True

    # ========================================================
    # LOOP
    # ========================================================

    while running:

        # ====================================================
        # EVENTS
        # ====================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:

                if not fullscreen:
                    screen = pygame.display.set_mode(
                        event.size,
                        pygame.RESIZABLE,
                    )

            elif event.type == pygame.KEYDOWN:

                # --------------------------------------------
                # Quit
                # --------------------------------------------

                if event.key == pygame.K_ESCAPE:

                    running = False

                # --------------------------------------------
                # Fullscreen
                # --------------------------------------------

                elif event.key == pygame.K_F11:

                    fullscreen = not fullscreen

                    if fullscreen:

                        screen = pygame.display.set_mode(
                            (0, 0),
                            pygame.FULLSCREEN,
                        )

                    else:

                        screen = pygame.display.set_mode(
                            (
                                WINDOW_WIDTH,
                                WINDOW_HEIGHT,
                            ),
                            pygame.RESIZABLE,
                        )

                # --------------------------------------------
                # Pause
                # --------------------------------------------

                elif event.key == pygame.K_SPACE:

                    paused = not paused

                # --------------------------------------------
                # Reset
                # --------------------------------------------

                elif event.key == pygame.K_r:

                    episode += 1

                    observation, info = (
                        env.reset()
                    )

                    probabilities = np.zeros(
                        4,
                        dtype=np.float32,
                    )

                    last_reward = 0.0
                    last_action = 0

                # --------------------------------------------
                # Single PPO step
                # --------------------------------------------

                elif (
                    event.key == pygame.K_s
                    and paused
                ):

                    probabilities = (
                        get_action_probabilities(
                            model,
                            observation,
                        )
                    )

                    action = int(
                        np.argmax(
                            probabilities
                        )
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

                        episode += 1

                        observation, info = (
                            env.reset()
                        )

                # --------------------------------------------
                # Manual movement while paused
                # --------------------------------------------

                elif paused:

                    manual_action = None

                    if event.key == pygame.K_UP:
                        manual_action = env.UP

                    elif event.key == pygame.K_DOWN:
                        manual_action = env.DOWN

                    elif event.key == pygame.K_LEFT:
                        manual_action = env.LEFT

                    elif event.key == pygame.K_RIGHT:
                        manual_action = env.RIGHT

                    if manual_action is not None:

                        last_action = manual_action

                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(
                            manual_action
                        )

                        last_reward = reward

                        probabilities = (
                            get_action_probabilities(
                                model,
                                observation,
                            )
                        )

                        if terminated or truncated:

                            episode += 1

                            observation, info = (
                                env.reset()
                            )

        # ====================================================
        # PPO STEP
        # ====================================================

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

                pygame.time.wait(800)

                episode += 1

                observation, info = (
                    env.reset()
                )

                probabilities = np.zeros(
                    4,
                    dtype=np.float32,
                )

        # ====================================================
        # SCREEN SIZE
        # ====================================================

        screen_width, screen_height = (
            screen.get_size()
        )

        sidebar_x = (
            screen_width - SIDEBAR_WIDTH
        )

        board_left = 20

        board_top = (
            TOPBAR_HEIGHT + 15
        )

        board_bottom = (
            screen_height
            - STATUSBAR_HEIGHT
            - 15
        )

        board_width = (
            sidebar_x
            - board_left
            - 20
        )

        board_height = (
            board_bottom
            - board_top
        )

        board_size = min(
            board_width,
            board_height,
        )

        board_left = (
            board_left
            + (
                board_width
                - board_size
            ) // 2
        )

        board_top = (
            board_top
            + (
                board_height
                - board_size
            ) // 2
        )

        cell_size = (
            board_size
            / config.grid_size
        )

        # ====================================================
        # BACKGROUND
        # ====================================================

        screen.fill(BG)

        # ====================================================
        # TOP BAR
        # ====================================================

        pygame.draw.rect(
            screen,
            PANEL,
            (
                0,
                0,
                screen_width,
                TOPBAR_HEIGHT,
            ),
        )

        # Logo

        draw_robot_icon(
            screen,
            (42, 36),
            42,
        )

        draw_text(
            screen,
            "WarehouseMind",
            (70, 14),
            title_font,
            BLUE_LIGHT,
        )

        draw_text(
            screen,
            "PPO Agent",
            (72, 43),
            small_font,
            TEXT_DIM,
        )

        # Controls

        control_x = 270

        controls = [
            ("F11", "Fullscreen"),
            ("SPACE", "Pause / Resume"),
            ("R", "Reset"),
        ]

        for key, label in controls:

            key_width = (
                54
                if key != "SPACE"
                else 72
            )

            total_width = (
                key_width + 100
            )

            rect = pygame.Rect(
                control_x,
                15,
                total_width,
                42,
            )

            rounded_rect(
                screen,
                PANEL_LIGHT,
                rect,
                radius=7,
                border_color=GRID_BORDER,
            )

            draw_centered_text(
                screen,
                key,
                (
                    control_x
                    + key_width // 2
                    + 5,
                    36,
                ),
                small_font,
                TEXT,
            )

            draw_text(
                screen,
                label,
                (
                    control_x
                    + key_width
                    + 15,
                    27,
                ),
                small_font,
                TEXT_DIM,
            )

            control_x += (
                total_width + 10
            )

        # ====================================================
        # WAREHOUSE BOARD
        # ====================================================

        board_rect = pygame.Rect(
            board_left,
            board_top,
            int(board_size),
            int(board_size),
        )

        rounded_rect(
            screen,
            PANEL,
            board_rect,
            radius=10,
            border_color=GRID_BORDER,
        )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        for row in range(
            config.grid_size
        ):

            for col in range(
                config.grid_size
            ):

                x = int(
                    board_left
                    + col * cell_size
                )

                y = int(
                    board_top
                    + row * cell_size
                )

                rect = pygame.Rect(
                    x,
                    y,
                    math.ceil(cell_size),
                    math.ceil(cell_size),
                )

                pygame.draw.rect(
                    screen,
                    PANEL,
                    rect,
                )

                pygame.draw.rect(
                    screen,
                    GRID,
                    rect,
                    1,
                )

        # ----------------------------------------------------
        # Obstacles
        # ----------------------------------------------------

        for row, col in env.obstacles:

            x = int(
                board_left
                + col * cell_size
            )

            y = int(
                board_top
                + row * cell_size
            )

            rect = pygame.Rect(
                x + 1,
                y + 1,
                int(cell_size) - 2,
                int(cell_size) - 2,
            )

            rounded_rect(
                screen,
                OBSTACLE,
                rect,
                radius=4,
            )

        # ----------------------------------------------------
        # Packages
        # ----------------------------------------------------

        for package in env.packages:

            if package.delivered:
                continue

            # If carrying a package, don't render it
            # at its original location.
            if (
                env.robot.carrying
                and package.position
                == env.robot.position
            ):
                continue

            row, col = package.position

            center = (
                int(
                    board_left
                    + col * cell_size
                    + cell_size / 2
                ),
                int(
                    board_top
                    + row * cell_size
                    + cell_size / 2
                ),
            )

            draw_package_icon(
                screen,
                center,
                int(cell_size * 0.72),
            )

        # ----------------------------------------------------
        # Delivery
        # ----------------------------------------------------

        row, col = (
            env.delivery_zone.position
        )

        center = (
            int(
                board_left
                + col * cell_size
                + cell_size / 2
            ),
            int(
                board_top
                + row * cell_size
                + cell_size / 2
            ),
        )

        draw_delivery_icon(
            screen,
            center,
            int(cell_size * 0.72),
        )

        # ----------------------------------------------------
        # Robot
        # ----------------------------------------------------

        row, col = env.robot.position

        center = (
            int(
                board_left
                + col * cell_size
                + cell_size / 2
            ),
            int(
                board_top
                + row * cell_size
                + cell_size / 2
            ),
        )

        draw_robot_icon(
            screen,
            center,
            int(cell_size * 0.76),
        )

        # ====================================================
        # SIDEBAR
        # ====================================================

        pygame.draw.rect(
            screen,
            PANEL,
            (
                sidebar_x,
                TOPBAR_HEIGHT,
                SIDEBAR_WIDTH,
                screen_height
                - TOPBAR_HEIGHT
                - STATUSBAR_HEIGHT,
            ),
        )

        sidebar_padding = 20

        sx = (
            sidebar_x
            + sidebar_padding
        )

        available_width = (
            SIDEBAR_WIDTH
            - sidebar_padding * 2
        )

        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------

        draw_text(
            screen,
            "EPISODE STATUS",
            (sx, 95),
            heading_font,
            BLUE_LIGHT,
        )

        card_y = 130

        stat_width = (
            available_width - 10
        ) / 2

        stat_height = 62

        stats = [
            ("STEP", str(env.steps)),
            (
                "REWARD",
                f"{env.total_reward:.1f}",
            ),
            (
                "PACKAGES",
                str(env._remaining_packages()),
            ),
            (
                "CARRYING",
                "YES"
                if env.robot.carrying
                else "NO",
            ),
        ]

        for i, (label, value) in enumerate(
            stats
        ):

            col = i % 2
            row = i // 2

            x = (
                sx
                + col
                * (
                    stat_width + 10
                )
            )

            y = (
                card_y
                + row
                * (
                    stat_height + 10
                )
            )

            rect = pygame.Rect(
                int(x),
                y,
                int(stat_width),
                stat_height,
            )

            rounded_rect(
                screen,
                PANEL_LIGHT,
                rect,
                radius=8,
                border_color=GRID_BORDER,
            )

            draw_text(
                screen,
                label,
                (
                    int(x + 12),
                    y + 9,
                ),
                tiny_font,
                TEXT_DIM,
            )

            value_color = TEXT

            if label == "REWARD":

                if env.total_reward < 0:
                    value_color = RED

                else:
                    value_color = GREEN_LIGHT

            elif label == "CARRYING":

                if env.robot.carrying:
                    value_color = YELLOW_LIGHT

            draw_text(
                screen,
                value,
                (
                    int(x + 12),
                    y + 29,
                ),
                heading_font,
                value_color,
            )

        # ----------------------------------------------------
        # Action probabilities
        # ----------------------------------------------------

        prob_title_y = 280

        draw_text(
            screen,
            "ACTION PROBABILITIES",
            (sx, prob_title_y),
            heading_font,
            BLUE_LIGHT,
        )

        bar_start_y = (
            prob_title_y + 38
        )

        bar_height = 8

        for i, name in enumerate(
            ACTION_NAMES
        ):

            y = (
                bar_start_y
                + i * 46
            )

            probability = float(
                probabilities[i]
            )

            draw_text(
                screen,
                ACTION_SYMBOLS[i],
                (sx, y - 2),
                heading_font,
                TEXT,
            )

            draw_text(
                screen,
                name,
                (sx + 30, y),
                small_font,
                TEXT,
            )

            percentage = (
                probability * 100
            )

            draw_text(
                screen,
                f"{percentage:5.1f}%",
                (
                    sidebar_x
                    + SIDEBAR_WIDTH
                    - 82,
                    y,
                ),
                small_font,
                TEXT,
            )

            bar_x = sx + 85

            bar_width = (
                available_width - 170
            )

            pygame.draw.rect(
                screen,
                BAR_BACKGROUND,
                (
                    bar_x,
                    y + 24,
                    bar_width,
                    bar_height,
                ),
                border_radius=4,
            )

            pygame.draw.rect(
                screen,
                BLUE,
                (
                    bar_x,
                    y + 24,
                    int(
                        bar_width
                        * probability
                    ),
                    bar_height,
                ),
                border_radius=4,
            )

        # ----------------------------------------------------
        # PPO Decision
        # ----------------------------------------------------

        decision_y = 510

        draw_text(
            screen,
            "PPO DECISION",
            (sx, decision_y),
            heading_font,
            BLUE_LIGHT,
        )

        decision_rect = pygame.Rect(
            sx,
            decision_y + 34,
            available_width,
            82,
        )

        rounded_rect(
            screen,
            PANEL_LIGHT,
            decision_rect,
            radius=8,
            border_color=GRID_BORDER,
        )

        draw_text(
            screen,
            "SELECTED ACTION",
            (
                sx + 15,
                decision_y + 46,
            ),
            tiny_font,
            TEXT_DIM,
        )

        draw_text(
            screen,
            (
                f"{ACTION_SYMBOLS[last_action]} "
                f"{ACTION_NAMES[last_action]}"
            ),
            (
                sx + 15,
                decision_y + 64,
            ),
            heading_font,
            BLUE_LIGHT,
        )

        reward_color = (
            GREEN_LIGHT
            if last_reward >= 0
            else RED
        )

        draw_text(
            screen,
            f"{last_reward:+.1f}",
            (
                sx
                + available_width
                - 65,
                decision_y + 62,
            ),
            heading_font,
            reward_color,
        )

        # ----------------------------------------------------
        # Pause indicator
        # ----------------------------------------------------

        pause_y = 625

        pause_rect = pygame.Rect(
            sx,
            pause_y,
            available_width,
            42,
        )

        if paused:

            rounded_rect(
                screen,
                (45, 40, 25),
                pause_rect,
                radius=7,
                border_color=YELLOW,
            )

            draw_centered_text(
                screen,
                "Ⅱ   PAUSED",
                pause_rect.center,
                small_font,
                YELLOW_LIGHT,
            )

        else:

            rounded_rect(
                screen,
                (25, 45, 32),
                pause_rect,
                radius=7,
                border_color=GREEN,
            )

            draw_centered_text(
                screen,
                "●   RUNNING",
                pause_rect.center,
                small_font,
                GREEN_LIGHT,
            )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        controls_y = 690

        draw_text(
            screen,
            "CONTROLS",
            (sx, controls_y),
            heading_font,
            BLUE_LIGHT,
        )

        control_items = [
            ("SPACE", "Pause / Resume"),
            ("R", "Reset Episode"),
            ("F11", "Toggle Fullscreen"),
            ("ESC", "Quit"),
            ("↑ ↓ ← →", "Manual Step"),
            ("S", "PPO Step"),
        ]

        y = controls_y + 34

        for key, description in control_items:

            key_rect = pygame.Rect(
                sx,
                y,
                72,
                23,
            )

            rounded_rect(
                screen,
                (48, 52, 62),
                key_rect,
                radius=4,
            )

            draw_centered_text(
                screen,
                key,
                key_rect.center,
                tiny_font,
                TEXT,
            )

            draw_text(
                screen,
                description,
                (
                    sx + 84,
                    y + 4,
                ),
                tiny_font,
                TEXT_DIM,
            )

            y += 28

        # ====================================================
        # STATUS BAR
        # ====================================================

        status_y = (
            screen_height
            - STATUSBAR_HEIGHT
        )

        pygame.draw.rect(
            screen,
            PANEL_LIGHT,
            (
                0,
                status_y,
                screen_width,
                STATUSBAR_HEIGHT,
            ),
        )

        draw_text(
            screen,
            f"Episode: {episode}",
            (20, status_y + 12),
            small_font,
            TEXT,
        )

        draw_text(
            screen,
            "|",
            (125, status_y + 12),
            small_font,
            TEXT_DIM,
        )

        draw_text(
            screen,
            f"Total Reward: {env.total_reward:.1f}",
            (145, status_y + 12),
            small_font,
            (
                GREEN_LIGHT
                if env.total_reward >= 0
                else RED
            ),
        )

        draw_text(
            screen,
            "|",
            (310, status_y + 12),
            small_font,
            TEXT_DIM,
        )

        draw_text(
            screen,
            f"Steps: {env.steps} / {config.max_steps}",
            (330, status_y + 12),
            small_font,
            TEXT,
        )

        draw_text(
            screen,
            "|",
            (470, status_y + 12),
            small_font,
            TEXT_DIM,
        )

        draw_text(
            screen,
            f"Packages Delivered: "
            f"{info['packages_delivered']}/"
            f"{config.num_packages}",
            (490, status_y + 12),
            small_font,
            TEXT,
        )

        draw_text(
            screen,
            "ESC to quit",
            (
                screen_width - 100,
                status_y + 12,
            ),
            small_font,
            TEXT_DIM,
        )

        # ====================================================
        # UPDATE
        # ====================================================

        pygame.display.flip()

        clock.tick(FPS)

    # ========================================================
    # CLEANUP
    # ========================================================

    env.close()

    pygame.quit()

    sys.exit()


if __name__ == "__main__":
    main()