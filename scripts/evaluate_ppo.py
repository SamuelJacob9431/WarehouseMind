from stable_baselines3 import PPO

from environment import (
    WarehouseEnv,
    WarehouseConfig,
)


config = WarehouseConfig(
    grid_size=10,
    num_packages=2,
    obstacle_count=3,
    max_steps=150,
)


env = WarehouseEnv(
    config=config,
    render_mode="ansi",
)


model = PPO.load(
    "models/ppo_warehouse",
)


for episode in range(5):

    observation, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0

    print()
    print("=" * 50)
    print(f"EPISODE {episode + 1}")
    print("=" * 50)

    while not terminated and not truncated:

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(int(action))
        )

        total_reward += reward

        print(
            f"Step {env.steps:03d} | "
            f"Action: {env.ACTION_NAMES[int(action)]:5s} | "
            f"Reward: {reward:7.2f} | "
            f"Robot: {info['robot_position']} | "
            f"Packages: {info['packages_remaining']} | "
            f"Carrying: {info['carrying']}"
        )

    print()
    print(env.render())

    print(
        f"Total reward: {total_reward:.2f}"
    )

    print(
        f"Delivered: {info['packages_delivered']}"
    )