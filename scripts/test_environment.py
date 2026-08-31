from aislemind.environment import (
    WarehouseConfig,
    WarehouseEnv,
)


def main():

    config = WarehouseConfig(
        grid_size=10,
        num_packages=3,
        obstacle_count=8,
        max_steps=200,
    )

    env = WarehouseEnv(
        config=config,
        render_mode="ansi",
    )

    observation, info = env.reset(seed=42)

    print("=" * 40)
    print("INITIAL WAREHOUSE")
    print("=" * 40)

    print(env.render())

    print("\nObservation:")
    print(observation)

    print("\nInfo:")
    print(info)

    print("\n" + "=" * 40)
    print("RANDOM AGENT")
    print("=" * 40)

    for step in range(50):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step {step + 1:03d} | "
            f"Action: {env.ACTION_NAMES[action]:5s} | "
            f"Reward: {reward:6.1f} | "
            f"Packages: {info['packages_remaining']} | "
            f"Carrying: {info['carrying']}"
        )

        if terminated or truncated:
            break

    print("\nFINAL WAREHOUSE")
    print(env.render())

    env.close()


if __name__ == "__main__":
    main()