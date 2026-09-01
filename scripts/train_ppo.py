import os

from stable_baselines3 import PPO

from environment import WarehouseEnv, WarehouseConfig


def main():

    os.makedirs("models", exist_ok=True)

    config = WarehouseConfig(
        grid_size=10,
        num_packages=2,
        obstacle_count=3,
        max_steps=150,
    )

    env = WarehouseEnv(
        config=config
    )

    print("Starting PPO training...")
    print()
    print(f"Grid size:       {config.grid_size}")
    print(f"Packages:        {config.num_packages}")
    print(f"Obstacles:       {config.obstacle_count}")
    print(f"Max steps:       {config.max_steps}")
    print()

    model = PPO(
        policy="MlpPolicy",
        env=env,

        # Learning
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,

        # PPO
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,

        # Encourage exploration
        ent_coef=0.01,

        # Small environment -> CPU is perfectly fine
        device="cpu",

        verbose=1,
    )

    model.learn(
        total_timesteps=50_000
    )

    model.save(
        "models/ppo_warehouse"
    )

    print()
    print("==============================")
    print("PPO TRAINING COMPLETE")
    print("==============================")
    print()
    print("Model saved to:")
    print("models/ppo_warehouse.zip")

    env.close()


if __name__ == "__main__":
    main()