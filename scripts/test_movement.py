from environment import WarehouseEnv


env = WarehouseEnv()

for action in range(4):

    observation, info = env.reset(seed=42)

    old_position = env.robot.position

    observation, reward, terminated, truncated, info = env.step(action)

    new_position = env.robot.position

    print(
        f"Action {action}: "
        f"{old_position} -> {new_position}, "
        f"reward={reward}"
    )

env.close()