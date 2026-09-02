# WarehouseMind

A custom reinforcement learning warehouse environment where an autonomous robot learns to collect packages and deliver them to a designated delivery zone.

The current implementation focuses on training and visualizing a **PPO (Proximal Policy Optimization) agent** inside a custom Gymnasium environment.

---

## Overview

WarehouseMind is a small warehouse navigation environment designed as a foundation for experimenting with reinforcement learning agents.

The agent controls a warehouse robot that must:

1. Navigate through the warehouse.
2. Avoid obstacles.
3. Locate packages.
4. Pick up packages.
5. Navigate to the delivery zone.
6. Deliver packages.
7. Repeat until all packages are delivered.

The environment is implemented using **Gymnasium**, while the PPO agent is trained using **Stable-Baselines3**.

A **Pygame visualization** provides a real-time view of the warehouse and exposes the PPO agent's action probabilities and decisions.

---

## Current Features

- Custom Gymnasium environment
- Grid-based warehouse
- Configurable warehouse size
- Randomly generated obstacles
- Multiple packages
- Delivery zone
- Robot package carrying state
- Collision handling
- Reward system
- PPO agent
- Action probability visualization
- Real-time Pygame visualization
- Deterministic environments through random seeds
- Automated environment tests

---

---

# Environment

The warehouse is represented as a discrete grid.

The robot can perform four actions:

| Action | Movement |
| ------ | -------- |
| `0`    | UP       |
| `1`    | DOWN     |
| `2`    | LEFT     |
| `3`    | RIGHT    |

The action space is:

```python
spaces.Discrete(4)
```

---

## Observation Space

The PPO agent receives a 12-dimensional observation vector.

The observation contains information about:

* Robot position
* Current target position
* Delivery zone position
* Whether the robot is carrying a package
* Whether movement is blocked in each direction
* Number of remaining packages

The observation space is normalized to approximately:

```text
[-1, 1]
```

This makes the environment suitable for standard neural-network-based RL algorithms.

---

# Reward System

The environment uses a simple reward structure to encourage efficient package delivery.

| Event                        | Reward |
| ---------------------------- | -----: |
| Normal movement              |   `-1` |
| Collision / invalid movement |   `-5` |
| Pick up package              |  `+20` |
| Deliver package              | `+100` |
| Complete all deliveries      | `+200` |

The negative movement reward encourages the agent to find shorter routes instead of wandering indefinitely.

The package and delivery rewards provide a strong learning signal toward the actual objective.

---

# PPO Agent

The current primary agent is **Proximal Policy Optimization (PPO)**.

PPO learns a policy that maps warehouse observations to probabilities over the four possible actions.

Conceptually:


Warehouse State
       │
       ▼
   PPO Policy
       │
       ├── UP      →  probability
       ├── DOWN    →  probability
       ├── LEFT    →  probability
       └── RIGHT   →  probability
                    │
                    ▼
                Selected Action

The Pygame interface exposes these probabilities in real time.

Example:


ACTION PROBABILITIES

UP       2.1%
DOWN    12.4%
LEFT     8.2%
RIGHT   77.3%

PPO DECISION

Selected: RIGHT


This makes it possible to observe not only **what the agent does**, but also how strongly its policy prefers each action.

---

# Installation

## Requirements

* Python 3.10+
* Gymnasium
* NumPy
* Stable-Baselines3
* PyTorch
* Pygame
* Pytest

Python 3.13 is currently being used for development.

---

## Install Dependencies

Clone the repository:

```bash
git clone https://github.com/SamuelJacob9431/WarehouseMind
cd WarehouseMind
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

Verify the environment installation:

```bash
python -c "from warehousemind.environment import WarehouseEnv; print('WarehouseMind OK')"
```

---

# Running Tests

Run:

```bash
pytest
```

The tests verify the basic functionality and Gymnasium behaviour of the warehouse environment.

---

# Running the Environment

To run the basic environment:

```bash
python scripts/run_environment.py
```

This allows the environment mechanics to be tested independently of the PPO agent.

---

# Training PPO

The PPO training script trains an agent directly inside the custom warehouse environment.

Run:

```bash
python scripts/train_ppo.py
```

The trained model is saved under:

```text
models/
```

For example:

```text
models/
└── ppo_warehouse.zip
```

Training parameters can be adjusted in the training script.

---

# Pygame Visualization

After training the PPO agent, run:

```bash
python scripts/pygame_visualizer.py
```

The visualizer loads the trained PPO model and runs it inside the warehouse environment.

The interface displays:

* Warehouse grid
* Obstacles
* Robot 🤖
* Packages 📦
* Delivery zone
* Current step
* Episode reward
* Packages remaining
* Carrying state
* PPO action probabilities
* Selected PPO action

---

## Controls

| Key     | Function          |
| ------- | ----------------- |
| `SPACE` | Pause / Resume    |
| `R`     | Reset environment |
| `ESC`   | Exit              |

---

# Example

A typical episode looks conceptually like:

```text
         📦
         │
         │
      🤖 ──────┐
               │
               │
               ▼
               D
```

The PPO agent must learn a policy that minimizes unnecessary movement while successfully completing deliveries.

---

# Why PPO?

PPO was selected as the first agent because it provides a relatively simple and robust baseline for the custom environment.

The project intentionally starts with a single-agent PPO implementation before introducing more complex approaches.

The current goal is not to build the most sophisticated warehouse simulator, but to create a clean RL environment where different types of agents can eventually be compared.

---

# Future Development

WarehouseMind is designed to evolve beyond the initial PPO implementation.

Planned experiments include:

### DQN

```text
WarehouseMind
     │
     └── DQN Agent
```

Compare value-based learning against PPO on the same environment.

### LLM Agent

A future agent can use an LLM to reason about the warehouse state and select actions.

```text
Warehouse State
       │
       ▼
    LLM Agent
       │
       ▼
     Action
       │
       ▼
 Warehouse Environment
```

### Multiple Agents

Multiple warehouse robots could eventually operate simultaneously.

Potential challenges include:

* Collision avoidance
* Task allocation
* Package assignment
* Coordination
* Competition for routes

### OpenEnv

The environment can eventually be adapted for experimentation with OpenEnv-compatible agent workflows.

The long-term goal is to allow the same warehouse environment to serve as a common benchmark for:

```text
             WarehouseMind
                   │
       ┌───────────┼───────────┐
       │           │           │
      PPO         DQN       LLM Agent
       │           │           │
       └───────────┼───────────┘
                   │
             Same Environment
```

This allows different agent architectures to be compared under the same warehouse dynamics.

---

# Design Philosophy

WarehouseMind follows a simple principle:

> **Keep the environment independent from the agent.**

The warehouse environment should not know whether it is being controlled by:

* PPO
* DQN
* An LLM
* A rule-based controller
* A human
* A multi-agent system

The environment only exposes:

```text
Observation → Action → Reward → Next Observation
```

This separation makes it possible to experiment with different agents without rewriting the warehouse simulation.

---

# Tech Stack

| Component              | Technology        |
| ---------------------- | ----------------- |
| Language               | Python            |
| RL Environment         | Gymnasium         |
| RL Algorithm           | PPO               |
| RL Framework           | Stable-Baselines3 |
| Neural Network Backend | PyTorch           |
| Visualization          | Pygame            |
| Numerical Computing    | NumPy             |
| Testing                | Pytest            |

---

# Project Status

### Current

* [x] Custom warehouse environment
* [x] Grid navigation
* [x] Obstacles
* [x] Package collection
* [x] Delivery mechanics
* [x] Reward system
* [x] Gymnasium API
* [x] Environment tests
* [x] PPO training
* [x] Pygame visualization
* [x] PPO action probability display

### Planned

* [ ] Improve PPO training performance
* [ ] DQN implementation
* [ ] Agent comparison
* [ ] LLM agent
* [ ] Multi-agent warehouse
* [ ] OpenEnv integration
* [ ] Hugging Face Space
* [ ] Interactive parameter controls

