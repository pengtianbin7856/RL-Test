# RL-Based Intelligent Traffic Light Control System (SUMO + Python)

This project implements and compares two reinforcement learning (RL) approaches—**Tabular Q-Learning** and **Deep Q-Network (DQN)**—to optimize intelligent traffic signal control at a standard four-way intersection. Built using Python and the **SUMO (Simulation of Urban MObility)** platform, the system demonstrates how AI can dynamically adapt traffic phases to minimize congestion while maintaining realistic operational constraints.

---

## 📌 Core Features & Mechanisms

### 1. Hard Traffic Phase Constraints (`MIN_GREEN_INTERVAL = 15`)

To prevent the erratic and rapid phase switching commonly seen during the early exploration stages of RL models (which causes safety hazards and yellow-light oscillation in real-world traffic), a strict temporal lock is implemented. Once a traffic light phase is switched, **it is strictly forced to remain green for at least 15 seconds** before any subsequent changes are permitted.

### 2. Dual-Penalty Reward Function

The reward design uses a custom objective function to balance global throughput and prevent local gridlocks:

* **Global Queue Optimization**: A baseline reward of `60 - total_queue` is utilized. The reward remains positive when the total network queue is below 60 vehicles, shifting to a negative penalty if congestion escalates beyond that threshold.
* **East-West Deadlock Penalty**: To prevent starvation on minor approaches, an additional massive penalty of `-25` is triggered if the combined queue length of the East and West lanes (`2i_0` and `4i_0`) exceeds 4 vehicles.

### 3. Comprehensive Performance Evaluation

Upon completing the training cycles, the framework automatically generates and exports a high-resolution 6-panel evaluation dashboard (`Result_80ep.png`). This visualization analyzes metrics including cumulative rewards, moving averages, total queue metrics, phase switching frequencies, and neural network loss curves.

---

## 🧠 Algorithm Design

### 1. State Space (State)

The environment perception vector is modeled as a 3-dimensional tensor:
S = [N_lane0, S_lane0, Phase]

* **N_lane0**: Truncated halting vehicle count on the Northbound approach (`1i_0`, capped at 10).
* **S_lane0**: Truncated halting vehicle count on the Southbound approach (`3i_0`, capped at 10).
* **Phase**: The active phase configuration of the target traffic light controller.

### 2. Action Space (Action)

A = {0, 1}

* **`0`**: Maintain the current traffic light phase configuration.
* **`1`**: Trigger a phase transition.

### 3. DQN Network Architecture

The DQN agent utilizes a Multi-Layer Perceptron (MLP) mapping states to action values:

* **Input Layer**: 3 nodes (matching the state vector dimension).
* **Hidden Layer 1**: 128 neurons with Rectified Linear Unit (ReLU) activation.
* **Hidden Layer 2**: 64 neurons with Rectified Linear Unit (ReLU) activation.
* **Output Layer**: 2 nodes (predicting the expected Q-value for each action).

---

## 🛠️ Requirements & Setup

### 1. SUMO Installation

Ensure that SUMO is fully installed and configured on your local system. Update the configuration variable on line 10 to map directly to your local target `.sumocfg` file path:

```python
SUMO_CONFIG = r"D:\SUMO\doc\examples\sumo\simple_nets\cross\cross1ltl\test.sumocfg"

```

> ⚠️ **Important**: Make sure your system environment variables include `SUMO_HOME`. Without this path variable, the Python `traci` bridge connection cannot initialize properly.

### 2. Python Dependencies

Install the required packages using pip:

```bash
pip install torch numpy matplotlib

```

---

## 📊 Hyperparameter Settings

| Hyperparameter | Description | Assigned Value |
| --- | --- | --- |
| `EPISODES` | Total training cycles | 80 |
| `STEP_LIMIT` | Maximum simulation steps per episode | 700 |
| `ALPHA` | Q-Learning learning rate | 0.1 |
| `GAMMA` | Reward discount factor | 0.99 |
| `EPSILON / EPS_DECAY` | Initial exploration rate & decay multiplier | 0.1 / 0.97 |
| `BATCH_SIZE` | Mini-batch sample size for DQN experience replay | 32 |
| `MEMORY_SIZE` | Maximum capacity of the DQN replay buffer | 15000 |
| `LEARNING_RATE_DQN` | Optimization learning rate for DQN training | 0.0008 |

---

## 🚀 Execution

Run the primary training script from your terminal. The script will open the SUMO environment automatically and run sequentially through the **Q-Learning** pipeline followed by the **DQN** execution phase:

```bash
python main.py

```

*(Note: Replace `main.py` with your actual filename if it differs).*

### Terminal Output Preview:

```text
▶ Training Q-Learning...
Q  Episode  1 | Reward:   -3420 | Queue:  12430
...
▶ Training DQN...
DQN Episode  1 | Reward:   -2105 | Queue:  10420
...
✅ Training Complete! 80 Episode analytics have been compiled and exported.

```

---

## 📈 Analytical Output (`Result_80ep.png`)

Once evaluation concludes, a publication-ready figure **`Result_80ep.png`** is saved locally. The dashboard contains six subplots:

1. **Total Reward per Episode**: Raw cumulative episodic reward progression across the 80-episode curve.
2. **Smoothed Reward (Moving Average)**: Filtered trends using a moving average window to highlight long-term policy convergence.
3. **Total Vehicle Queue**: Aggregated vehicle halting metrics over time (lower values indicate optimal traffic fluidization).
4. **Traffic Light Switch Count**: Frequency of active phase interventions (evaluating the stability of structural constraints).
5. **DQN Training Loss (Smoothed)**: Mean Squared Error (MSE) loss trajectory showcasing neural network convergence.
6. **Reward Distribution Comparison**: Comparative boxplot distributions emphasizing reward variance, stability, and median performance metrics across both agents.
