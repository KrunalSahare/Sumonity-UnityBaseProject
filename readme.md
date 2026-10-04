# Multi-Agent Deep Reinforcement Learning for Traffic Signal Control with Static Emergency Vehicle Pre-emption

[![Platform](https://img.shields.io/badge/Platform-SUMO%20%7C%20Unity%203D-blue.svg)](#)
[![Algorithm](https://img.shields.io/badge/Algorithm-CoLight%20%2B%20DQN%20%2B%20GAT-green.svg)](#)
[![Preemption](https://img.shields.io/badge/Module-Static%20EV%20Preemption-orange.svg)](#)

A hybrid traffic signal management system that combines cooperative Graph-Attention Multi-Agent Reinforcement Learning (MARL) for urban traffic optimization with a static route-based pre-emption mechanism to guarantee swift, uninterrupted clearance for Emergency Vehicles (EVs).

Developed as a Bachelor of Engineering Mini Project in Artificial Intelligence and Machine Learning at **M. S. Ramaiah Institute of Technology (MSRIT)**, Bengaluru[cite: 4].

---

## Authors & Contributors

* **Azeem Ahmad** (1MS22AI011) – *Team Lead & Model Designer*[cite: 4]
* **Krunal Sahare** (1MS22AI024) – *Simulation & Visualization Developer*[cite: 4]
* **Mohamed Sahil** (1MS22AI028) – *Software Developer & UI/UX Designer*[cite: 4]

**Project Guide:** Mrs. SriRaksha PJ, Associate Professor, Department of AI & ML[cite: 4]  
**Head of Department:** Dr. Jagadish S Kallimani, Department of AI & ML[cite: 4]

---

## Overview

Urban traffic congestion critically degrades the response times of emergency services like ambulances, where delays directly risk patient lives[cite: 4]. Conventional reactive traffic systems fail to coordinate across complex intersection grids[cite: 4]. 

This project implements a hybrid control framework:
1. **Network-Level Cooperative Control**: Uses a Graph Attention Network (GAT) combined with Deep Q-Networks (DQN), inspired by the **CoLight** framework, enabling signal controllers across multiple intersections to learn spatial-temporal coordination policies.
2. **Static Route-Based EV Pre-emption**: Overrides the learned MARL actions along an approaching emergency vehicle's pre-planned route, establishing a dynamic "green corridor" and immediately restoring cooperative MARL control once the emergency vehicle clears.
3. **Co-Simulation Bridge (SUMO + Unity)**: Microscopic traffic simulation executed in Eclipse SUMO via TraCI and visualized with realistic vehicle dynamics and 3D digital-twin environments in Unity Engine[cite: 1, 3, 4].

---

## Methodology & Formulation

### 1. State & Action Spaces
* **State ($s_i^t$)**: Each intersection agent $i$ at timestep $t$ observes:
  $$s_i^t = (p_i^t, q_i^t)$$
  where $p_i^t$ is a one-hot encoding of the current signal phase and $q_i^t = (q_{i,1}^t, \dots, q_{i,n_i}^t)$ represents the vehicle queue lengths/occupancies on all $n_i$ incoming lanes.
* **Action ($\mathcal{A}_i$)**: Each agent selects an allowable phase combination $a_i^t \in \mathcal{A}_i$ (e.g., NS-Green, EW-Green) applied for duration $\Delta t$[cite: 3, 4].

### 2. Graph Attention Mechanism (GAT)
Spatial neighbor states within connectivity graph $G = (V, E)$ are aggregated to enable collaborative phase transitions without central bottlenecks[cite: 3]:
1. Agent state is projected into latent space: $h_i^t = \phi_\theta(s_i^t)$[cite: 3].
2. Attention coefficients between neighbor $j \in \mathcal{N}_i$ and agent $i$ are computed:
   $$e_{ij} = \text{LeakyReLU}(w^\top [W_1 h_i^t \parallel W_2 h_j^t])$$
   $$\alpha_{ij} = \frac{\exp(e_{ij} / \tau)}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik} / \tau)}$$
[cite: 3]
3. Neighborhood context representation: $c_i^t = \sum_{j \in \mathcal{N}_i} \alpha_{ij} W_h h_j^t$[cite: 3].

### 3. Q-Value Update Rule
Agents share weights $\theta$ for homogeneous learning, optimized by minimizing TD loss[cite: 3, 4]:
$$\mathcal{L}_i(\theta) = \left( y_i^t - Q_i(s_i^t, a_i^t; \theta) \right)^2$$
$$y_i^t = r^t + \gamma \max_{a'} Q_i(s_i^{t+1}, a'; \theta^-)$$
[cite: 3, 4]

### 4. Combined Multi-Objective Reward Function
Balances overall network delay reduction against high-priority emergency clearance[cite: 3, 4]:
$$r^t = -\left( \lambda D_{\text{norm}}^t + (1 - \lambda) D_{\text{EV}}^t \right), \quad 0 \le \lambda \le 1$$
$$r^t = -\lambda \sum_{v \in V} \tau_v^t - (1 - \lambda) \sum_{e \in E} \tau_e^t$$
where $\tau_v^t$ is the waiting time of standard vehicle $v$, $\tau_e^t$ is the delay of emergency vehicle $e$, and $\lambda < 0.5$ places prioritized weight on EV passage[cite: 3, 4].

### 5. Static EV Pre-emption Override Logic
For a precomputed sequence of route intersections $i_1, i_2, \dots, i_K$:
* $z_i^t = 1$ when an emergency vehicle is approaching intersection $i$ along the route, otherwise $z_i^t = 0$[cite: 3, 4].
* **Control Rule**:
  $$a_i^t = \begin{cases} a_i^{\text{EV}} & \text{if } z_i^t = 1 \text{ (Pre-emption Override)} \\ \arg\max_a Q(s_i^t, a; \theta) & \text{otherwise (MARL Policy)} \end{cases}$$
[cite: 3, 4]

---

## Experimental Setup & Benchmark Topologies

The model is evaluated against real-world and synthetic grid benchmarks[cite: 3, 4]:

| Network Topology | Description | Intersections | Traffic Load |
| :--- | :--- | :---: | :--- |
| **Manhattan $7 \times 28$** | Large-scale rectangular downtown grid[cite: 4] | 196[cite: 4] | High[cite: 4] |
| **Hangzhou $4 \times 4$** | Asymmetric real-world arterial city layout[cite: 3, 4] | 16[cite: 4] | Moderate[cite: 4] |
| **Hangzhou $1 \times 1$** | Microscopic isolated intersection baseline[cite: 4] | 1[cite: 4] | Low to Medium[cite: 4] |
| **Cologne 3** | European road geometry and signal corridor[cite: 4] | 3[cite: 4] | Medium to High[cite: 4] |

### Baselines Compared
* **Fixed-Time**: Manually tuned cyclic signal scheduling (Webster's method)[cite: 3, 4].
* **MaxPressure**: Decentralized pressure-based queue balancing[cite: 3, 4].
* **CoLight**: Graph attention MARL without explicit EV pre-emption[cite: 3, 4].
* **MAGD**: Multi-agent gradient descent baseline[cite: 3, 4].

---

## Key Results & Performance Analysis

Experimental evaluation across 20 randomized seeds per configuration showed[cite: 3]:
* **Emergency Vehicle Travel Time**: Reduced by **30% to 40%** compared to MARL alone, and up to **50% EV delay reduction** over traditional baseline schemes[cite: 3, 4].
* **Average Vehicle Delay**: Reduced by **35–45%** compared to Fixed-Time control and **20–25%** over standalone CoLight and MAGD[cite: 3, 4].
* **Network Throughput**: Improved by **8–15%**, maintaining up to **1,100 vehicles/hour** under heavy Manhattan grid traffic[cite: 3, 4].
* **Queue Lengths**: Consistently maintained the lowest average queue lengths across all topologies without bottlenecking non-emergency lanes[cite: 3, 4].

---

## Repository Structure

```text
├── Assets/                     # Unity scenes, prefabs, vehicle models, and bridge scripts
├── Packages/                   # Unity Package Manager manifests
├── ProjectSettings/            # Engine configuration, physics, and input definitions
├── map/                        # Road networks, SUMO configs (.sumocfg, .net.xml, .rou.xml)[cite: 1, 2]
├── assets.repos                # VCS modular dependency map[cite: 1]
├── setup.bat                   # Automated Windows environment configuration script[cite: 1]
├── setup.ps1                   # PowerShell setup script[cite: 1]
├── vcs_import.sh               # Bash script for importing external dependencies via vcstool[cite: 1]
├── checkGitStatus.py           # Multi-repository git status validation helper[cite: 1]
├── run_scene_automated.ps1     # Headless scene runner for automated testing & CI validation[cite: 1]
├── automation_readme.md        # Extended automation documentation[cite: 1]
├── CITesting_README.md         # CI/CD and automated test suite specifications[cite: 1]
└── readme.md                   # Project documentation[cite: 1]
