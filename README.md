# Coverage-Aware Guidance for Novelty-Driven Exploration (RND+CAE)

**Automated Game Testing in Sparse-Reward 3D Environments**  
Minecraft / Project Malmo · DQN · Random Network Distillation · Coverage-Guided Adaptive Exploration

This repository contains the research implementation for combining **novelty-driven exploration** with **coverage-aware guidance** in reinforcement-learning-based automated game testing.

## Motivation

In sparse-reward 3D environments, RL agents can become concentrated on a small subset of trajectories and interactions. RND provides a novelty signal, but novelty alone does not always encourage systematic coverage of task-relevant states and interactions.

This work introduces **Coverage-Guided Adaptive Exploration (CAE)** and combines it with RND to encourage broader, more structured exploration.

## Method

The agent is trained with a reward composed of the external task reward and intrinsic exploration signals:

- **RND novelty reward** — predictor-target prediction error
- **CAE coverage reward** — visit-count-based coverage guidance and stagnation recovery

The combined intrinsic signal is integrated with the external reward before the DQN update.

Conceptually:

```text
Observation
   │
   ├──> RND predictor / target ──> novelty reward
   │
   ├──> coverage tracker ────────> CAE reward
   │
   └──> environment reward
                │
                ▼
        combined training reward
                │
                ▼
               DQN
```

## Environments

The experiments use **Project Malmo / Minecraft** with two complementary settings.

### Randomized Maze
Used to evaluate spatial exploration and cumulative coverage.

### Partitioned Arena
Used to evaluate interaction-driven exploration and seeded-fault discovery.

The Arena configuration includes structured observations such as position/orientation, voxel information, and inventory state, together with a larger discrete action space.

## Evaluation

Experiments were conducted under the same training budget with repeated runs and multi-seed evaluation.

Representative Arena results:

| Method | Final Unique Bugs | Bug Discovery AUC |
|---|---:|---:|
| Random | 3.50 ± 0.53 | 3.10 ± 0.53 |
| RELINE | 5.50 ± 0.97 | 4.27 ± 0.54 |
| BEAGT | 4.80 ± 1.81 | 4.20 ± 0.47 |
| RND | 4.70 ± 1.95 | 4.30 ± 0.64 |
| CAE | 5.20 ± 1.40 | 4.23 ± 0.79 |
| **RND+CAE** | **5.70 ± 1.06** | **4.66 ± 0.67** |

The evaluation also includes Maze coverage, repeated runs, ablation studies, and fault-discovery analysis.

## Repository Structure

```text
coverageguidedexploration/
├── malmo_bug_project/
│   ├── agents/
│   │   ├── Maze_BEAGT.py
│   │   ├── Maze_CAE.py
│   │   ├── Maze_RL.py
│   │   ├── train_BEAGT.py
│   │   ├── train_CAE.py
│   │   └── train_RELINE.py
│   └── envs/
│       ├── bug_definitions.json
│       ├── bug_detector.py
│       ├── callbacks.py
│       ├── malmo_env.py
│       └── simple_voxel_env.py
└── README.md
```

## My Contribution

My contribution to this work included:

- building the randomized Maze and Arena benchmarks in Project Malmo,
- implementing interaction-based seeded-fault logging,
- building the DQN/RND training pipeline,
- designing and integrating CAE reward signals,
- conducting multi-seed comparisons and ablation studies,
- evaluating coverage and fault-discovery behavior.

## Environment Setup

**Tested OS:** Ubuntu 20.04.6 LTS

### 1. Install Project Malmo

Follow the official Project Malmo setup instructions:

https://github.com/microsoft/malmo

> [!IMPORTANT]
> This research environment was developed on an older Ubuntu/Malmo stack. Newer operating-system or dependency versions may require compatibility fixes.

### 2. Place the project directory

Move `malmo_bug_project` into your Malmo installation directory, then navigate to it.

```bash
cd MalmoPlatform/malmo_bug_project
```

### 3. Run Maze agents

```bash
python3 agents/Maze_RL.py
python3 agents/Maze_BEAGT.py
python3 agents/Maze_CAE.py
```

### 4. Run Arena / fault-discovery agents

```bash
python3 agents/train_RELINE.py
python3 agents/train_BEAGT.py
python3 agents/train_CAE.py
```

## Publication

**Tae-Hyeon Jang**, Hyeon-Uk Lee, Hyunseok Kim,  
*Coverage-Aware Guidance for Novelty-Driven Exploration in Automated Game Testing under Sparse-Reward 3D Environments*,  
IEEE Access, 2026.  
**First Author**

## Research Scope

This repository focuses on the RND+CAE study and its evaluation environment. It is separate from my current world-model-based policy-adaptation research, whose implementation is not publicly released.
