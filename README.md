# coverageguidedexploration

The Malmo platform is used: https://github.com/microsoft/malmo

## Environment Setup

**OS:** Ubuntu 20.04.6 LTS

### 1. Setup Minecraft Environment
Follow the instructions at the [Malmo Link] to set up the environment.
> [!IMPORTANT]  
> It is highly recommended to use version 20.*  
> *Note: Errors have been reported starting from version 22.*

### 2. Project Installation
1. Move the `malmo_bug_project` folder into the generated `MalmoPlatform` directory.
2. Navigate to the project directory:
   ```bash
   cd MalmoPlatform/malmo_bug_project

### 3. Running the Agent
Once the environment is set up, you can start training the agent by selecting one of the available algorithms.

- To train using RELINE
  ```bash
  python3 agent/train_RELINE.py

- To train using BEAGT
  ```bash
  python3 agent/train_BEAGT.py

- To train using CAE
  ```bash
  python3 agent/train_CAE.py
