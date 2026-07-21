# CXR-Triage-Scheduler

A reinforcement learning (RL) system for intelligent radiology worklist scheduling. A computer vision model scores the urgency of incoming chest X-rays, and a PPO-trained RL agent learns a queue management policy inside a simulated radiology department — reducing urgent-case turnaround time while preserving throughput and fairness, compared to FIFO and severity-only baselines.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [1. CV Urgency Scoring](#1-cv-urgency-scoring)
  - [2. Hospital Simulation](#2-hospital-simulation)
  - [3. RL Training](#3-rl-training)
  - [4. Baseline Policies](#4-baseline-policies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup: Healthy Mean Embedding](#setup-healthy-mean-embedding)
- [Configuration](#configuration)
- [Running the Baselines](#running-the-baselines)
- [Training the RL Agent](#training-the-rl-agent)
- [Testing](#testing)
- [Reward Design](#reward-design)
- [Results](#results)

---

## Overview

Radiology departments face a constant challenge: incoming chest X-rays (CXRs) must be read quickly, but not all scans are equally urgent. A first-come-first-served queue can leave critically ill patients waiting behind routine cases. This project addresses that problem by:

1. **Scoring urgency** — using a pretrained DenseNet-121 CXR model to compute an embedding-distance-based "unhealthy score" for each new scan.
2. **Simulating a radiology department** — a discrete-time simulation models the queue, radiologist workload, patient arrivals, and treatment times.
3. **Learning a scheduling policy** — a PPO (Proximal Policy Optimization) agent is trained inside the simulation to learn which patient the radiologist should read next at every timestep.
4. **Comparing against baselines** — FIFO (first-in, first-out), random, and severity-greedy policies are provided for benchmarking.

---

## How It Works

### 1. CV Urgency Scoring

**File:** `cv/extract_embeddings.py`

Each patient is assigned a chest X-ray from either a `normal` or `abnormal` pool. The `extractor` class:

- Loads a **DenseNet-121** model pretrained on all conditions (`densenet121-res224-all`) from [torchxrayvision](https://github.com/mlmed/torchxrayvision).
- Loads a **mean healthy embedding** computed offline from a set of normal CXRs (see [Setup](#setup-healthy-mean-embedding)).
- For each new patient CXR, computes a feature embedding and measures its **Euclidean distance** from the healthy mean — producing a scalar `unhealthy_score`. A score of `0` means the scan looks healthy; a higher score indicates greater deviation from normal.
- This score is multiplied by 10 to produce the patient's `severity` value used throughout the simulation.

**File:** `cv/extract_healthy_embeddings.py`

A one-time offline script that:
- Reads healthy image filenames from a NIH Chest X-ray CSV (`Finding Labels == 'No Finding'`).
- Computes embeddings for all healthy images using the same DenseNet-121 model.
- Saves the **online mean embedding** to `healthy_mean_weight/healthy_mean_embedding.pt`.

---

### 2. Hospital Simulation

**Files:** `sim/`

The simulation is a discrete-time environment modelling a radiology department.

**`Patient`** — Each patient has:
- A unique ID and arrival time.
- A randomly selected CXR (50% normal / 50% abnormal).
- A CV-derived `severity` score (scaled unhealthy distance) and an `initial_severity` snapshot.
- A `wait_time` counter that increments each timestep they remain in the queue.
- A `state` field: `"waiting"` → `"treating"` → `"done"`.

**`Doctor`** — Each radiologist has:
- A `beta` parameter (treatment speed multiplier).
- A `busy` flag and `remaining_time` counter.
- A `current_patient` reference while treating.

**`HospitalEnv`** — The core simulation:
- Manages a queue (capped at `max_queue`), a list of doctors, and a completed-patient log.
- Each `step(actions)` call:
  1. Increments time; probabilistically spawns a new patient (controlled by `arrival_rate`).
  2. Increments `wait_time` for all queued patients.
  3. Applies the provided `(doctor_idx, patient_idx)` assignment actions.
  4. Advances all active treatments by one timestep, completing them when `remaining_time` reaches 0.
  5. Computes and returns the step reward.

**`HospitalRewarder`** — The reward model:
- **Effective priority:** `P_eff = severity_0 + α × severity_0 × wait_time` — priority compounds with wait time for high-severity cases.
- **Treatment reward:** proportional to effective priority × doctor speed (encouraging timely treatment of urgent cases).
- **Queue penalty:** sum of effective priorities of all waiting patients (penalising delayed urgent cases).
- **Per-patient penalties:** linear in current severity and wait time.

---

### 3. RL Training

**Files:** `rl_environment/`, `rl_environment/train.py`

**`HospitalGymEnv`** wraps the `HospitalEnv` as a standard [Gymnasium](https://gymnasium.farama.org/) environment:

- **Observation space:** A flat vector of `top_n_cases × 3 + num_doctors × 2` features. For each of the top-N queue slots: `[current_severity, wait_time, initial_severity]`. For each doctor: `[is_busy, remaining_time]`.
- **Action space:** `Discrete(max_queue + 1)` — the agent picks which queue index to assign to the next free doctor. Action `max_queue` means "do nothing".
- **Episode length:** `shift_duration` timesteps (default 480, representing one work shift).

Training uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO:

```bash
python -m rl_environment.train --config rl_environment/configs/poc.yaml
```

Outputs:
- Periodic checkpoints in `checkpoints/`
- Best model saved to `checkpoints/best/`
- TensorBoard logs in `runs/tensorboard/`
- Monitor logs for training and evaluation in `runs/`

---

### 4. Baseline Policies

**Files:** `policies/`

Three rule-based baselines run against the same `HospitalEnv`:

| Policy | Description |
|--------|-------------|
| `fifo.py` | **First-In, First-Out** — assigns the patient who has been waiting longest. |
| `severity.py` | **Severity-Greedy** — always assigns the patient with the highest current severity score. |
| `random.py` | **Random** — assigns a randomly chosen queued patient. |

Each baseline script runs a 100-step simulation and saves a plot of queue length and cumulative reward to `results/<policy>/`.

---

## Project Structure

```
CXR-Triage-Scheduler/
├── cv/
│   ├── extract_embeddings.py        # DenseNet-121 urgency scorer
│   └── extract_healthy_embeddings.py # Offline healthy mean embedding builder
├── healthy_mean_weight/
│   └── healthy_mean_embedding.pt    # Precomputed healthy baseline embedding
├── policies/
│   ├── fifo.py                      # FIFO baseline
│   ├── severity.py                  # Severity-greedy baseline
│   └── random.py                    # Random baseline
├── rl_environment/
│   ├── configs/
│   │   └── poc.yaml                 # Training/simulation configuration
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── datatypes.py
│   │   └── gym_env.py               # Gymnasium-compatible RL wrapper
│   ├── requirements.txt
│   └── train.py                     # PPO training script
├── sim/
│   ├── __init__.py
│   ├── doctor.py                    # Radiologist model
│   ├── hospital_env.py              # Core simulation engine
│   ├── patient.py                   # Patient model (with CV scoring)
│   └── rewards.py                   # Reward computation
├── results/
│   ├── fifo/                        # FIFO baseline result plots
│   ├── severity/                    # Severity-greedy result plots
│   └── random/                      # Random baseline result plots
├── test/
│   ├── __init__.py
│   └── test_patient.py              # Patient instantiation test
├── main.py
└── README.md
```

---

## Installation

**Requirements:** Python 3.9+, CUDA GPU recommended (the CV model runs on GPU by default).

```bash
# Clone the repository
git clone https://github.com/SanjayS598/CXR-Triage-Scheduler.git
cd CXR-Triage-Scheduler

# Install dependencies
pip install -r rl_environment/requirements.txt
pip install torchxrayvision scikit-image pandas tqdm
```

You will also need a directory of chest X-ray images to serve as the patient pool:

```
x-ray_imgs/
├── normal/      # Normal (healthy) CXR images (.png / .jpg / .jpeg)
└── abnormal/    # Abnormal (pathological) CXR images
```

---

## Setup: Healthy Mean Embedding

Before running any simulation, generate the healthy baseline embedding that the CV model uses to score patients.

1. Download healthy CXR images (e.g. the [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)) and the accompanying `sample_labels.csv`.
2. Arrange them as:
   ```
   healthy_imgs/
   ├── sample_labels.csv
   └── images/
       └── *.png
   ```
3. Run the embedding script:
   ```bash
   python cv/extract_healthy_embeddings.py
   ```
   This saves `healthy_mean_weight/healthy_mean_embedding.pt`.

---

## Configuration

All simulation and training parameters are controlled by a YAML config file (`rl_environment/configs/poc.yaml`):

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `simulation` | `shift_duration` | `480` | Episode length (timesteps per shift) |
| `simulation` | `num_radiologists` | `1` | Number of doctors |
| `simulation` | `random_seed` | `42` | RNG seed |
| `environment` | `max_queue` | `50` | Maximum queue capacity |
| `environment` | `arrival_rate` | `0.3` | Probability of a new patient per timestep |
| `environment` | `initial_numbers` | `5` | Patients pre-loaded at episode start |
| `environment` | `doctor_betas` | `[2.0]` | Treatment speed per doctor |
| `environment` | `reward_alpha` | `2.0` | Priority escalation rate with wait time |
| `environment` | `reward_t_max` | `10.0` | Minimum treatment time |
| `environment` | `reward_mu` | `1.5` | Treatment reward multiplier |
| `environment` | `severity_negative` | `0.1` | Per-step severity penalty weight |
| `environment` | `waiting_negative` | `0.05` | Per-step wait time penalty weight |
| `observation` | `top_n_cases` | `25` | Number of queue slots in observation |
| `training` | `total_timesteps` | `100000` | Total PPO training steps |
| `training` | `learning_rate` | `0.0003` | PPO learning rate |
| `training` | `n_steps` | `2048` | PPO rollout buffer size |
| `training` | `batch_size` | `64` | Mini-batch size |
| `training` | `gamma` | `0.99` | Discount factor |

---

## Running the Baselines

Each baseline script runs the simulator for 100 steps and saves a results plot:

```bash
# FIFO baseline
python policies/fifo.py

# Severity-greedy baseline
python policies/severity.py

# Random baseline
python policies/random.py
```

Plots are saved to `results/fifo/`, `results/severity/`, and `results/random/` respectively.

---

## Training the RL Agent

```bash
python -m rl_environment.train --config rl_environment/configs/poc.yaml
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir runs/tensorboard/
```

The final model is saved to `checkpoints/ppo_hospital_final.zip`. The best model (by eval reward) is saved to `checkpoints/best/`.

---

## Testing

```bash
pytest test/
```

The test suite currently covers patient instantiation and CV score assignment.

---

## Reward Design

The agent maximises a composite reward signal at each timestep:

```
reward = Σ treatment_rewards − Σ queue_penalties − Σ per_patient_penalties
```

- **Treatment reward** (positive): awarded when a doctor is assigned a patient.  
  `reward = μ × t_treatment`, where `t_treatment = max(t_max, P_eff / β)`.  
  High-priority patients treated by fast doctors yield higher rewards.

- **Queue penalty** (negative): at every step, the total effective priority of all waiting patients is subtracted. This directly incentivises clearing high-severity cases promptly.

- **Per-patient penalties** (negative): each waiting patient contributes `−α_sev × severity − α_wait × wait_time` per step.

**Effective priority** compounds severity with wait time:  
`P_eff = severity_0 × (1 + α × wait_time)`  
This ensures that even moderately severe cases become high priority if left waiting too long.

---

## Results

Baseline result plots (queue length and reward over time) are saved under `results/`:

```
results/
├── fifo/fifo_results_new.png
├── severity/severity_results.png
└── random/
```

After training the PPO agent, compare its evaluation reward against the baselines to quantify the scheduling improvement.
