import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
from sim.hospital_env import HospitalEnv


class HospitalGymEnv(gym.Env):

    def __init__(
        self,
        num_doctors=1,
        max_queue=50,
        arrival_rate=0.3,
        initial_numbers=5,
        treatment_time_per_doctor=[4],
        positive_reward=10.0,
        severity_negative=0.1,
        waiting_negative=0.05,
        max_steps=480,    # shift_duration from poc.yaml
        top_n_cases=25,   # observation.top_n_cases from poc.yaml
    ):
        super().__init__()

        self._max_steps = max_steps
        self._top_n = top_n_cases
        self._num_doctors = num_doctors
        self._max_queue = max_queue
        self._steps = 0

        self._env = HospitalEnv(
            num_doctors=num_doctors,
            max_queue=max_queue,
            arrival_rate=arrival_rate,
            initial_numbers=initial_numbers,
            treatment_time_per_doctor=treatment_time_per_doctor,
            positive_reward=positive_reward,
            severity_negative=severity_negative,
            waiting_negative=waiting_negative,
        )

        # Fixed-size flat observation vector
        obs_size = top_n_cases * 3 + num_doctors * 2
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Discrete action: patient index to assign, or max_queue for "do nothing"
        self.action_space = spaces.Discrete(max_queue + 1)

    @classmethod
    def from_yaml(cls, config_path: str) -> "HospitalGymEnv":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        sim = cfg.get("simulation", {})
        env = cfg.get("environment", {})
        obs = cfg.get("observation", {})

        return cls(
            num_doctors=sim.get("num_radiologists", 1),
            max_queue=env.get("max_queue", 50),
            arrival_rate=env.get("arrival_rate", 0.3),
            initial_numbers=env.get("initial_numbers", 5),
            treatment_time_per_doctor=env.get("treatment_time_per_doctor", [4]),
            positive_reward=env.get("positive_reward", 10.0),
            severity_negative=env.get("severity_negative", 0.1),
            waiting_negative=env.get("waiting_negative", 0.05),
            max_steps=sim.get("shift_duration", 480),
            top_n_cases=obs.get("top_n_cases", 25),
        )

    def _build_obs(self, state):
        sev = state["queue_severity"]
        wait = state["queue_wait"]
        init_sev = [p.initial_severity for p in self._env.queue]

        n = self._top_n
        sev      = (sev      + [0.0] * n)[:n]
        wait     = (wait     + [0.0] * n)[:n]
        init_sev = (init_sev + [0.0] * n)[:n]

        # 3 features per patient slot: current severity, wait time, initial severity
        patient_feats = []
        for i in range(n):
            patient_feats.extend([sev[i], wait[i], init_sev[i]])

        doctor_feats = state["doctor_busy"] + state["doctor_remaining"]

        return np.array(patient_feats + doctor_feats, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        state = self._env.reset()
        return self._build_obs(state), {}

    def step(self, action):
        self._steps += 1

        # Build assignment list from the discrete action
        actions = []
        for d_idx, doctor in enumerate(self._env.doctors):
            if not doctor.busy and len(self._env.queue) > 0:
                if int(action) < len(self._env.queue):
                    actions.append((d_idx, int(action)))
                # action == max_queue (or out of range) => do nothing

        state, reward, _ = self._env.step(actions)
        obs = self._build_obs(state)

        terminated = False                        # no natural terminal state
        truncated  = self._steps >= self._max_steps  # episode ends at shift end

        return obs, reward, terminated, truncated, {}
