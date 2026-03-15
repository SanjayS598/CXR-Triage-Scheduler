from sim.hospital_env import HospitalEnv
import matplotlib.pyplot as plt
import random

env = HospitalEnv()

state = env.reset()

rewards = []
queue_lengths = []

for t in range(100):

    actions = []
    available_patient_indices = list(range(len(env.queue)))
    random.shuffle(available_patient_indices)

    for i, doctor in enumerate(env.doctors):
        if not doctor.busy and len(available_patient_indices) > 0:
            # Assign doctor i to the next available patient index
            random_patient_idx = available_patient_indices.pop()
            actions.append((i, random_patient_idx))

    state, reward, done = env.step(actions)

    rewards.append(reward)
    queue_lengths.append(len(env.queue))

print("Simulation finished")

plt.figure()

plt.subplot(2,1,1)
plt.plot(queue_lengths)
plt.title("Queue Length Over Time")
plt.ylabel("Queue Size")

plt.subplot(2,1,2)
plt.plot(rewards)
plt.title("Reward Over Time")
plt.ylabel("Reward")
plt.xlabel("Step")

plt.tight_layout()
plt.savefig("results/random/random_policy_results.png", dpi=300)