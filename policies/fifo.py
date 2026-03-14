from sim.hospital_env import HospitalEnv
import matplotlib.pyplot as plt

env = HospitalEnv()

state = env.reset()

rewards = []
queue_lengths = []

for t in range(100):

    actions = []
    assigned_patients = 0 
    for i, doctor in enumerate(env.doctors):
        # If doctor is free AND we haven't run out of waiting patients
        if not doctor.busy and assigned_patients < len(env.queue):
            # Assign doctor i to the next available patient index
            actions.append((i, assigned_patients))
            assigned_patients += 1

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
plt.savefig("results/fifo/fifo_results_new.png", dpi=300)