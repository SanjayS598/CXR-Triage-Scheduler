from sim.hospital_env import HospitalEnv
import matplotlib.pyplot as plt

env = HospitalEnv()

state = env.reset()

rewards = []
queue_lengths = []

for t in range(100):

    actions = []

    for i, doctor in enumerate(env.doctors):
        if not doctor.busy and len(env.queue) > 0:
            actions.append((i, 0))

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
plt.savefig("results/fifo/fifo_results.png", dpi=300)