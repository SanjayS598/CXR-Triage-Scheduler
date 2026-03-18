from sim.hospital_env import HospitalEnv
import matplotlib.pyplot as plt

env = HospitalEnv()

state = env.reset()

rewards = []
queue_lengths = []

for t in range(100):

    actions = []
    
    patient_severities = [(idx, p.severity) for idx, p in enumerate(env.queue)]
    
    # sort by severity
    patient_severities.sort(key=lambda x: x[1], reverse=True)
    
    assigned_count = 0

    # assigning doctor to patient
    for i, doctor in enumerate(env.doctors):
        if not doctor.busy and assigned_count < len(patient_severities):
            sickest_patient_idx = patient_severities[assigned_count][0]
            
            actions.append((i, sickest_patient_idx))
            assigned_count += 1

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
plt.savefig("results/severity/severity_results.png", dpi=300)