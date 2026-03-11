from sim.patient import Patient

def main():

    patients = []

    for i in range(10):
        p = Patient(pid=i, arrival_time=i)
        patients.append(p)

        print("------")
        print(f"Patient ID: {p.id}")
        print(f"Arrival time: {p.arrival_time}")
        print(f"X-ray path: {p.xray_path}")
        print(f"Severity score: {p.severity}")

if __name__ == "__main__":
    main()