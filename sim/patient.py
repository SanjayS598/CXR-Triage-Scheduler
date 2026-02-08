class Patient:
    def __init__(self, pid, severity, arrival_time):
        self.id = pid
        self.severity = severity
        self.arrival_time = arrival_time
        self.wait_time = 0
        self.state = "waiting" # waiting, active