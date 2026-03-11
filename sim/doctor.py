class Doctor:
    def __init__(self, did, beta=4):
        self.id = did
        self.busy = False
        self.total_treatment_time = 0
        self.remaining_time = 0
        self.current_patient = None
        self.beta = beta