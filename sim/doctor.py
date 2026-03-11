class Doctor:
    def __init__(self, did, treatment_time=4):
        self.id = did
        self.busy = False
        self.remaining_time = 0
        self.current_patient = None
        self.treatment_time = treatment_time # takes x number of steps to treat a patient 