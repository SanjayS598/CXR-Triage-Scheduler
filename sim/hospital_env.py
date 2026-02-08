from sim.doctor import Doctor
from sim.patient import Patient
import numpy as np

class HospitalEnv:
    def __init__(
        self,
        num_doctors=1,
        max_queue=50,
        max_severity=30.0,      # max possible severity
        arrival_rate=0.3,      # 30% chance new patient per timestep
        initial_numbers=5,     # initial patients in queue
        treatment_time_per_doctor=[4]  # list of treatment times
    ):

        # current time
        self.curr_time = 0

        # patient id generator
        self.pid_counter = 0

        # number of doctors
        self.num_doctors = num_doctors

        # the max amount of patients we can have
        self.max_queue = max_queue

        # the highest severity the patient could come in with
        self.max_severity = max_severity

        # the probability a new patient is spawned per time step
        self.arrival_rate = arrival_rate

        # init the doctors
        self.doctors = []
        for i in range(num_doctors):
            t = treatment_time_per_doctor[i % len(treatment_time_per_doctor)]
            self.doctors.append(Doctor(i, treatment_time=t))

        # system buffer (queue)
        self.queue = []

        # completed patients (metrics + reward tracking)
        self.done_patients = []

        # init the initial patients already in line
        for _ in range(initial_numbers):
            self.queue.append(self._spawn_patient())
    
    def _spawn_patient(self):
        """spawn a new patient"""
        new_patient = Patient(
            pid=self.pid_counter,
            severity=(np.random.rand() * self.max_severity),
            arrival_time=self.curr_time
        )
        self.pid_counter += 1
        return new_patient

    def _assign(self, doctor_idx, patient_idx):
        """assigning a patient to a doctor"""
        doctor = self.doctors[doctor_idx]
        patient = self.queue[patient_idx]

        # reassigning attr
        doctor.busy = True
        doctor.current_patient = patient
        doctor.remaining_time = doctor.treatment_time
        patient.state = "treating"

        # pop them off the treatment queue
        self.queue.pop(patient_idx)

    

    