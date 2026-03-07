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
        treatment_time_per_doctor=[4],  # list of treatment times
        positive_reward = 10.0, # reward for each treated patients
        severity_negative = 0.1, # for computing negative reward of a waiting patient based on severity
        waiting_negative = 0.05 # for computing negative reward of a waiting patient based on wait time
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

        # storing initial number of patients
        self.initial_numbers = initial_numbers

        # reward parameters
        self.positive_reward = positive_reward
        self.severity_negative = severity_negative
        self.waiting_negative = waiting_negative

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

    def _treat_step(self, doctor):
        """step function for docotor treating the patient"""
        p = doctor.current_patient

        # reduction is based on the initial servity
        # note, doctor.treatment_time is the number of steps the doctor need to treat the patient
        # also, doctor remaining time is to show how more steps the doctor need with this patient
        # this is done for metrics
        reduction = p.initial_severity / doctor.treatment_time
        p.severity = max(0.0, p.severity - reduction)

        doctor.remaining_time -= 1

    def _finish_treatment(self, doctor):
        """mark patient done and free doctor"""
        p = doctor.current_patient
        p.state = "done"
        self.done_patients.append(p)
        
        # reset doctor's state
        doctor.current_patient = None
        doctor.busy = False
        doctor.remaining_time = 0

    def _reward(self, treated_now):
        """
        treated_now: number of patients treated that step
        """

        # positive reward for treatment
        reward = 0
        reward = self.positive_reward * treated_now
        
        # negative penalty for waiting patients
        for p in self.queue:
            reward -= self.severity_negative * p.severity
            reward -= self.waiting_negative * p.wait_time
        
        return reward

    def get_state(self):
        """get the current environment state"""
        return {
            "queue_severity": [p.severity for p in self.queue],
            "queue_wait": [p.wait_time for p in self.queue],
            "doctor_busy": [int(d.busy) for d in self.doctors],
            "doctor_remaining": [d.remaining_time for d in self.doctors],
            "time": self.curr_time
        }

    def step(self, actions):
        """
        actions: list of (doctor_idx, patient_idx)
        """
        self.curr_time += 1
        treated_now = 0

        # Adding new patient if we can
        if np.random.rand() < self.arrival_rate and len(self.queue) < self.max_queue:
            self.queue.append(self._spawn_patient())

        # increment the wait time for everyone
        for p in self.queue:
            p.wait_time += 1

        # sort actions to avoid queue index issues when popping
        actions = sorted(actions, key=lambda x: x[1], reverse=True)

        # assigning the patients
        for d_idx, p_idx in actions:
            # just a simple check to make sure the doctor id and the patient id is legit
            if d_idx < len(self.doctors) and p_idx < len(self.queue):
                if not self.doctors[d_idx].busy:
                    self._assign(d_idx, p_idx)

        # treatment update
        for d in self.doctors:
            if d.busy:
                self._treat_step(d)
                if d.remaining_time <= 0:
                    self._finish_treatment(d)
                    treated_now += 1

        reward = self._reward(treated_now)
        state = self.get_state()
        done = False  # continuous environment

        return state, reward, done

    def reset(self):
        """reset the sim"""
        self.curr_time = 0
        self.pid_counter = 0
        self.queue = []
        self.done_patients = []

        for d in self.doctors:
            d.busy = False
            d.remaining_time = 0
            d.current_patient = None

        for _ in range(self.initial_numbers):
            self.queue.append(self._spawn_patient())

        return self.get_state()