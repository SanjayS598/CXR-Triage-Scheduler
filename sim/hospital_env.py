from sim.doctor import Doctor
from sim.patient import Patient
from sim.rewards import HospitalRewarder
import numpy as np

class HospitalEnv:
    def __init__(
        self,
        num_doctors=1,
        max_queue=50,
        arrival_rate=0.3,      # 30% chance new patient per timestep
        initial_numbers=5,     # initial patients in queue
        doctor_betas = [2.0],  # list of treatment times

        reward_alpha=2.0,         # Scales how fast priority increases with wait time
        reward_t_max=10.0,        # Minimum/Maximum boundary for treatment time
        reward_mu=1.5,             # Multiplier for the positive treatment rewar

        severity_negative = 0.1, 
        waiting_negative = 0.05
    ):

        # current time
        self.curr_time = 0

        # patient id generator
        self.pid_counter = 0

        # number of doctors
        self.num_doctors = num_doctors

        # the max amount of patients we can have
        self.max_queue = max_queue

        # the probability a new patient is spawned per time step
        self.arrival_rate = arrival_rate

        # storing initial number of patients
        self.initial_numbers = initial_numbers

        # Initialize the new rewarder math
        self.rewarder = HospitalRewarder(
            alpha=reward_alpha, 
            t_treatment_max=reward_t_max, 
            mu=reward_mu
        )

        # init the doctors
        self.doctors = []
        for i in range(num_doctors):
            # Cycles through the list if you have more doctors than beta values
            b = doctor_betas[i % len(doctor_betas)]
            self.doctors.append(Doctor(i, beta=b))

        # system buffer (queue)
        self.queue = []

        # completed patients (metrics + reward tracking)
        self.done_patients = []

        # init the initial patients already in line
        for _ in range(initial_numbers):
            self.queue.append(self._spawn_patient())

        self.severity_negative = severity_negative
        self.waiting_negative = waiting_negative
    
    def _spawn_patient(self):
        """spawn a new patient"""
        new_patient = Patient(
            pid=self.pid_counter,
            arrival_time=self.curr_time
        )
        self.pid_counter += 1
        return new_patient

    def _assign(self, doctor_idx, patient_idx):
        """assigning a patient to a doctor"""
        doctor = self.doctors[doctor_idx]
        patient = self.queue.pop(patient_idx) 

        # calculate rewards and time
        pos_reward, t_treat = self.rewarder.get_treatment_reward_and_time(patient, doctor)

        # reassigning attr
        doctor.busy = True
        doctor.current_patient = patient
        doctor.remaining_time = t_treat
        doctor.total_treatment_time = t_treat
        
        patient.state = "treating"

        return pos_reward

    def _treat_step(self, doctor):
        """step function for docotor treating the patient"""
        p = doctor.current_patient

        # reduction is based on the initial servity
        reduction = p.initial_severity / doctor.total_treatment_time
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
        doctor.total_treatment_time = 0

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
        step_reward = 0 

        if np.random.rand() < self.arrival_rate and len(self.queue) < self.max_queue:
            self.queue.append(self._spawn_patient())

        for p in self.queue:
            p.wait_time += 1

        actions = sorted(actions, key=lambda x: x[1], reverse=True)

        for d_idx, p_idx in actions:
            if d_idx < len(self.doctors) and p_idx < len(self.queue):
                if not self.doctors[d_idx].busy:
                    step_reward += self._assign(d_idx, p_idx)

        for d in self.doctors:
            if d.busy:
                self._treat_step(d)
                if d.remaining_time <= 0:
                    self._finish_treatment(d)

        step_reward += self.rewarder.get_queue_penalty(self.queue)

        for p in self.queue:
            step_reward -= (self.severity_negative * p.severity)
            step_reward -= (self.waiting_negative * p.wait_time)

        state = self.get_state()
        done = False  

        return state, step_reward, done

    def reset(self):
        self.curr_time = 0
        self.pid_counter = 0
        self.queue = []
        self.done_patients = []

        for d in self.doctors:
            d.busy = False
            d.remaining_time = 0
            d.total_treatment_time = 0
            d.current_patient = None

        for _ in range(self.initial_numbers):
            self.queue.append(self._spawn_patient())

        return self.get_state()