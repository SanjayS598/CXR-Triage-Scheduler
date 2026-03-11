class HospitalRewarder:
    def __init__(self, alpha=2.0, t_treatment_max=10.0, mu=1.5):
        self.alpha = alpha
        self.t_treatment_max = t_treatment_max
        self.mu = mu

    def get_effective_priority(self, patient):
        """Calculates P_eff,i(t)"""
        p0 = patient.initial_severity
        wait_time = patient.wait_time
        
        p_eff = p0 + (self.alpha * p0 * wait_time)
        return p_eff

    def get_queue_penalty(self, queue):
        """Calculates the total negative reward for all waiting patients"""
        total_penalty = 0
        for patient in queue:
            # We subtract the effective priority for each waiting patient
            total_penalty -= self.get_effective_priority(patient)
        return total_penalty

    def get_treatment_reward_and_time(self, patient, doctor):
        """Calculates positive rewards and total treatment time"""
        patient_priority = self.get_effective_priority(patient)
        
        # Uses doctor.beta (treatment speed) as defined in your HospitalEnv
        t_treatment = max(self.t_treatment_max, patient_priority / doctor.beta)
        
        reward = self.mu * t_treatment
        return reward, t_treatment