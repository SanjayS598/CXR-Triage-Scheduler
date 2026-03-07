from dataclasses import dataclass
from typing import Optional


@dataclass
class Case:
    id: int
    arrival_time: float
    true_urgent: bool
    pred_urgency: float
    uncertainty: float
    read_time: float
    completion_time: Optional[float] = None
    
    def get_wait_time(self, current_time: float) -> float:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return current_time - self.arrival_time
    
    def get_turnaround_time(self) -> float:
        if self.completion_time is None:
            raise ValueError(f"Case {self.id} has not been completed yet")
        return self.completion_time - self.arrival_time
    
    def is_sla_violated(self, sla_threshold: float) -> bool:
        if self.completion_time is None:
            return False
        return self.true_urgent and self.get_turnaround_time() > sla_threshold


@dataclass
class QueueStats:
    current_time: float
    queue_length: int
    urgent_count: int
    non_urgent_count: int
    max_wait_time: float
    avg_wait_time: float


@dataclass
class EpisodeMetrics:
    completed_cases: int
    urgent_tats: list[float]
    non_urgent_tats: list[float]
    sla_violations: int
    total_urgent: int
    max_non_urgent_wait: float
    episode_duration: float
    
    def get_urgent_median_tat(self) -> float:
        if not self.urgent_tats:
            return 0.0
        sorted_tats = sorted(self.urgent_tats)
        n = len(sorted_tats)
        if n % 2 == 0:
            return (sorted_tats[n//2 - 1] + sorted_tats[n//2]) / 2
        return sorted_tats[n//2]
    
    def get_urgent_p90_tat(self) -> float:
        if not self.urgent_tats:
            return 0.0
        sorted_tats = sorted(self.urgent_tats)
        idx = int(0.9 * len(sorted_tats))
        return sorted_tats[min(idx, len(sorted_tats) - 1)]
    
    def get_sla_violation_rate(self) -> float:
        if self.total_urgent == 0:
            return 0.0
        return self.sla_violations / self.total_urgent
    
    def get_throughput(self) -> float:
        if self.episode_duration == 0:
            return 0.0
        return self.completed_cases / (self.episode_duration / 60.0)
