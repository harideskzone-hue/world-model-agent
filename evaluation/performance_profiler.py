import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class TurnProfile:
    turn_id: int
    total_ms: float
    components: Dict[str, float]  # {"query_layer": 12.5, "prompt_builder": 8.3, ...}
    tokens_used: int

@dataclass
class EpisodeProfile:
    episode_id: int
    total_turns: int
    total_ms: float
    avg_turn_ms: float
    turn_profiles: List[TurnProfile]
    slowest_component: Tuple[str, float]

class PerformanceProfiler:
    """
    Tracks latencies of key components during episode execution.
    """
    
    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.turn_profiles: List[TurnProfile] = []
        self.current_turn_start = None
        self.current_turn_components: Dict[str, float] = {}
        self.current_turn_id = 0
    
    def start_turn(self, turn_id: int):
        """Mark start of a turn."""
        self.current_turn_start = time.time()
        self.current_turn_components = {}
        self.current_turn_id = turn_id
    
    def end_turn(self, tokens_used: int):
        """Mark end of a turn and store profile."""
        if self.current_turn_start is not None:
            total_ms = (time.time() - self.current_turn_start) * 1000
        else:
            total_ms = 0.0
            
        profile = TurnProfile(
            turn_id=self.current_turn_id,
            total_ms=total_ms,
            components=dict(self.current_turn_components),
            tokens_used=tokens_used,
        )
        self.turn_profiles.append(profile)
    
    def measure_component(self, component_name: str, duration_ms: float):
        """Record a component's latency."""
        self.current_turn_components[component_name] = duration_ms
    
    def compile_report(self) -> EpisodeProfile:
        """Generate final profile report."""
        total_ms = sum(p.total_ms for p in self.turn_profiles)
        avg_ms = total_ms / len(self.turn_profiles) if self.turn_profiles else 0
        
        # Find slowest component across all turns
        slowest_comp = None
        slowest_time = 0.0
        for profile in self.turn_profiles:
            for comp, time_ms in profile.components.items():
                if time_ms > slowest_time:
                    slowest_time = time_ms
                    slowest_comp = (comp, time_ms)
        
        return EpisodeProfile(
            episode_id=self.episode_id,
            total_turns=len(self.turn_profiles),
            total_ms=total_ms,
            avg_turn_ms=avg_ms,
            turn_profiles=self.turn_profiles,
            slowest_component=slowest_comp or ("N/A", 0.0),
        )
