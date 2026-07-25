from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class EpisodeMetrics:
    episode_id: int
    turn_id: int = 0
    total_tokens_used: int = 0
    total_valid_actions: int = 0
    total_invalid_actions: int = 0
    consecutive_invalid_actions: int = 0
    
    def record_extractor_tokens(self, token_count: int):
        self.total_tokens_used += token_count
    
    def record_planner_tokens(self, token_count: int):
        self.total_tokens_used += token_count
    
    def record_valid_action(self):
        self.total_valid_actions += 1
        self.consecutive_invalid_actions = 0
    
    def record_invalid_action(self):
        self.total_invalid_actions += 1
        self.consecutive_invalid_actions += 1
    
    def is_over_budget(self, max_tokens: int) -> bool:
        return self.total_tokens_used >= max_tokens
    
    def is_stuck(self, max_invalid_in_row: int) -> bool:
        return self.consecutive_invalid_actions >= max_invalid_in_row

@dataclass
class EpisodeResult:
    episode_id: int
    win: bool
    lost: bool
    steps_taken: int
    total_tokens_used: int
    avg_tokens_per_turn: float
    invalid_action_count: int
    invalid_action_rate: float
    world_model_json: str
    timestamp: str
