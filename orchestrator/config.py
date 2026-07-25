from dataclasses import dataclass

@dataclass
class OrchestratorConfig:
    max_turns: int = 100
    max_tokens_per_episode: int = 5000
    max_invalid_actions_in_row: int = 5
    export_world_model: bool = True
