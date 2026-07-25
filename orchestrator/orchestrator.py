from datetime import datetime

from orchestrator.config import OrchestratorConfig
from orchestrator.metrics import EpisodeMetrics, EpisodeResult
from shared.models import WorkingMemory, Observation
from world_model.graph_store import InMemoryGraphStore
from updater.updater import Updater
from pipeline.preprocessor import MinimalPreprocessor
from pipeline.json_parser import JSONParser
from pipeline.schema_validator import SchemaValidator
from pipeline.confidence_assigner import ConfidenceAssigner
from query_engine.query_layer import QueryLayer
from query_engine.prompt_builder import PromptBuilder
from slm_actions.slm_extractor import SLMExtractor
from slm_actions.slm_agent import SLMAgent
from slm_actions.action_parser import ActionParser
from slm_actions.environment_validator import EnvironmentValidator

class BudgetExceeded(Exception):
    pass

class Orchestrator:
    """The decision loop orchestrating all agent modules."""
    
    def __init__(self, environment, slm_client, config: OrchestratorConfig):
        self.env = environment
        self.slm = slm_client
        self.config = config
        
        # Initialize world model stack
        self.graph_store = InMemoryGraphStore()
        from shared.models import Node
        from shared.enums import NodeType
        self.graph_store.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
        # Note: Phase 1 world_model might just be GraphStore depending on implementation, 
        # but Updater and QueryLayer usually accept graph_store directly.
        self.updater = Updater(self.graph_store)
        
        # Initialize pipeline
        self.preprocessor = MinimalPreprocessor()
        self.json_parser = JSONParser()
        self.schema_validator = SchemaValidator()
        self.confidence_assigner = ConfidenceAssigner()
        
        # Initialize query & prompt
        self.query_layer = QueryLayer()
        self.prompt_builder = PromptBuilder()
        
        # Initialize SLM wrappers
        self.slm_extractor = SLMExtractor(self.slm)
        self.slm_agent = SLMAgent(self.slm)
        self.action_parser = ActionParser()
        self.env_validator = EnvironmentValidator()
        
        # Will be initialized per episode
        self.working_memory = None
        self.metrics = None
        self.episode_id = 0
    
    def run_episode(self) -> EpisodeResult:
        """Run a single episode from reset to terminal."""
        self.episode_id += 1
        
        # Reset environment
        initial_obs = self.env.reset()
        
        # Initialize episode state
        self.working_memory = WorkingMemory(objective=getattr(initial_obs, 'objective', ""))
        self.metrics = EpisodeMetrics(episode_id=self.episode_id)
        
        # Re-initialize graph store per episode
        self.graph_store = InMemoryGraphStore()
        from shared.models import Node
        from shared.enums import NodeType
        self.graph_store.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
        self.updater = Updater(self.graph_store)
        
        # Turn loop
        observation = initial_obs
        while not self._is_terminal(observation):
            # Check hard limits
            if self.metrics.turn_id >= self.config.max_turns:
                break
            if self.metrics.is_over_budget(self.config.max_tokens_per_episode):
                break
            if self.metrics.is_stuck(self.config.max_invalid_actions_in_row):
                break
            
            # Execute turn
            try:
                observation = self._run_turn(observation)
                self.metrics.turn_id += 1
            except BudgetExceeded:
                break
            except Exception as e:
                # Log error, terminate episode
                print(f"Episode terminated due to error: {e}")
                break
        
        # Export results
        result = self._compile_result(observation)
        return result
    
    def _run_turn(self, observation: Observation) -> Observation:
        """Execute a single turn: extract → update → query → prompt → act."""
        
        # 1. Extract facts from observation
        raw_text = self.preprocessor.preprocess(observation.description)
        
        slm_response = self.slm_extractor.extract(raw_text)
        extractor_tokens = len(raw_text.split()) * 1.3
        self.metrics.record_extractor_tokens(int(extractor_tokens))
        
        raw_objects = self.json_parser.parse(slm_response)
        validated_facts = self.schema_validator.validate(raw_objects, self.metrics.turn_id)
        final_facts = self.confidence_assigner.assign(validated_facts)
        
        # 2. Update world model
        self.updater.update(final_facts, self.metrics.turn_id)
        
        # 3. Query current world state
        self.working_memory.last_observation = observation
        prev_act = getattr(self.working_memory, "previous_action", "")
        if prev_act:
            if "(invalid grammar)" in prev_act:
                fb_clean = "Action rejected by environment: not admissible or invalid command in this room."
            else:
                fb_text = getattr(observation, "feedback", "") or getattr(observation, "description", "")
                fb_clean = (fb_text.strip() or "No immediate effect.").split("\n")[0]
            self.working_memory.recent_observations.append(f"Tried action '{prev_act}' -> Feedback: {fb_clean}")
            if len(self.working_memory.recent_observations) > 5:
                self.working_memory.recent_observations.pop(0)
        # QueryLayer.retrieve expects GraphStoreBase and WorkingMemory
        context_slice = self.query_layer.retrieve(self.graph_store, self.working_memory)
        
        # 4. Build prompt
        grammar_desc = self.env.get_available_commands()
        prompt = self.prompt_builder.build(context_slice, grammar_desc)
        
        planner_tokens = len(prompt.split()) * 1.3
        self.metrics.record_planner_tokens(int(planner_tokens))
        
        # Check budget BEFORE calling SLM
        if self.metrics.is_over_budget(self.config.max_tokens_per_episode):
            raise BudgetExceeded()
        
        # 5. Select action
        slm_action_response = self.slm_agent.decide(prompt)
        action_string = self.action_parser.parse(slm_action_response)
        
        # 6. Validate action
        admissible = self.env.get_available_commands()
        action_result = self.env_validator.validate(action_string, admissible)
        
        # 7. Execute or skip
        if action_result.valid:
            next_observation = self.env.step(action_result.action)
            self.metrics.record_valid_action()
            self.working_memory.previous_action = action_string
        else:
            next_observation = observation  # Same observation
            self.metrics.record_invalid_action()
            self.working_memory.previous_action = f"{action_string} (invalid grammar)"
        
        return next_observation
    
    def _is_terminal(self, observation: Observation) -> bool:
        """Check if episode should terminate."""
        return observation.won or observation.lost
    
    def _compile_result(self, final_obs: Observation) -> EpisodeResult:
        """Compile final episode result."""
        turns = max(self.metrics.turn_id, 1)
        avg_tokens = self.metrics.total_tokens_used / turns
        invalid_rate = self.metrics.total_invalid_actions / turns
        
        world_json = self.graph_store.serialize() if self.config.export_world_model else ""
        
        return EpisodeResult(
            episode_id=self.metrics.episode_id,
            win=final_obs.won,
            lost=final_obs.lost,
            steps_taken=self.metrics.turn_id,
            total_tokens_used=self.metrics.total_tokens_used,
            avg_tokens_per_turn=avg_tokens,
            invalid_action_count=self.metrics.total_invalid_actions,
            invalid_action_rate=invalid_rate,
            world_model_json=world_json,
            timestamp=datetime.now().isoformat(),
        )
