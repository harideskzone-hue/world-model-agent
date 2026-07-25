import sys
import os
# Ensure the root directory is in sys.path when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint

from orchestrator.orchestrator import Orchestrator, BudgetExceeded
from orchestrator.config import OrchestratorConfig
from orchestrator.metrics import EpisodeMetrics, EpisodeResult
from shared.models import Observation
from shared.config import SLMConfig

console = Console()

class VisualOrchestrator(Orchestrator):
    def _run_turn(self, observation: Observation) -> Observation:
        turn_id = self.metrics.turn_id
        
        # --- UI: Start Turn ---
        console.rule(f"[bold blue]Turn {turn_id}[/bold blue]")
        
        # Display Observation
        obs_parts = [observation.description or ""]
        if getattr(observation, "inventory", "") and getattr(observation, "inventory", "").strip():
            inv_text = observation.inventory.strip()
            if "carrying nothing" in inv_text.lower():
                obs_parts.append("Player Inventory: holding nothing")
            else:
                obs_parts.append(f"Player Inventory: {inv_text}")
                
        fb = getattr(observation, "feedback", "") or ""
        fb = fb.strip()
        if fb and fb not in (observation.description or "") and "$" not in fb and "Welcome to TextWorld" not in fb and "_ _" not in fb:
            obs_parts.append(f"Action feedback: {fb}")
            
        full_text = "\n\n".join([p for p in obs_parts if p.strip()])
        if not full_text:
            full_text = observation.feedback or "No description available."
        
        obs_text = Text(full_text.strip(), style="green")
        console.print(Panel(obs_text, title="👀 Observation", border_style="green"))
        
        # 1. Extract facts from observation
        with console.status("[bold yellow]SLM Extracting Facts...[/bold yellow]"):
            raw_text = self.preprocessor.preprocess(full_text)
            slm_response = self.slm_extractor.extract(raw_text)
            
            extractor_tokens = len(raw_text.split()) * 1.3
            self.metrics.record_extractor_tokens(int(extractor_tokens))
            
            raw_objects = self.json_parser.parse(slm_response)
            validated_facts = self.schema_validator.validate(raw_objects, self.metrics.turn_id)
            final_facts = self.confidence_assigner.assign(validated_facts)
        
        if final_facts:
            facts_str = "\n".join([f"• [cyan]{f.subject}[/cyan] --([bold]{f.relation.value}[/bold])--> [cyan]{f.object}[/cyan]" for f in final_facts])
            console.print(Panel(facts_str, title="🧠 Extracted Facts", border_style="cyan"))
        else:
            console.print("[dim italic]No facts extracted this turn.[/dim italic]")
            
        # 2. Update world model
        self.updater.update(final_facts, self.metrics.turn_id)
        
        # --- UI: Display World Model ---
        nodes = self.graph_store.get_all_nodes()
        edges = self.graph_store.get_all_active_edges()
        
        wm_table = Table(show_header=True, header_style="bold magenta", border_style="magenta")
        wm_table.add_column("Subject")
        wm_table.add_column("Relation")
        wm_table.add_column("Object")
        
        for e in edges:
            wm_table.add_row(e.subject, e.relation.value, e.object)
            
        console.print(Panel(wm_table, title="🌍 World Model (Active Edges)", border_style="magenta"))
        
        # 3. Query current world state
        self.working_memory.last_observation = observation
        context_slice = self.query_layer.retrieve(self.graph_store, self.working_memory)
        
        # 4. Build prompt
        grammar_desc = self.env.get_available_commands()
        prompt = self.prompt_builder.build(context_slice, grammar_desc)
        
        planner_tokens = len(prompt.split()) * 1.3
        self.metrics.record_planner_tokens(int(planner_tokens))
        
        if self.metrics.is_over_budget(self.config.max_tokens_per_episode):
            raise BudgetExceeded()
            
        # 5. Select action
        with console.status("[bold red]SLM Planning Action...[/bold red]"):
            slm_action_response = self.slm_agent.decide(prompt)
            action_string = self.action_parser.parse(slm_action_response)
            
        # 6. Validate action
        admissible = self.env.get_available_commands()
        action_result = self.env_validator.validate(action_string, admissible)
        
        # --- UI: Action ---
        if action_result.valid:
            action_text = Text(f"🚀 {action_result.action}", style="bold red")
        else:
            action_text = Text(f"❌ INVALID ACTION: {action_string}", style="bold red strike")
            
        console.print(Panel(action_text, title="🤖 Agent Action", border_style="red"))
        
        # 7. Execute or skip
        if action_result.valid:
            next_observation = self.env.step(action_result.action)
            self.metrics.record_valid_action()
            self.working_memory.previous_action = action_string
        else:
            next_observation = observation
            self.metrics.record_invalid_action()
            
        print("\n")
        return next_observation

def run_visual_demo(world_file: str):
    from textworld import EnvInfos
    import textworld.gym
    from slm_actions.slm_client import SLMRunner
    
    console.print(f"[bold green]Starting HackTronix Visual Demo[/bold green] on [cyan]{world_file}[/cyan]")
    
    request_infos = EnvInfos(
        feedback=True, description=True, inventory=True, 
        location=True, objective=True, won=True, lost=True, 
        admissible_commands=True,
    )
    env_id = textworld.gym.register_game(world_file, request_infos)
    raw_env = textworld.gym.make(env_id)
    
    from scripts.benchmark_suite import EnvWrapper
    env = EnvWrapper(raw_env)
    
    slm = SLMRunner(SLMConfig())
    config = OrchestratorConfig(max_turns=20)
    
    orchestrator = VisualOrchestrator(env, slm, config)
    
    try:
        result = orchestrator.run_episode()
        
        console.rule("[bold yellow]EPISODE COMPLETE[/bold yellow]")
        if result.win:
            console.print(Panel("🎉 [bold green]VICTORY![/bold green] The agent successfully achieved the objective!", border_style="green"))
        else:
            console.print(Panel("💀 [bold red]DEFEAT[/bold red] The agent failed to achieve the objective.", border_style="red"))
            
        # Validate World Model
        from evaluation.world_model_validator import WorldModelValidator
        validator = WorldModelValidator()
        wm_report = validator.validate(orchestrator.graph_store, orchestrator.metrics.turn_id)
        
        console.print(f"\n[bold]World Model Consistency:[/bold] {'[green]100% VALID[/green]' if wm_report.is_valid else '[red]INVALID[/red]'}")
        if not wm_report.is_valid:
            for v in wm_report.violations:
                console.print(f"  - [red]{v.description}[/red]")
                
        console.print(f"[bold]Total Tokens Used:[/bold] {result.total_tokens_used}")
        console.print(f"[bold]Turns Taken:[/bold] {result.steps_taken}")
        console.print(f"[bold]Invalid Action Rate:[/bold] {result.invalid_action_rate*100:.1f}%")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo Interrupted[/bold red]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visual_demo.py <path_to_z8_file>")
        sys.exit(1)
    
    run_visual_demo(sys.argv[1])
