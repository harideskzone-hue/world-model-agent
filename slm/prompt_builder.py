# slm/prompt_builder.py
# ============================================================================
# Serializes the semantic World Model into a structured prompt for the SLM.
# Spec Reference: Implementation Plan (SLM-Only Semantic Agent)
# ============================================================================

from typing import List, Optional
from shared.models import ContextSlice, WorkingMemory, Observation, Edge

class PromptBuilder:
    """
    Constructs a purely semantic, structured state representation for the SLM.
    Does NOT send raw game narrative. Translates the graph into stable lists.
    """

    @staticmethod
    def build_prompt(
        context_slice: ContextSlice,
        working_memory: WorkingMemory,
        observation: Observation,
        grammar_description: str,
        use_metadata: bool = True
    ) -> str:
        """
        Build the decision prompt based on structured semantic state using abstract node IDs.
        """
        prompt_parts = [
            "You are an AI agent reasoning over a semantic world model to play a text adventure game.",
            "Analyze the structured Current World State below and select exactly one valid action.",
            "Do NOT output conversational filler, markdown formatting, or explanations.",
            "\n=== CURRENT WORLD STATE ==="
        ]

        import json
        from collections import defaultdict
        
        # 1. Goal
        if working_memory.current_sub_goal:
            prompt_parts.append(f"\n[GOAL]\n{working_memory.current_sub_goal}")
        elif working_memory.objective:
            prompt_parts.append(f"\n[GOAL]\n{working_memory.objective}")

        # Abstract ID Generation
        type_inferences = defaultdict(lambda: "object")
        if working_memory.current_room:
            type_inferences[working_memory.current_room] = "room"
            
        for edge in context_slice.included_facts:
            rel = edge.relation.value
            if rel == "connects_to":
                type_inferences[edge.subject] = "room"
                type_inferences[edge.object] = "room"
            elif rel == "contains":
                type_inferences[edge.subject] = "container"

        node_ids = {}
        node_counts = defaultdict(int)
        
        def get_id(surface_name: str) -> str:
            if surface_name == "player" or surface_name == "unknown":
                return surface_name
            if surface_name not in node_ids:
                ntype = type_inferences[surface_name]
                node_counts[ntype] += 1
                node_ids[surface_name] = f"{ntype}_{node_counts[ntype]}"
            return node_ids[surface_name]

        # 2. Current Location
        loc_surface = working_memory.current_room or "unknown"
        loc_id = get_id(loc_surface)
        prompt_parts.append(f"\n[CURRENT LOCATION]\n{loc_id}")

        # 3. Known Graph (Structured JSON-like format)
        entities = defaultdict(lambda: {"node": "", "type": "", "properties": [], "contains": [], "connections": {}})
        inventory_items = []
        
        for edge in context_slice.included_facts:
            subj_surface = edge.subject
            rel = edge.relation.value
            obj_surface = edge.object
            
            subj_id = get_id(subj_surface)
            obj_id = get_id(obj_surface) if rel not in ["has_state", "is"] else obj_surface
            
            entities[subj_surface]["node"] = subj_id
            entities[subj_surface]["type"] = type_inferences[subj_surface]
            
            if rel == "holds":
                inventory_items.append(get_id(obj_surface))
            elif rel == "connects_to":
                direction = edge.direction or "path"
                entities[subj_surface]["connections"][direction] = obj_id
            elif rel == "has_state" or rel == "is":
                entities[subj_surface]["properties"].append(obj_surface)
            elif rel == "contains":
                entities[subj_surface]["contains"].append(obj_id)
            else:
                entities[subj_surface].setdefault("relations", []).append({rel: obj_id})

        prompt_parts.append("\n[INVENTORY]")
        if inventory_items:
            prompt_parts.append(json.dumps(inventory_items, indent=2))
        else:
            prompt_parts.append("[]")

        prompt_parts.append("\n[ENTITIES & RELATIONSHIPS]")
        if entities:
            entity_list = list(entities.values())
            prompt_parts.append(json.dumps(entity_list, indent=2))
        else:
            prompt_parts.append("[]")

        # Optional Metadata Mapping
        if use_metadata and node_ids:
            metadata_mapping = {v: k for k, v in node_ids.items()}
            prompt_parts.append("\n[METADATA MAPPING]")
            prompt_parts.append(json.dumps(metadata_mapping, indent=2))

        # 4. Memory (Failed actions)
        prompt_parts.append("\n[MEMORY: RECENT FAILED ACTIONS]")
        if working_memory.failed_actions:
            prompt_parts.append(json.dumps(list(working_memory.failed_actions[-5:]), indent=2))
            prompt_parts.append("(Do not repeat these failed actions)")
        else:
            prompt_parts.append("[]")

        # 5. Grammar / Valid Actions
        prompt_parts.append("\n=== ACTION RULES ===")
        prompt_parts.append(grammar_description)
        
        prompt_parts.append("\nAnalyze the state and select the single best action to progress toward the [GOAL].")
        prompt_parts.append("Respond with ONLY the exact text of the action you want to take.")
        prompt_parts.append("ACTION:")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_retry_prompt(invalid_action: str, grammar_description: str) -> str:
        """
        Builds a corrective prompt if the SLM failed to produce a valid command.
        """
        return (
            f"Your previous output '{invalid_action}' was invalid or malformed.\n"
            f"You must output exactly one valid TextWorld command.\n\n"
            f"=== ACTION RULES ===\n{grammar_description}\n\n"
            f"Respond with ONLY the exact text of the action. No explanations.\n"
            f"ACTION:"
        )
