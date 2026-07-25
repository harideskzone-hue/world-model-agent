import json
from query_engine.context_slice import ContextSlice

class PromptBuilder:
    """
    Formats the ContextSlice into the final SLM prompt with strict truncation rules.
    """
    
    TEMPLATE = """---
You are an AI agent reasoning over a semantic world model to play a text adventure game.
Analyze the structured Current World State below and select exactly one valid command from the [GRAMMAR DESCRIPTION] list.
Do NOT output conversational filler, markdown formatting, or explanations.

=== CURRENT WORLD STATE ===

[GOAL]
{objective}

[CURRENT LOCATION]
{current_room}

[REACHABLE ROOMS]
{reachable_rooms}

[INVENTORY]
{inventory}

[OBJECTS IN CURRENT ROOM]
{objects_in_current_room}

[LOCKED DOORS]
{locked_doors}

[RECENT CHANGES]
{recent_changes}

[GRAMMAR DESCRIPTION]
{grammar_description}

Analyze the state and select exactly ONE action from the [GRAMMAR DESCRIPTION] list to progress toward the [GOAL]. Never select an action that failed or was rejected in [RECENT CHANGES].
Respond with ONLY a JSON object with an "action" field containing the exact command string from the list.
---"""

    def _estimate_tokens(self, text: str) -> int:
        """Conservative estimate: split + 30% overhead."""
        words = len(text.split())
        return int(words * 1.3)

    def _format_template(self, context_slice: ContextSlice, grammar_description: str) -> str:
        return self.TEMPLATE.format(
            objective=context_slice.objective,
            current_room=context_slice.current_room,
            reachable_rooms=json.dumps(context_slice.reachable_rooms),
            inventory=json.dumps(context_slice.inventory),
            objects_in_current_room=json.dumps(context_slice.objects_in_current_room),
            locked_doors=json.dumps(context_slice.locked_doors),
            recent_changes=json.dumps(context_slice.recent_changes),
            grammar_description=grammar_description
        )

    def _truncate(self, context_slice: ContextSlice, grammar_description: str) -> str:
        """Truncate elements to fit within 1500 tokens."""
        # We start with the full object, and gradually drop lists
        
        # 1. Drop recent_changes entirely
        context_slice.recent_changes = []
        prompt = self._format_template(context_slice, grammar_description)
        if self._estimate_tokens(prompt) <= 1500:
            return prompt
            
        # 2. Truncate objects_in_current_room one by one (from the end since it's sorted by recency)
        while len(context_slice.objects_in_current_room) > 0:
            context_slice.objects_in_current_room.pop()
            prompt = self._format_template(context_slice, grammar_description)
            if self._estimate_tokens(prompt) <= 1500:
                return prompt
                
        # 3. Truncate locked_doors one by one
        while len(context_slice.locked_doors) > 0:
            context_slice.locked_doors.pop()
            prompt = self._format_template(context_slice, grammar_description)
            if self._estimate_tokens(prompt) <= 1500:
                return prompt
                
        # 4. If we are STILL over budget, we just return whatever we have.
        # Mandatory fields (objective, current_room, reachable_rooms, inventory) must never be dropped.
        return prompt

    def build(self, context_slice: ContextSlice, grammar_description: str) -> str:
        # Create a working copy so we don't mutate the original
        working_copy = ContextSlice(
            objective=context_slice.objective,
            current_room=context_slice.current_room,
            reachable_rooms=list(context_slice.reachable_rooms),
            inventory=list(context_slice.inventory),
            objects_in_current_room=list(context_slice.objects_in_current_room),
            locked_doors=list(context_slice.locked_doors),
            recent_changes=list(context_slice.recent_changes)
        )
        
        prompt = self._format_template(working_copy, grammar_description)
        
        tokens = self._estimate_tokens(prompt)
        if tokens > 1500:
            prompt = self._truncate(working_copy, grammar_description)
            
        return prompt
