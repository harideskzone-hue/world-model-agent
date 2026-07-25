class SLMExtractor:
    """
    Calls the local SLM to extract facts from a raw observation.
    Spec: SLM_INTERFACES.md Section 1
    """
    
    def __init__(self, slm_client):
        """
        Args:
            slm_client: Local SLM client (e.g., SLMRunner)
            Must support: client.generate(prompt=..., temperature=...) -> str
        """
        self.slm = slm_client
    
    def extract(self, observation: str) -> str:
        """
        Build the extractor prompt and call the SLM.
        
        INPUT: Raw observation string
        OUTPUT: Raw SLM text
        """
        prompt = self._build_extractor_prompt(observation)
        raw_response = self.slm.generate(prompt=prompt, temperature=0.0)
        return raw_response
    
    def _build_extractor_prompt(self, observation: str) -> str:
        """Zero-shot extraction prompt without static examples or keywords."""
        return f"""You are a strict data extractor for a TextWorld environment.
Your ONLY task is to extract current factual relationships from the observation text and return them as a valid JSON array using zero-shot semantic comprehension.

SCHEMA:
- subject: entity name in lowercase (objects, rooms, doors, containers, portals, gateways, characters). When text refers to "you", "your inventory", or what "you are carrying", strictly use "player" as the subject with subject_type "CHARACTER".
- subject_type: strictly one of [ROOM, OBJECT, CHARACTER]. Use "OBJECT" for inanimate items, containers, doors, portals, gates, and passageways.
- relation: strictly one of the following:
  * "contains": a room or inanimate furniture piece holds an object, door, portal, or gateway. If a container is described as empty, or if an object was just removed/taken from it, do NOT emit a contains relation!
  * "holds": a character (specifically "player") carries an item in their inventory. Always prefer "holds" over "contains" when the subject is a character.
  * "located_in": an object, door, portal, gateway, or character is situated inside a room or container. Note that rooms are physical locations; a room can NEVER be located_in a character or object.
  * "connects_to": an exit, door, portal, gateway, or room links to a direction (east, west, north, south) or adjoining room.
  * "has_state": an entity possesses a physical state (e.g. open, closed, locked, unlocked, empty). Meticulously extract the open, closed, or locked physical state of every door, gateway, portal, or container described!
- object: target entity name, direction, or state value in lowercase.
- object_type: strictly one of [ROOM, OBJECT, CHARACTER, STATE]
- extraction_type: strictly one of [direct, implied, negation]. Use "negation" if text explicitly states an object is missing or nothing is present.

RULES:
1. Output ONLY a valid JSON array. No explanations, no markdown blocks, no conversational text.
2. Only extract CURRENT, ACTIVE facts. If text indicates an object is in the player's inventory or was just taken, it is no longer inside its previous container or on its previous supporter! Never emit conflicting facts where an object is simultaneously held by the player and contained in an inanimate object.
3. If no facts are present, output an empty array: []
4. Perform direct semantic inference without relying on hardcoded examples or keywords.
5. You MUST extract EVERY door, portal, and exit mentioned, including their physical state (e.g., has_state: open/closed/locked) and where they connect (e.g., connects_to: north/south/east/west). Do NOT skip them!
6. You MUST extract the player's current room location from the text (e.g., {{"subject": "player", "subject_type": "CHARACTER", "relation": "located_in", "object": "<room_name>", "object_type": "ROOM"}}). The room name is often indicated at the very top of the text between dashes (e.g. "-= Garage =-").

TEXT: "{observation}"
FACTS:"""
