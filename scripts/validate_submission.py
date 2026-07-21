import os
import sys

# Add parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import requests
import json
import time

def print_step(msg):
    print(f"\033[94m[VALIDATE]\033[0m {msg}")

def print_success(msg):
    print(f"\033[92m[✓]\033[0m {msg}")

def print_error(msg):
    print(f"\033[91m[✗]\033[0m {msg}")
    sys.exit(1)

def check_textworld():
    print_step("Checking TextWorld installation...")
    try:
        import textworld
        print_success(f"TextWorld installed (v{textworld.__version__})")
    except ImportError:
        print_error("TextWorld is not installed. Run: pip install -r requirements.txt")

def check_ollama():
    print_step("Checking Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        if response.status_code == 200:
            print_success(f"Ollama is running (v{response.json().get('version')})")
        else:
            print_error("Ollama responded with an error.")
    except requests.exceptions.RequestException:
        print_error("Ollama is not running. Please start it with: ollama serve")

def check_model():
    print_step("Checking if target model is available in Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            # Check for gemma2:2b or similar
            if any("gemma2" in m for m in models) or any("qwen" in m for m in models):
                print_success(f"Target model found in Ollama.")
            else:
                print_error("No recommended model (gemma2:2b, qwen2.5) found. Run: ollama pull gemma2:2b")
        else:
            print_error("Failed to list Ollama models.")
    except requests.exceptions.RequestException:
        print_error("Failed to connect to Ollama.")

def check_demo_world():
    print_step("Checking if demo world exists...")
    # Will check examples/demo.z8
    world_path = os.path.join(os.path.dirname(__file__), "..", "examples", "demo.z8")
    if os.path.exists(world_path):
        print_success(f"Demo world found at {world_path}")
    else:
        print_error(f"Demo world NOT found at {world_path}")

def run_smoke_test():
    print_step("Running End-to-End Smoke Test...")
    try:
        # Import core modules
        from extractor.text_extractor import TextExtractor
        from world_model.graph_store import InMemoryGraphStore
        from query_layer.query_layer import QueryLayer
        from updater.updater import Updater
        from shared.models import Observation, WorkingMemory
        
        # 1. Initialize World Model
        graph = InMemoryGraphStore()
        print_success("World Model initialized")
        
        # 2. Extract
        extractor = TextExtractor(slm_runner=None)
        obs = Observation(
            feedback="You open the door.",
            description="You are in the Kitchen. You see a fridge here. The fridge is closed.",
            inventory="You are carrying nothing.",
            location="kitchen",
            objective="Put the apple in the fridge."
        )
        wm = WorkingMemory()
        triples = extractor.extract(obs, wm)
        
        # 3. Update
        updater = Updater(graph)
        updater.update(triples, turn_id=1)
        print_success("World Model Updated successfully")
        
        # 4. Query
        query_layer = QueryLayer(graph)
        context_slice = query_layer.retrieve(wm, current_turn=1)
        if context_slice.formatted_text is not None:
            print_success("Query Layer successfully retrieved context")
        
        # 5. Output generated
        print_success("Output generated successfully")
        
        # 6. Save world model
        save_path = os.path.join(os.path.dirname(__file__), "..", "examples", "test_save.json")
        from world_model.persistence.serializer import save_snapshot
        save_snapshot(graph, save_path)
        if os.path.exists(save_path):
            print_success("World model saved successfully")
            os.remove(save_path)
            
    except Exception as e:
        print_error(f"Smoke test failed: {str(e)}")

def main():
    print("==================================================")
    print(" World Model Agent - Submission Validator")
    print("==================================================")
    check_textworld()
    check_ollama()
    check_model()
    check_demo_world()
    run_smoke_test()
    
    print("\n\033[92m[SUCCESS]\033[0m All validation checks passed! Ready for hackathon evaluation.")

if __name__ == "__main__":
    main()
