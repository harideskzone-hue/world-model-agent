import os
import sys

# Add parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import json
import time
import urllib.request
import urllib.error

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
        req = urllib.request.Request("http://localhost:11434/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print_success(f"Ollama is running (v{data.get('version')})")
    except urllib.error.URLError:
        print_error("Ollama is not running. Please start it with: ollama serve")
    except Exception as e:
        print_error(f"Ollama health check failed: {e}")

def check_model():
    print_step("Checking if target model is available in Ollama...")
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            # Check for gemma2:2b or similar
            if any("gemma2" in m for m in models) or any("qwen" in m for m in models):
                print_success(f"Target model found in Ollama.")
            else:
                print_error("No recommended model (gemma2:2b, qwen2.5) found. Run: ollama pull gemma2:2b")
    except urllib.error.URLError:
        print_error("Failed to connect to Ollama.")
    except Exception as e:
        print_error(f"Failed to list Ollama models: {e}")

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
        from slm_actions.slm_extractor import SLMExtractor
        from world_model.graph_store import InMemoryGraphStore
        from query_engine.query_layer import QueryLayer
        from updater.updater import Updater
        from shared.models import Observation, WorkingMemory
        
        # 1. Initialize World Model
        graph = InMemoryGraphStore()
        print_success("World Model initialized")
        
        # 2. Updater and Memory setup
        updater = Updater(graph)
        print_success("World Model Updater initialized successfully")
        
        # 3. Query Layer
        query_layer = QueryLayer()
        wm = WorkingMemory()
        wm.last_observation = Observation(
            feedback="You open the door.",
            description="You are in the Study. You see a workbench here.",
            inventory="You are carrying nothing.",
            location="study",
            objective="Retrieve the keycard."
        )
        context_slice = query_layer.retrieve(graph, wm)
        if context_slice.current_room is not None:
            print_success("Query Layer successfully retrieved context")
        
        # 4. Save world model
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
