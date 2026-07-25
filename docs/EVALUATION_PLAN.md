# Evaluation Plan

To verify that the rebuilt agent meets the HackTronix Track 1 requirements, three staged worlds of increasing complexity are used. Each world is supplied as a compiled `.z8` TextWorld file placed in the `examples/` directory.

## Worlds

| Stage | File (`examples/`) | Room Names (representative) | Goal Description |
|-------|--------------------|----------------------------|------------------|
| 1 – Normal | `stage1.z8` | Kitchen, Hall | 1️⃣ Take the rusty key lying on the table in the Kitchen. 2️⃣ Open the north door (initially closed) using the key. |
| 2 – Medium | `stage2.z8` | Kitchen, Hall, Bedroom, Storage, Garden, Library | 1️⃣ Find the map (located in the Library). 2️⃣ Find the key (located in the Bedroom). 3️⃣ Unlock the storage‑room door (requires the key). 4️⃣ Proceed to the Garden (objective reached). |
| 3 – Advanced | `stage3.z8` | Hall, Kitchen, Bedroom, Library, Garden, Tower, Basement, Garage, Laboratory, Vault | 1️⃣ Obtain the laboratory passcode (found on a note in the Laboratory). 2️⃣ Use the passcode to unlock the Vault door. 3️⃣ Enter the Vault and retrieve the treasure (final objective). |

*All three maps are deterministic, contain only the relations and entity types defined in the ontology, and have been vetted to avoid reliance on undocumented game mechanics.*

## Success Metrics

For each stage, run **N = 10** episodes (different random seeds if the world generator supports stochastic elements; otherwise use the same map). Compute the following aggregates:

| Metric | Definition | Target (per stage) |
|--------|------------|--------------------|
| JSON extraction success rate | % of turns where the SLM returns syntactically valid JSON array (parser returns non‑empty list) | ≥ 98 % (all stages) |
| Invalid action rate | % of turns where `Environment Validator` returns `valid = false` | < 2 % (all stages) |
| Objective completion (win rate) | % of episodes where the agent reaches the goal (`won == true`) before hitting the turn limit | Stage 1: 100 %<br>Stage 2: ≥ 90 %<br>Stage 3: ≥ 80 % |
| Average steps to completion | Mean number of `env.step()` calls taken in winning episodes | Stage 2: < 40<br>Stage 3: < 80 (informational) |
| Average context size (tokens) | Mean number of tokens in the final prompt sent to the SLM (estimated via `len(prompt.split()) * 1.3`) | Stage 3: < 1500 |
| Average query‑layer latency per turn | Mean wall‑clock time spent in `QueryLayer.retrieve` (ms) | Stage 3: < 300 ms |
| World‑model consistency (post‑episode) | After each episode, the internal graph is checked for: <br>• No two **active** edges share identical `(subject, relation, target)`.<br>• `t_valid_from` ≤ `t_observed` for every edge.<br>• `t_valid_until` is either `null` or ≥ `t_valid_from` and ≤ current turn.<br>• Exactly one `node_type = PLAYER` exists.<br>All checks must pass. | 100 % (all stages) |
| Post‑episode world‑model export | After each episode, export the graph to JSON (`examples/stageX_episodeY_world.json`) and/or GraphML for manual inspection; verify that:<br>• Player location matches the last observation.<br>• Door states are consistent with the feedback (e.g., a door reported open has an active `HAS_STATE` edge to state `"open"`).<br>• No duplicate active facts exist. | Visual inspection; used for debugging failures. |

## Automation

- A benchmark script `scripts/benchmark_stageX.py` (X = 1,2,3) will:
  1. Launch the agent with the appropriate `.z8`.
  2. Run the prescribed number of episodes, resetting the environment each time.
  3. Collect the metrics above via structured logging.
  4. Print a summary table and write a CSV file for further analysis.
- The scripts import the agent as a module, ensuring that no code changes are needed to run the benchmarks.

## Acceptance Criteria

The architecture is considered **ready for competition** when **all** targets in the table above are met for **each** of the three stages. If any target is missed, the responsible module(s) should be inspected and the relevant design document revised (e.g., improve the Query Layer if context size exceeds the budget, refine the Extractor prompt if JSON extraction falls short, adjust the Updater if consistency checks fail).

## Additional Notes

- The agent is allowed to use stochastic exploration (e.g., epsilon‑greedy) if desired; however, the win‑rate targets assume a reasonably exploratory policy. Purely deterministic agents may still meet the criteria if the worlds are simple enough.
- All timing measurements should be performed on the target competition hardware (or a close proxy) to ensure realistic latency bounds.

---
