import sys
from scripts.benchmark_suite import run_benchmark_suite

if __name__ == "__main__":
    report = run_benchmark_suite(
        world_file="examples/stage3.z8",
        num_episodes=10,
        output_csv="results/stage3_results.csv"
    )
    if report is None:
        sys.exit(1)
