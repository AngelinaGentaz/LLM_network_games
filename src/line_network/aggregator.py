import os
import json
from collections import defaultdict

# Get the root directory and tests directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(root_dir, "tests", "results.json")

# Load results
with open(results_path, 'r') as f:
    results = json.load(f)

# Group decisions by cost value
scenarios = defaultdict(dict)
for result in results:
    # Extract cost value from "c = X"
    cost = float(result['cost'].split('=')[1].strip())
    # Extract player ID from "a_X = Y"
    player_id = int(result['decision'].split('_')[1].split('=')[0].strip())
    # Extract decision (0 or 1) from "a_X = Y"
    decision = int(result['decision'].split('=')[1].strip())
    scenarios[cost][player_id] = decision

# Print equilibria for each cost value
print("\nEquilibrium Analysis:")
print("-" * 50)
for cost in sorted(scenarios.keys()):
    decisions = scenarios[cost]
    print(f"\nCost c = {cost}:")
    print("Players' decisions:", end=" ")
    # Print decisions in order of player IDs
    strategy_profile = [decisions[i] for i in sorted(decisions.keys())]
    print(f"{strategy_profile}")