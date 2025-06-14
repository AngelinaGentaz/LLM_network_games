import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.normpath(os.path.join(current_dir, "../../tests"))

decision_counts = defaultdict(lambda: defaultdict(lambda: {"ai=1": 0, "ai=0": 0}))

# Load and count decisions
for filename in os.listdir(tests_dir):
    if filename.endswith(".json"):
        with open(os.path.join(tests_dir, filename), "r") as f:
            data = json.load(f)
        for entry in data:
            cost = entry["cost"]
            player_id, decision_value = entry["decision"].split("=")
            player_id = player_id.strip()
            decision_value = decision_value.strip()
            key = "ai=1" if decision_value == "1" else "ai=0"
            decision_counts[cost][player_id][key] += 1

# Fancy bar charts per cost
# use a built-in style with grid\ nplt.style.use('ggplot')
cmap = plt.get_cmap('Pastel2')
for idx, (cost, players) in enumerate(sorted(decision_counts.items())):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(players.keys())
    ai1 = [players[p]["ai=1"] for p in labels]
    ai0 = [players[p]["ai=0"] for p in labels]
    x = range(len(labels))

    # Stacked bars
    ax.bar(x, ai1, label="Coordinated (ai=1)", color=cmap(0))
    ax.bar(x, ai0, bottom=ai1, label="Not Coordinated (ai=0)", color=cmap(1))

    # Axes and title
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=12)
    ax.set_ylim(0, max((ai1[i] + ai0[i] for i in x)) * 1.1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # only integer ticks
    ax.set_xlabel('Players', fontsize=14)
    ax.set_ylabel('Number of Decisions', fontsize=14)
    ax.set_title(f'Decision Distribution per Player (Cost = {cost})', fontsize=16)

    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Tight layout and save
    fig.tight_layout()
    path = os.path.join(tests_dir, f"decision_distribution_{cost}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved fancy plot for cost={cost} at: {path}")
