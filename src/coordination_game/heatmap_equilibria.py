import os
import glob
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DIR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS_DIR = os.path.join(DIR_ROOT, "tests")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parse_file(path):
    """Return a mapping cost -> profile tuple for a single result file."""
    with open(path, "r") as f:
        data = json.load(f)
    by_cost = defaultdict(dict)
    for entry in data:
        resp = entry.get("llm_response", {})
        cost = float(resp.get("cost", "c = 0").split("=")[1].strip())
        decision = resp.get("decision", "")
        if "=" not in decision:
            continue
        pid_part, val_part = decision.split("=")
        pid = int(pid_part.split("_")[1].strip())
        val = int(val_part.strip())
        by_cost[cost][pid] = val
    out = {}
    for cost, decisions in by_cost.items():
        profile = tuple(decisions[i] for i in sorted(decisions))
        out[cost] = profile
    return out


def is_equilibrium(profile, cost):
    """Check if profile is a Nash equilibrium for the given cost."""
    all_zero = (0, 0, 0, 0)
    all_one = (1, 1, 1, 1)
    if cost < 1.0:
        return profile == all_one
    elif cost == 1.0:
        return profile in (all_zero, all_one)
    else:
        return profile == all_zero

# ------------------------------------------------------------
# Aggregate probabilities per provider and cost
# ------------------------------------------------------------
provider_dirs = [d for d in glob.glob(os.path.join(TESTS_DIR, "*")) if os.path.isdir(d)]
if not provider_dirs:
    raise RuntimeError("No provider data found in /tests directory")

results = {}
all_costs = set()
for prov_dir in provider_dirs:
    prov = os.path.basename(prov_dir)
    files = sorted(glob.glob(os.path.join(prov_dir, "results_*.json")))
    if not files:
        continue
    total = defaultdict(int)
    eq = defaultdict(int)
    for fp in files:
        parsed = parse_file(fp)
        for cost, profile in parsed.items():
            total[cost] += 1
            if is_equilibrium(profile, cost):
                eq[cost] += 1
    cost_probs = {c: (eq[c] / total[c]) for c in total}
    results[prov] = cost_probs
    all_costs.update(total.keys())

if not results:
    raise RuntimeError("No result files parsed")

providers = sorted(results)
cost_values = sorted(all_costs)

heatmap = np.full((len(providers), len(cost_values)), np.nan)
for i, prov in enumerate(providers):
    for j, c in enumerate(cost_values):
        heatmap[i, j] = results.get(prov, {}).get(c, np.nan)

# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1.5 * len(cost_values), 0.8 * len(providers) + 2))
im = ax.imshow(heatmap, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(cost_values)))
ax.set_xticklabels([str(c) for c in cost_values])
ax.set_yticks(np.arange(len(providers)))
ax.set_yticklabels([p.capitalize() for p in providers])
ax.set_xlabel("Cost")
ax.set_ylabel("Provider")

for i in range(len(providers)):
    for j in range(len(cost_values)):
        val = heatmap[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Equilibrium probability")
fig.tight_layout()

out_path = os.path.join(TESTS_DIR, "coordination_heatmap.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"Saved heatmap: {out_path}")
