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

# Mapping provider folder name -> human readable model string
MODEL_MAP = {
    "anthropic": "Claude 3.7 Sonnet",
    "google": "Gemini 2.0 Flash",
    "openai": "GPT-4o",
    "mistral": "Mistral Small Latest",
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parse_file(path):
    """Yield ``(cost, profile)`` tuples for each CFP in ``path``."""
    with open(path, "r") as f:
        data = json.load(f)

    scenarios = defaultdict(lambda: defaultdict(dict))  # cfp -> cost -> {pid: val}
    for entry in data:
        cfp = entry.get("cfp")
        resp = entry.get("llm_response", {})
        decision = resp.get("decision", "")
        if "=" not in decision:
            continue
        cost = float(resp.get("cost", "c = 0").split("=")[1].strip())
        pid_part, val_part = decision.split("=")
        pid = int(pid_part.split("_")[1].strip())
        val = int(val_part.strip())
        scenarios[cfp][cost][pid] = val

    outputs = []
    for cost_map in scenarios.values():
        for cost, decisions in cost_map.items():
            profile = tuple(decisions[i] for i in sorted(decisions))
            outputs.append((cost, profile))
    return outputs


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
        for cost, profile in parsed:
            total[cost] += 1
            if is_equilibrium(profile, cost):
                eq[cost] += 1
    cost_probs = {c: (eq[c] / total[c]) for c in total}
    results[prov] = cost_probs
    all_costs.update(total.keys())

if not results:
    raise RuntimeError("No result files parsed")

provider_keys = sorted(results)
cost_values = sorted(all_costs)

provider_labels = [MODEL_MAP.get(p, p.capitalize()) for p in provider_keys]

heatmap = np.full((len(provider_keys), len(cost_values)), np.nan)
for i, prov in enumerate(provider_keys):
    for j, c in enumerate(cost_values):
        heatmap[i, j] = results.get(prov, {}).get(c, np.nan)

# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1.5 * len(cost_values), 0.8 * len(provider_keys) + 2))
im = ax.imshow(heatmap, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(cost_values)))
ax.set_xticklabels([str(c) for c in cost_values])
ax.set_yticks(np.arange(len(provider_keys)))
ax.set_yticklabels(provider_labels)
ax.set_xlabel("Cost")
ax.set_ylabel("Provider")

for i in range(len(provider_keys)):
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
