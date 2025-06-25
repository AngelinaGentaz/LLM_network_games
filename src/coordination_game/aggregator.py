import os
import glob
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# 0. PATHS
# ---------------------------------------------------------------------
dir_root       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tests_root_dir = os.path.join(dir_root, "tests")

# ---------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def process_file(path):
    with open(path, "r") as f:
        results = json.load(f)

    scenarios = defaultdict(lambda: defaultdict(dict))  # cfp -> cost -> {player: decision}
    for entry in results:
        cfp      = entry.get("cfp")
        resp     = entry["llm_response"]
        cost     = float(resp["cost"].split("=")[1].strip())
        pid      = int(resp["decision"].split("_")[1].split("=")[0].strip())
        decision = int(resp["decision"].split("=")[1].strip())
        scenarios[cfp][cost][pid] = decision

    out = defaultdict(dict)
    for cfp_key, cost_map in scenarios.items():
        for cost, decisions in cost_map.items():
            profile = tuple(decisions[i] for i in sorted(decisions))
            out[cfp_key][cost] = profile
    return out

# ---------------------------------------------------------------------
# 2. MAIN LOOP PER PROVIDER
# ---------------------------------------------------------------------
provider_dirs = [
    os.path.join(tests_root_dir, d)
    for d in os.listdir(tests_root_dir)
    if os.path.isdir(os.path.join(tests_root_dir, d))
]
if not provider_dirs:
    raise RuntimeError("No provider sub-folders found in /tests.")

for provider_dir in provider_dirs:
    provider = os.path.basename(provider_dir).capitalize()
    files = sorted(
        glob.glob(os.path.join(provider_dir, "results_baseline.json")),
        key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0].split("_")[1])
    )
    if not files:
        continue

    # Aggregate counts
    counts_by_cfp = defaultdict(lambda: defaultdict(Counter))  # cfp -> cost -> Counter(profile)
    for fpath in files:
        processed = process_file(fpath)
        for cfp_key, cost_map in processed.items():
            for cost, profile in cost_map.items():
                counts_by_cfp[cfp_key][cost][profile] += 1

    # Prepare global profile color mapping
    all_profiles = sorted({profile
                           for cost_map in counts_by_cfp.values()
                           for profs in cost_map.values()
                           for profile in profs})
    pastel = plt.get_cmap('Pastel1')
    color_map = {p: pastel(idx) for idx, p in enumerate(all_profiles)}

    # Single combined figure: rows = CFPs, cols = costs
    cfps = sorted(counts_by_cfp)
    cost_cols = sorted({cost for c in cfps for cost in counts_by_cfp[c]})
    n_rows = len(cfps)
    n_cols = len(cost_cols)

    # Adjusted figure size
    width_per_plot = 2 
    height_per_plot = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width_per_plot * n_cols, height_per_plot * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = [axes]

    bar_width = 0.7
    # Plot grid
    for i, cfp_key in enumerate(cfps):
        for j, cost in enumerate(cost_cols):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            count_map = counts_by_cfp[cfp_key].get(cost, Counter())
            profiles = sorted(count_map)
            counts   = [count_map[p] for p in profiles]
            xpos     = np.arange(len(profiles))

            ax.bar(xpos, counts, width=bar_width,
                   color=[color_map[p] for p in profiles], edgecolor='none')
            # Grid
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

            # Titles and labels
            if i == 0:
                ax.set_title(f"c = {cost}", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"CFP={cfp_key}\nCount", fontsize=10)
            ax.set_xticks(xpos)
            ax.set_xticklabels([''.join(map(str,p)) for p in profiles], rotation=45, fontsize=8)
            ax.set_xlabel('Profile', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(provider_dir, f"dist_combined_{provider.lower()}.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined figure: {out_file}")
