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

# Mapping provider folder name
MODEL_MAP = {
    "anthropic": "Claude 3.7 Sonnet",
    "google":    "Gemini 2.0 Flash",
    "openai":    "GPT-4o",
    "mistral":   "mistral-medium-2505",
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def parse_file(path):
    """Parse a results file and return {cfp: {cost: profile}}."""
    with open(path, "r") as f:
        data = json.load(f)

    scenarios = defaultdict(lambda: defaultdict(dict))
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

    out = defaultdict(dict)
    for cfp_key, cost_map in scenarios.items():
        for cost, decisions in cost_map.items():
            profile = tuple(decisions[i] for i in sorted(decisions))
            out[cfp_key][cost] = profile
    return out


def is_equilibrium(profile, cost):
    all_zero = (0, 0, 0, 0)
    all_one = (1, 1, 1, 1)
    if cost > 1.0:
        return profile == all_zero
    else:
        return profile in (all_zero, all_one) 


def main():
    # ------------------------------------------------------------
    # Aggregate probabilities per provider, cost and CFP
    # ------------------------------------------------------------
    provider_dirs = [
        d for d in glob.glob(os.path.join(TESTS_DIR, "*"))
        if os.path.isdir(d)
    ]
    if not provider_dirs:
        raise RuntimeError(f"No provider data found in {TESTS_DIR!r}")

    # structure: results[cfp][provider][cost] -> {'eq': int, 'total': int}
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {'eq': 0, 'total': 0}))
    )
    all_costs = set()
    all_cfps  = set()

    for prov_dir in provider_dirs:
        prov = os.path.basename(prov_dir)
        files = sorted(glob.glob(os.path.join(prov_dir, "results_baseline*.json")))
        for fp in files:
            parsed = parse_file(fp)
            for cfp_key, cost_map in parsed.items():
                all_cfps.add(cfp_key)
                for cost, profile in cost_map.items():
                    all_costs.add(cost)
                    rec = results[cfp_key][prov][cost]
                    rec['total'] += 1
                    if is_equilibrium(profile, cost):
                        rec['eq'] += 1

    if not all_cfps:
        raise RuntimeError("No result files parsed")

    provider_keys  = sorted(os.path.basename(d) for d in provider_dirs)
    cost_values    = sorted(all_costs)
    cfp_keys       = sorted(all_cfps)
    provider_labels = [MODEL_MAP.get(p, p.capitalize()) for p in provider_keys]

    # build heatmaps
    heatmaps = {}
    for cfp in cfp_keys:
        mat = np.full((len(provider_keys), len(cost_values)), np.nan)
        for i, prov in enumerate(provider_keys):
            for j, c in enumerate(cost_values):
                rec = results[cfp].get(prov, {}).get(c)
                if rec and rec['total'] > 0:
                    mat[i, j] = rec['eq'] / rec['total']
        heatmaps[cfp] = mat

    # ------------------------------------------------------------
    # Plot heatmap
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1, len(cfp_keys),
        figsize=(1 * len(cost_values) * len(cfp_keys),
                 len(provider_keys) ),
        sharey=True
    )
    if len(cfp_keys) == 1:
        axes = [axes]

    ims = []
    for idx, cfp in enumerate(cfp_keys):
        ax = axes[idx]
        mat = heatmaps[cfp]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ims.append(im)

        ax.set_title(f"CFP = {cfp}")
        ax.set_xticks(range(len(cost_values)))
        ax.set_xticklabels([str(c) for c in cost_values])
        ax.set_yticks(range(len(provider_keys)))
        ax.set_yticklabels(provider_labels)
        ax.set_xlabel("Cost")
        if idx == 0:
            ax.set_ylabel("Model")

        for i in range(len(provider_keys)):
            for j in range(len(cost_values)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center", color="white" if val>0.6 else "black")


    fig.tight_layout()
    out_path = os.path.join(TESTS_DIR, "coordination_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    main()
