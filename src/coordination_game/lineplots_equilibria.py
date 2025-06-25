import os
import glob
import json
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

DIR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS_DIR = os.path.join(DIR_ROOT, "tests")

MODEL_MAP = {
    "anthropic": "Claude 3.7 Sonnet",
    "google":    "Gemini 2.0 Flash",
    "openai":    "GPT-4o",
    "mistral":   "Mistral Medium",
}


def parse_file(path):
    """Parse a results file and return {cfp: {cost: profile}}."""
    with open(path, "r") as f:
        data = json.load(f)

    scenarios = defaultdict(lambda: defaultdict(dict))
    for entry in data:
        cfp = entry.get("cfp")
        resp = entry.get("llm_response", {})
        dec  = resp.get("decision", "")
        if "=" not in dec:
            continue
        cost = float(resp.get("cost", "c=0").split("=")[1].strip())
        pid_part, val_part = dec.split("=")
        pid = int(pid_part.split("_")[1].strip())
        val = int(val_part.strip())
        scenarios[cfp][cost][pid] = val

    out = defaultdict(dict)
    for cfp_key, cost_map in scenarios.items():
        for cost, decisions in cost_map.items():
            if len(decisions) == 4:
                profile = tuple(decisions[i] for i in sorted(decisions))
                out[cfp_key][cost] = profile
    return out


def is_equilibrium(profile, cost):
    all_zero = (0, 0, 0, 0)
    all_one = (1, 1, 1, 1)
    if cost > 1.0 or cost < 1.0: # STABLE EQUILIBRIUM
        return profile == all_zero
    else:
        return profile in (all_zero, all_one) 


def hamming_distance(profile, cost):
    targets = [(1, 1, 1, 1)] if cost < 1.0 else [(0, 0, 0, 0)]
    if cost == 1.0:
        targets.append((1, 1, 1, 1))
    dists = [sum(a != b for a, b in zip(profile, t)) for t in targets]
    return min(dists)


def aggregate():
    provider_dirs = [d for d in glob.glob(os.path.join(TESTS_DIR, "*")) if os.path.isdir(d)]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"eq": 0, "total": 0, "dist": 0.0})))
    all_costs = set()
    all_cfps = set()
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
                    rec["total"] += 1
                    if is_equilibrium(profile, cost):
                        rec["eq"] += 1
                    rec["dist"] += hamming_distance(profile, cost)
    return results, sorted(all_costs), sorted(all_cfps), [os.path.basename(d) for d in provider_dirs]


def plot_equilibrium_prob(results, costs, providers, cfps):
    fig, axes = plt.subplots(1, len(cfps), figsize=(4 * len(cfps), 3), sharey=True)
    if len(cfps) == 1:
        axes = [axes]
    for idx, cfp in enumerate(cfps):
        ax = axes[idx]
        for prov in providers:
            probs = []
            for c in costs:
                rec = results[cfp].get(prov, {}).get(c)
                if rec and rec["total"] > 0:
                    probs.append(rec["eq"] / rec["total"])
                else:
                    probs.append(np.nan)
            ax.scatter(costs, probs, label=MODEL_MAP.get(prov, prov))

        ax.set_title(f"CFP = {cfp}")
        ax.set_xlabel("Cost")
        ax.set_xticks(costs)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", color="lightgrey")
        if idx == 0:
            ax.set_ylabel("Equilibrium probability")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(TESTS_DIR, "equilibrium_prob_lineplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved line plot: {out_path}")


def plot_equilibrium_prob_per_cfp(results, costs, providers, cfps):
    """Plot equilibrium probability for each provider with CFP as points."""
    fig, axes = plt.subplots(1, len(providers), figsize=(4 * len(providers), 3), sharey=True)
    if len(providers) == 1:
        axes = [axes]

    for idx, prov in enumerate(providers):
        ax = axes[idx]
        for cfp in cfps:
            probs = []
            for c in costs:
                rec = results[cfp].get(prov, {}).get(c)
                if rec and rec["total"] > 0:
                    probs.append(rec["eq"] / rec["total"])
                else:
                    probs.append(np.nan)
            ax.scatter(costs, probs, label=f"CFP {cfp}")

        ax.set_title(MODEL_MAP.get(prov, prov))
        ax.set_xlabel("Cost")
        ax.set_xticks(costs)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", color="lightgrey")
        if idx == 0:
            ax.set_ylabel("Equilibrium probability")

    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(TESTS_DIR, "equilibrium_prob_per_cfp.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved line plot: {out_path}")


def plot_grouped_bar(results, costs, providers, cfps):
    """Grouped bar chart of equilibrium probability per provider."""
    x     = np.arange(len(costs))
    width = 0.8 / len(cfps)
    palette = plt.get_cmap("Set2")

    # --- lay out 2 rows, ceil(#providers/2) cols ---
    n_prov = len(providers)
    ncols  = math.ceil(n_prov / 2)
    fig, axes = plt.subplots(2, ncols,
                             figsize=(4 * ncols, 6),  # taller figure
                             sharey=True)
    axes = axes.flatten()  # make it easy to index

    for p_idx, prov in enumerate(providers):
        ax = axes[p_idx]
        for c_idx, cfp in enumerate(cfps):
            vals = [
                (results[cfp].get(prov, {})
                         .get(c, {"eq":0,"total":1})["eq"] /
                 results[cfp].get(prov, {})
                         .get(c, {"eq":0,"total":1})["total"])
                for c in costs
            ]
            pos = x + (c_idx - (len(cfps)-1)/2)*width
            ax.bar(pos, vals, width=width,
                   color=palette(c_idx),
                   label=f"{cfp}")

        ax.set_title(MODEL_MAP.get(prov, prov))
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in costs])
        ax.set_xlabel("Cost")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", axis="y")
        if p_idx % ncols == 0:
            ax.set_ylabel("Stable Nash equilibrium probability")

    # hide any empty subplots
    for idx in range(n_prov, len(axes)):
        axes[idx].axis("off")

    # gather handles from one of the axes
    handles, labels = axes[0].get_legend_handles_labels()
    # place shared legend on the right
    fig.legend(handles, labels,
               loc='center right',
               title='CFP',
               bbox_to_anchor=(1.03, 0.51))
    # make room for the legend
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    out_path = os.path.join(TESTS_DIR, "stable_NE_prob_grouped_bar.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grouped bar chart: {out_path}")


def plot_hamming_distance(results, costs, providers, cfps):
    fig, axes = plt.subplots(1, len(cfps), figsize=(4 * len(cfps), 3), sharey=True)
    if len(cfps) == 1:
        axes = [axes]
    for idx, cfp in enumerate(cfps):
        ax = axes[idx]
        for prov in providers:
            dists = []
            for c in costs:
                rec = results[cfp].get(prov, {}).get(c)
                if rec and rec["total"] > 0:
                    dists.append(rec["dist"] / rec["total"])
                else:
                    dists.append(np.nan)
            ax.scatter(costs, dists, label=MODEL_MAP.get(prov, prov))
        ax.set_title(f"CFP = {cfp}")
        ax.set_xlabel("Cost")
        ax.set_xticks(costs)
        ax.set_ylim(0, 4)
        ax.grid(True, linestyle="--", color="lightgrey")
        if idx == 0:
            ax.set_ylabel("Avg. Hamming distance")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(TESTS_DIR, "hamming_distance_lineplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved line plot: {out_path}")


def main():
    results, costs, cfps, providers = aggregate()
    plot_equilibrium_prob(results, costs, providers, cfps)
    plot_hamming_distance(results, costs, providers, cfps)
    plot_equilibrium_prob_per_cfp(results, costs, providers, cfps)
    plot_grouped_bar(results, costs, providers, cfps)


if __name__ == "__main__":
    main()