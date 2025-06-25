import os
import glob
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

DIR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS_DIR = os.path.join(DIR_ROOT, "tests")

MODEL_MAP = {
    "anthropic": "Claude 3.7 Sonnet",
    "google":    "Gemini 2.0 Flash",
    "openai":    "GPT-4o",
    "mistral":   "Mistral Medium",
}


def parse_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    scenarios = defaultdict(lambda: defaultdict(dict))  # cost -> pid -> val
    for entry in data:
        if entry.get("cfp") != "min":
            continue
        resp = entry.get("llm_response", {})
        dec = resp.get("decision", "")
        if "=" not in dec:
            continue
        cost = float(resp.get("cost", "c=0").split("=")[1].strip())
        pid_part, val_part = dec.split("=")
        pid = int(pid_part.split("_")[1].strip())
        val = int(val_part.strip())
        scenarios[cost][pid] = val
    out = {}
    for cost, players in scenarios.items():
        if len(players) == 4:
            profile = tuple(players[i] for i in sorted(players))
            out[cost] = profile
    return out


def aggregate(provider_dir, tag_pattern):
    files = sorted(glob.glob(os.path.join(provider_dir, tag_pattern)))
    counts = defaultdict(Counter)  # cost -> Counter(profile)
    for fp in files:
        parsed = parse_file(fp)
        for cost, profile in parsed.items():
            counts[cost][profile] += 1
    return counts


def plot_provider(provider_dir, provider):
    baseline = aggregate(provider_dir, "results_baseline*.json")
    neip100 = aggregate(provider_dir, "results_neip*.json")
    if not baseline and not neip100:
        return
    all_costs = sorted(set(baseline) | set(neip100))
    all_profiles = sorted({p for c in all_costs for p in set(baseline.get(c, {}).keys()) | set(neip100.get(c, {}).keys())})
    if not all_profiles:
        return
    color_map = {p: plt.get_cmap('Pastel1')(idx) for idx, p in enumerate(all_profiles)}

    fig, axes = plt.subplots(2, len(all_costs), figsize=(3 * len(all_costs), 4), sharey='row')
    model_name = MODEL_MAP.get(provider, provider)
    fig.suptitle(model_name, fontsize=14)
    if len(all_costs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    bar_width = 0.7
    for i, cost in enumerate(all_costs):
        for row, (counts, title) in enumerate([(baseline, 'baseline'), (neip100, 'neip100')]):
            ax = axes[row][i]
            count_map = counts.get(cost, Counter())
            profiles = all_profiles
            counts_vals = [count_map.get(p, 0) for p in profiles]
            xpos = np.arange(len(profiles))
            ax.bar(xpos, counts_vals, width=bar_width, color=[color_map[p] for p in profiles])
            ax.set_xticks(xpos)
            ax.set_xticklabels([''.join(map(str, p)) for p in profiles], rotation=45, fontsize=8)
            if i == 0:
                ax.set_ylabel(f"{title}\nCount")
            if row == 0:
                ax.set_title(f"c = {cost}")
            ax.set_xlabel('Profile', fontsize=9)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', color='lightgrey')
            ax.set_ylim(0, 10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(TESTS_DIR, f"compare_neip_min_{provider}.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    provider_dirs = [d for d in glob.glob(os.path.join(TESTS_DIR, '*')) if os.path.isdir(d)]
    if not provider_dirs:
        raise RuntimeError(f"No provider data found in {TESTS_DIR}")
    for pdir in provider_dirs:
        prov = os.path.basename(pdir)
        plot_provider(pdir, prov)


if __name__ == '__main__':
    main()