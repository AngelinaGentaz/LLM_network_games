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
# 1. THEORETICAL EQUILIBRIA  (extend as needed)
# ---------------------------------------------------------------------
theoretical_equilibria = {
    0.5: [[1, 1, 1, 1], [0, 0, 0, 0]],  # Multiple stable equilibria
    1.0: [[0, 0, 0, 0]],                # Only no‑coordination is stable
    2.0: [[0, 0, 0, 0]],                # No‑coordination strictly dominant
}

# ---------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def compare_to_theoretical(cost, observed_profile):
    """Return (min_hamming, closest_theoretical) or (None, None)."""
    if cost not in theoretical_equilibria:
        return None, None
    best_div, best_eq = float("inf"), None
    for eq in theoretical_equilibria[cost]:
        div = sum(o != t for o, t in zip(observed_profile, eq))
        if div < best_div:
            best_div, best_eq = div, eq
    return best_div, best_eq


def process_file(path):
    """Read a results_*.json file → {cost: (profile_tuple, divergence)} dict."""
    with open(path, "r") as f:
        results = json.load(f)

    # aggregate decisions by cost inside that file
    scenarios = defaultdict(dict)  # cost → {player_id: decision}
    for entry in results:
        cost     = float(entry["cost"].split("=")[1].strip())
        pid      = int(entry["decision"].split("_")[1].split("=")[0].strip())
        decision = int(entry["decision"].split("=")[1].strip())
        scenarios[cost][pid] = decision

    out = {}
    for cost, decisions in scenarios.items():
        profile = tuple(decisions[i] for i in sorted(decisions))  
        div, _  = compare_to_theoretical(cost, list(profile))
        out[cost] = (profile, div)
    return out

# ---------------------------------------------------------------------
# 3. MAIN AGGREGATION LOOP OVER PROVIDERS
# ---------------------------------------------------------------------
provider_dirs = [
    os.path.join(tests_root_dir, d)
    for d in os.listdir(tests_root_dir)
    if os.path.isdir(os.path.join(tests_root_dir, d))
]

if not provider_dirs:
    raise RuntimeError("No provider sub‑folders found in /tests. Nothing to aggregate.")

for provider_dir in provider_dirs:
    provider = os.path.basename(provider_dir)
    pattern  = os.path.join(provider_dir, "results_*.json")
    result_files = sorted(
        glob.glob(pattern),
        key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0].split("_")[1]),
    )
    if not result_files:
        print(f"[{provider}] No results_*.json files found — skipping.")
        continue

    # Containers specific to this provider
    profile_counts   = defaultdict(Counter)   
    divergence_lists = defaultdict(list)     

    # --- aggregate over Monte‑Carlo runs
    for file in result_files:
        data = process_file(file)
        for cost, (profile, div) in data.items():
            profile_counts[cost][profile] += 1
            if div is not None:
                divergence_lists[cost].append(div)

    # -----------------------------------------------------------------
    # 4. PRINT SUMMARY
    # -----------------------------------------------------------------
    print(f"\n========== Provider: {provider} | {len(result_files)} runs ==========")

    print("\nProfile distribution:")
    for cost in sorted(profile_counts):
        print(f"  Cost c = {cost}:")
        for profile, count in profile_counts[cost].items():
            print(f"    profile {list(profile)} → {count}×")

    print("\nDivergence stats:")
    for cost in sorted(divergence_lists):
        vals = divergence_lists[cost]
        print(
            f"  Cost c = {cost}:  n={len(vals)}, "
            f"mean={np.mean(vals):.2f}, min={min(vals)}, max={max(vals)}"
        )

    # -----------------------------------------------------------------
    # 5. VISUALISATIONS  (saved in provider folder)
    # -----------------------------------------------------------------
    # 5‑A. Profile distribution bar charts
    costs_profile = sorted(profile_counts)
    fig, axs = plt.subplots(
        1, len(costs_profile), figsize=(5 * len(costs_profile), 5), sharey=True
    )
    if len(costs_profile) == 1:
        axs = [axs]
    cmap = plt.get_cmap("Pastel1")

    for ax, cost in zip(axs, costs_profile):
        profiles = list(profile_counts[cost])
        counts   = [profile_counts[cost][p] for p in profiles]
        labels   = ["".join(map(str, p)) for p in profiles]
        colors   = [cmap(i) for i in range(len(profiles))]
        ax.bar(labels, counts, color=colors)
        ax.set_title(f"Cost c = {cost}")
        ax.set_xlabel("Equilibrium Profile")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"Equilibrium Distribution – {provider}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    profile_plot = os.path.join(provider_dir, f"profile_distribution_{provider}.png")
    plt.savefig(profile_plot)
    plt.close()
    print(f"  ↳ saved {profile_plot}")

    # 5‑B. Mean divergence bar chart
    costs       = sorted(divergence_lists)
    mean_vals   = [np.mean(divergence_lists[c]) for c in costs]
    y_pos       = np.arange(len(costs))

    plt.figure(figsize=(6, 5))
    plt.bar(y_pos, mean_vals, color=cmap(1))
    plt.xticks(y_pos, [str(c) for c in costs])
    plt.xlabel("Cost (c)")
    plt.ylabel("Mean Hamming Divergence")
    plt.title(f"Mean Divergence – {provider}")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    mean_plot = os.path.join(provider_dir, f"mean_divergence_{provider}.png")
    plt.savefig(mean_plot)
    plt.close()
    print(f" saved {mean_plot}")
