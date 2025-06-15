import os
import glob
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

"""
aggregate_complete_info.py

Aggregates Nash equilibrium profile distributions for
complete-information network games (substitutes & complements)
across providers, and network topology
"""

# ---------------------------------------------------------------------
# 0. PATHS
# ---------------------------------------------------------------------
root_dir       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tests_root_dir = os.path.join(root_dir, "tests_charness")

if not os.path.isdir(tests_root_dir):
    raise RuntimeError(f"No tests directory found at {tests_root_dir}")

# ---------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------
GAME_TYPES    = ["substitutes", "complements"]
FILE_PREFIXES = {
    "substitutes": "complete_substitutes_results_",
    "complements": "complete_complements_results_"
}
LBL_ORDER     = ["A","B","C","D","E"]

# ---------------------------------------------------------------------
# 2. AGGREGATOR DATA STRUCTURE
# ---------------------------------------------------------------------
# counts[provider][game_type][network] = Counter({ profile_tuple: frequency })
counts = defaultdict(lambda: {
    "substitutes": defaultdict(Counter),
    "complements": defaultdict(Counter)
})

# ---------------------------------------------------------------------
# 3. MAIN LOOP: READ & AGGREGATE
# ---------------------------------------------------------------------
providers = sorted([
    d for d in os.listdir(tests_root_dir)
    if os.path.isdir(os.path.join(tests_root_dir, d))
])

for provider in providers:
    provider_dir = os.path.join(tests_root_dir, provider)

    for game_type in GAME_TYPES:
        prefix  = FILE_PREFIXES[game_type]
        pattern = os.path.join(provider_dir, f"{prefix}*.json")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"[{provider} | {game_type}] No result files found, skipping.")
            continue

        for path in files:
            with open(path, "r") as f:
                entries = json.load(f)

            # collect decisions per network
            by_network = defaultdict(dict)  # net -> { player_label: decision_int }
            for entry in entries:
                net      = entry.get("network")
                resp     = entry.get("llm_response", {})
                dec_text = resp.get("decision", "").strip()

                # support multi-line decision strings
                for line in dec_text.splitlines():
                    if "=" not in line:
                        continue
                    key_part, val_part = line.split("=", 1)
                    val_str = val_part.strip()
                    try:
                        val = int(val_str)
                    except ValueError:
                        continue

                    # derive label: prefer explicit 'player' field
                    player = entry.get("player")
                    if player:
                        label = player
                    else:
                        parts = key_part.strip().split("_")
                        label = parts[-1]

                    by_network[net][label] = val

            # tally a complete-profile per network
            for net, decisions in by_network.items():
                profile = tuple(decisions.get(lbl) for lbl in LBL_ORDER)
                if None in profile:
                    # skip incomplete runs
                    continue
                counts[provider][game_type][net][profile] += 1

# ---------------------------------------------------------------------
# 4. PRINT SUMMARY
# ---------------------------------------------------------------------
for provider in providers:
    print(f"\n===== Provider: {provider} =====")
    for game_type in GAME_TYPES:
        nets = counts.get(provider, {}).get(game_type, {})
        print(f"\n-- Game type: complete-info {game_type} --")
        if not nets:
            print("  (no data)")
            continue
        for net, counter in nets.items():
            total = sum(counter.values())
            print(f"\nNetwork: {net}  [total runs: {total}]")
            if total == 0:
                print("  No complete equilibria found.")
                continue
            for profile, freq in counter.most_common():
                pct = freq / total
                print(f"  Profile {list(profile)} → {freq} runs ({pct:.1%})")

# ---------------------------------------------------------------------
# 5. PLOTTING: COMBINED SIDE-BY-SIDE PLOTS WITH PASTEL AESTHETIC
# ---------------------------------------------------------------------
for provider in providers:
    provider_dir = os.path.join(tests_root_dir, provider)
    output_dir   = provider_dir
    os.makedirs(output_dir, exist_ok=True)

    for game_type in GAME_TYPES:
        nets = counts.get(provider, {}).get(game_type, {})
        if not nets:
            print(f"[{provider} | {game_type}] No data to plot.")
            continue

        networks = sorted(nets.keys())
        fig, axs = plt.subplots(
            1,
            len(networks),
            figsize=(6 * len(networks), 5),
            sharey=True
        )
        if len(networks) == 1:
            axs = [axs]
        cmap = plt.get_cmap("Pastel1")

        for ax, net in zip(axs, networks):
            counter  = nets[net]
            profiles = list(counter.keys())
            freqs    = [counter[p] for p in profiles]
            labels   = ["".join(map(str, p)) for p in profiles]
            colors   = [cmap(i) for i in range(len(profiles))]

            ax.bar(labels, freqs, color=colors)
            ax.set_title(f"{net}")
            ax.set_xlabel("Equilibrium profile")
            if ax is axs[0]:
                ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle(f"Nash Equilibrium Distribution — {provider} ({game_type})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = os.path.join(
            output_dir,
            f"{provider}_{game_type}_distribution.png"
        )
        plt.savefig(out_path)
        plt.close()
        print(f"  ↳ saved {out_path}")
