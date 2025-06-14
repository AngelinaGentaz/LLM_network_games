"""
run_bestshot_game.py
--------------------
Command‑line driver for a one‑shot best‑shot (strategic‑substitutes)
game on the three five‑node networks used by Charness et al. (2014).

Example call
------------
python run_bestshot_game.py \
    --players A B C D E \
    --networks orange green purple \
    --experiment_id 1 \
    --provider openai
"""

import argparse
import json
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LLM_clients

# ======================== Helper: network database ========================= #

NETWORKS = {
    # Edges are undirected; each tuple listed once
    "orange": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "D")],
    "green":  [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],            # orange minus B‑D
    "purple": [("A", "B"), ("B", "C"), ("B", "D"), ("D", "E")],            # orange minus C‑D
}

def neighbours(label: str, edge_list):
    """Return sorted list of neighbours of `label` in `edge_list`."""
    nbs = {b if a == label else a
           for (a, b) in edge_list
           if a == label or b == label}
    return sorted(nbs)

def format_edges(edge_list):
    """Return a single‑line ASCII description of edges."""
    return "Edges: " + ", ".join([f"{a}-{b}" for (a, b) in edge_list])

# ======================== Prompts (ASCII‑only) ============================ #

SYSTEM_PROMPT = r"""
You are taking part in a one shot best shot (strategic substitutes) network game.

--------------------------------------------------------------------
GAME BASICS
--------------------------------------------------------------------
- Players: 5, labelled A, B, C, D, E.
- Network: the undirected graph for the current round is shown to all
  players before they choose.
- Actions: each player i chooses a_i in {0, 1},
           where 1 = Active / contribute, 0 = Inactive.

--------------------------------------------------------------------
PAYOFF FUNCTION  (common knowledge)
--------------------------------------------------------------------
For any action profile a = (a_A, …, a_E) and with N(i) the neighbours
of player i in the displayed graph,

  u_i(a) = 100 * 1{ a_i = 1 or (exists j in N(i) with a_j = 1) }
           - 50 * a_i.

The indicator 1{} is 1 when its condition is true and 0 otherwise.

Interpretation
--------------
- Being Active always costs 50.
- Any player (active or inactive) gains 100 if at least one member of
  her closed neighbourhood (herself plus neighbours) is Active.
- If no one in that set is Active, she earns 0.

--------------------------------------------------------------------
GAME RULES
--------------------------------------------------------------------
1. You will be assigned one labelled node (A‑E) for this round.
2. You observe the full network and know which labels other players
   occupy, but you do NOT observe their pending actions.
3. All players decide simultaneously.
4. Your goal is to maximise your own monetary payoff as defined above.
"""

USER_PROMPT_TEMPLATE = """
You are **Player {player_label}** in this round of the best shot game.

----------------------------------------------------------------
NETWORK FOR THIS ROUND
----------------------------------------------------------------
{network_description}
(Edges are undirected; each edge is listed once.)

Your neighbours: {neighbour_list}

----------------------------------------------------------------
ACTION CHOICES
----------------------------------------------------------------
- a_{player_label} = 1  then  Active / contribute   (pay cost 50)
- a_{player_label} = 0  then  Inactive              (no cost)

Payoff recap for you:
  u_{player_label}(a) = 100 if (a_{player_label}=1
                                or any neighbour is Active),
                       else 0,
                     minus 50 * a_{player_label}.

----------------------------------------------------------------
YOUR TASK
----------------------------------------------------------------
1. Decide whether to play a_{player_label} = 0 or a_{player_label} = 1.
2. Compute your WORST CASE and BEST CASE payoffs under each action:
   - Worst case: none of your neighbours are Active.
   - Best case: at least one neighbour is Active.

----------------------------------------------------------------
OUTPUT FORMAT  (JSON only no extra text)
----------------------------------------------------------------
{{
  "decision": "a_{player_label} = <0 or 1>",
  "expected_payoff": "a_{player_label}=1: Best=<…>, Worst=<…>; \
a_{player_label}=0: Best=<…>, Worst=<…>"
}}
"""

# ================================ Main ==================================== #

def main():
    parser = argparse.ArgumentParser(description="Run one‑shot best‑shot games on five‑node networks.")
    parser.add_argument("--players", nargs="+", type=str, required=True,
                        help="List of player labels (A B C D E)")
    parser.add_argument("--networks", nargs="+", type=str, required=True,
                        choices=list(NETWORKS.keys()),
                        help="List of network names to test (orange green purple)")
    parser.add_argument("--experiment_id", type=int, required=True,
                        help="Experiment iteration number")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "anthropic", "google", "mistral"])
    args = parser.parse_args()

    # -------------------------------------------------------------------- #
    #  Directory handling & API keys
    # -------------------------------------------------------------------- #
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(root_dir, ".env"))
    tests_dir = os.path.join(root_dir, "tests_charness")
    provider_dir = os.path.join(tests_dir, args.provider)
    os.makedirs(provider_dir, exist_ok=True)

    if args.provider == "anthropic":
        from LLM_clients.anthropic import call_anthropic_api as call_llm_api
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif args.provider == "openai":
        from LLM_clients.openai import call_openai_api as call_llm_api
        api_key = os.getenv("OPENAI_API_KEY")
    elif args.provider == "google":
        from LLM_clients.google import call_gemini_api as call_llm_api
        api_key = os.getenv("GEMINI_API_KEY")
    elif args.provider == "mistral":
        from LLM_clients.mistral import call_mistral_api as call_llm_api
        api_key = os.getenv("MISTRAL_API_KEY")
    else:
        raise ValueError("Unsupported provider.")

    if not api_key:
        raise ValueError("API key not found. Check your .env file.")

    # -------------------------------------------------------------------- #
    #  Run the games
    # -------------------------------------------------------------------- #
    results = []
    for net_name in args.networks:
        edge_list = NETWORKS[net_name]
        network_description = format_edges(edge_list)

        for player_label in args.players:
            nbs = neighbours(player_label, edge_list)
            neighbour_list = ", ".join(nbs) if nbs else "None"

            user_prompt = USER_PROMPT_TEMPLATE.format(
                player_label=player_label,
                network_description=network_description,
                neighbour_list=neighbour_list
            )

            print(f"Calling {args.provider}: Network={net_name}, Player={player_label} …")
            result = call_llm_api(
                api_key,                 
                SYSTEM_PROMPT,           
                user_prompt,             
                player_label,            
                net_name                 
            )
            results.append({
                "provider": args.provider,
                "network": net_name,
                "player": player_label,
                "llm_response": result
            })

    # -------------------------------------------------------------------- #
    #  Save
    # -------------------------------------------------------------------- #
    outfile = os.path.join(tests_dir, f"complete_substitutes_results_{args.experiment_id}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {outfile}")

if __name__ == "__main__":
    main()
