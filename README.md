# LLM_network_games

Strategic Reasoning of LLM-Based Players in Network Game Environments

This repository contains a small experiment exploring how LLMs converge or not to the Nash equilibrium in a simple network coordination game (strategic complements on a four node line network).  The code lets you run players backed by different LLM providers and aggregate the resulting play distributions.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   You will also need the SDKs for any providers you wish to use (`openai`, `anthropic`, `mistral`, etc.).

2. **API keys**
   Create a `.env` file in the repository root containing the API keys for the models you intend to call:
   ```
   OPENAI_API_KEY=...
   GEMINI_API_KEY=...
   ANTHROPIC_API_KEY=...
   MISTRAL_API_KEY=...
   ```


## Running the coordination game

The main driver for the coordination game is [`src/coordination_game/line_network.py`](src/coordination_game/line_network.py).  A typical call looks like:

```bash
python line_network.py --players 1 2 3 4 --costs 0.5 1 2 \
    --experiment_id 1 --provider openai --cfp min --neip baseline
```

Results are written to `tests/<provider>/results_<id>.json`.


### Workflow overview

![Workflow of the line network game](images/workflow_codebase.png)

### Robustness 
We define a Nash Equilibirum Invariant Perturbation (NEIP) as a modification to the numerical values within a game’s payoff structure (in the system prompt) that preserves the set of Nash equilibria. A Context Framing Perturbation (CFP) can be viewed as a special —purely linguistic— instance of a NEIP in which the perturbation acts only on the textual presentation of the game (e.g.tone, narrative embedding, role labels etc.).  


Visualization utilities are provided:

- [`aggregator.py`](src/coordination_game/aggregator.py) collects the profile/equilibirum distributions from the result files.
- [`heatmap_equilibria.py`](src/coordination_game/heatmap_equilibria.py) plots a heatmap of the Nash equilibrium probability across models, costs and Context Framing Perturbations (CFP).
- [`lineplots_equilibria.py`](src/coordination_game/lineplots_equilibria.py) generates line plots and grouped bar charts of equilibrium probability and
  average Hamming distance across providers.
- [`compare_neip_min.py`](src/coordination_game/compare_neip_min.py) compares baseline results (min NEIP) with the numerical NEIP (NEIP100).


## Repository layout

```
src/                    python modules and scripts
  LLM_clients/          wrappers for OpenAI, Gemini, Anthropic and Mistral APIs
  coordination_game/    coordination game driver and analysis tools
tests/                  sample result files for the coordination game
experiment1.sh/         run Monte Carlo simulations for the coordination game
```

---