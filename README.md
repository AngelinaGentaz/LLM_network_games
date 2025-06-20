# LLM_network_games

Strategic Reasoning of LLM-Based Agents in Network Game Environments

This repository contains a small experiment exploring how large language models reason in a simple network coordination game.  The code lets you run agents backed by different LLM providers and aggregate the resulting play distributions.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   You will also need the SDKs for any providers you wish to use (`openai`, `anthropic`, `mistralai`, etc.).

2. **API keys**
   Create a `.env` file in the repository root containing the API keys for the models you intend to call.  For example:
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

Visualization utilities are provided:

- [`aggregator.py`](src/coordination_game/aggregator.py) collects the profile distributions from the result files and creates bar charts per provider.
- [`heatmap_equilibria.py`](src/coordination_game/heatmap_equilibria.py) plots a heatmap of Nash equilibrium frequencies across providers, costs and context framing perturbations.


## Repository layout

```
src/                    python modules and scripts
  LLM_clients/          wrappers for OpenAI, Gemini, Anthropic and Mistral APIs
  coordination_game/    coordination game driver and analysis tools
tests/                  sample result files for the coordination game
```

---