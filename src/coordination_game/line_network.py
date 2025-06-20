import argparse
import json
from dotenv import load_dotenv
import prompts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LLM_clients

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a coordination game on a line network.")
    parser.add_argument("--players", nargs="+", type=int, required=True, help="List of player IDs (1 2 3 4)")
    parser.add_argument("--costs", nargs="+", type=float, required=True, help="List of cost values (e.g., 0.1 0.5 1.0)")
    parser.add_argument("--experiment_id", type=int, required=True, help="Experiment iteration number")
    parser.add_argument("--provider", type=str, required=True, default = "google")
    parser.add_argument("--cfp", nargs="+", type=str, default="baseline", help="Context Framing Perturbation")
    parser.add_argument("--neip", type=str, default="baseline", help="Nash Equilibrium Invariant Perturbation")
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(root_dir, ".env"))
    base_tests_dir = os.path.join(root_dir, "tests")
    provider_dir = os.path.join(base_tests_dir, args.provider)
    os.makedirs(provider_dir, exist_ok=True)

    # Load API key and appropriate function
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

    if not api_key:
        raise ValueError("API key not found. Check your .env file.")

   # Run experiments
    results = []
    for player_id in args.players:
        for cost in args.costs:
            for cfp in args.cfp:       
                user_prompt_template = prompts.get_user_prompt(player_id, cost, cfp=cfp)
                user_prompt = user_prompt_template.format(player_id=player_id, cost=cost)
                print(f"Calling {args.provider} for Player {player_id} with cost {cost} under {cfp}...")
                system_prompt = prompts.get_system_prompt(args.neip)
                result = call_llm_api(api_key, system_prompt, user_prompt, player_id, cost)
                results.append({
                        "provider": args.provider,
                        "neip": args.neip,
                        "cfp": cfp,
                        "llm_response": result
                    })

    # Save results
    with open(os.path.join(provider_dir, f"results_neip_{args.experiment_id}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved for {args.provider} in experiment {args.experiment_id}.")

if __name__ == "__main__":
    main()