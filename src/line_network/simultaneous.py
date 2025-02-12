import os
import anthropic
import argparse
import json
from dotenv import load_dotenv

def call_anthropic_api(client, system_prompt, player_id, cost):
    """
    Call the Anthropic API for a specific player and cost.
    """
    # Format the user prompt with the given player_id and cost
    user_prompt = f"""
    You are Player {player_id} in a coordination game played on a line network.

    **Game Context:**
    - You must decide whether to coordinate (a_{player_id} = 1) or not (a_{player_id} = 0).

    **Current Situation:**
    - The cost of coordination is c = {cost}.

    **Your Objective:**
    - Maximise your payoff:
    [
    u_{player_id}(a_{player_id}, a_-{player_id}) = sum_j in N({player_id}) delta(a_{player_id} = a_j) - {cost} cdot a_{player_id},
    ]
    where:
    - ( delta(a_{player_id} = a_j) ): 1 if your strategy matches that of your neighbour ( j ), and 0 otherwise.

    **Your Task:**
    1. Based on your position and the game's context, decide whether a_{player_id} = 0 or a_{player_id} = 1.
    2. Provide reasoning for your decision. Be concise.
    3. Evaluate your expected payoffs for both a_{player_id} = 0 and a_{player_id} = 1, explaining the worst and best-case outcomes.

    **Output Format:**
    Respond in the following JSON format:
      "cost": "c = {cost}",
      "decision": "a_{player_id} = <your decision>",
      "reasoning": "<explain why you chose this strategy>",
      "expected_payoff": "a_{player_id}=1: Best=<best expected payoff of Player {player_id} when playing 1>, Worst=<worst expected payoff of Player {player_id} when playing 1; a_{player_id}=0: Best=<best expected payoff of Player {player_id} when playing 0>, Worst=<worst expected payoff of Player {player_id} when playing 0>"
    Do NOT write down anything else in your output.
    """
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1500,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    

    response_text = response.content[0].text
    try:
        parsed_response = json.loads(response_text)
        return json.dumps(parsed_response, indent=2)
    except json.JSONDecodeError:
        return response_text 

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(root_dir, ".env")
    load_dotenv(dotenv_path=env_path)
    tests_dir = os.path.join(root_dir, "tests")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set ANTHROPIC_API_KEY in your .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    # Define the system prompt
    system_prompt = r"""
    You are participating in a coordination game played on a line network.

    **Network Topology:**
    - There are 4 players: Player 1, Player 2, Player 3, and Player 4.
    - Players are connected in a line:
      - Player 1 is connected to Player 2.
      - Player 2 is connected to Player 1 and Player 3.
      - Player 3 is connected to Player 2 and Player 4.
      - Player 4 is connected to Player 3.

    **Strategy Space:**
    - Each player chooses a strategy (a_i) from {0, 1}, where:
      - a_i = 1: You coordinate.
      - a_i = 0: You do not coordinate.

    **Payoff Function:**
    - Your payoff depends on your own strategy \( a_i \), the strategies of your neighbours, and the cost of coordinating (\( c \)):
    \[
    u_i(a_i, a_{-i}) = \sum_{j \in N(i)} \delta(a_i = a_j) - c \cdot a_i,
    \]
    where:
    - \( \delta(a_i = a_j) = 1 \) if your strategy matches your neighbour's strategy, and \( 0 \) otherwise.
    - \( c > 0 \): Cost of choosing \( a_i = 1 \).

    **Game Rules:**
    - You are assigned one player (e.g., Player 1).
    - You will make a decision simultaneously with other players based on your current understanding of their potential strategies.
    - The objective is to maximise your own payoff.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a coordination game on a line network.")
    parser.add_argument("--players", nargs="+", type=int, required=True, help="List of player IDs (1 2 3 4)")
    parser.add_argument("--costs", nargs="+", type=float, required=True, help="List of cost values (e.g., 0.1 0.5 1.0)")
    parser.add_argument("--experiment_id", type=int, required=True, help="Experiment iteration number")
    args = parser.parse_args()

    # Run the experiments
    results = []
    for player_id in args.players:
        for cost in args.costs:
            print(f"Processing Player {player_id} with cost {cost}...")
            response = call_anthropic_api(client, system_prompt, player_id, cost)
            results.append(json.loads(response))

    experiment_id = args.experiment_id
    results_path = os.path.join(tests_dir, f"results_{experiment_id}.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Experiment {experiment_id} completed. Results saved to {results_path}")

if __name__ == "__main__":
    main()
