import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(current_dir, "../../tests") 
tests_dir = os.path.normpath(tests_dir)

decision_counts = defaultdict(lambda: defaultdict(lambda: {"ai=1": 0, "ai=0": 0}))

for filename in os.listdir(tests_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(tests_dir, filename)
        
        with open(filepath, "r") as f:
            data = json.load(f) 
            
            # process each player's decision
            for entry in data:
                cost = entry["cost"] 
                decision_str = entry["decision"]  
                
                player_id, decision_value = decision_str.split("=")
                player_id = player_id.strip()   
                decision_value = decision_value.strip() 

                if decision_value == "1":
                    decision_counts[cost][player_id]["ai=1"] += 1
                else:
                    decision_counts[cost][player_id]["ai=0"] += 1

# Generate separate bar charts for each cost level
for cost, players in decision_counts.items():
    plt.figure(figsize=(8, 5))
    
    labels = list(players.keys()) 
    ai_1_counts = [players[p]["ai=1"] for p in labels]
    ai_0_counts = [players[p]["ai=0"] for p in labels]

    plt.bar(labels, ai_1_counts, label="ai=1 (Coordinated)", color="blue")
    plt.bar(labels, ai_0_counts, bottom=ai_1_counts, label="ai=0 (Not Coordinated)", color="orange")

    plt.xlabel("Players")
    plt.ylabel("Number of Decisions")
    plt.title(f"Decision Distribution per Player (Cost = {cost})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(tests_dir, f"decision_distribution_{cost}.png")
    plt.savefig(plot_path)


