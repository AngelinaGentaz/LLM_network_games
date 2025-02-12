import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np


class CascadingFailureGame:
    def __init__(self, args):
        self.args = args

        # 4-node star network
        self.G = self.create_star_network()
        self.pos = nx.spring_layout(self.G)
        self.set_node_attributes()
        self.states = []

    def create_star_network(self):
        return nx.star_graph(3)  # Central hub (0) + 3 peripherals (1, 2, 3)

    def set_node_attributes(self):
        nx.set_node_attributes(self.G, {
            0: {'capacity': 1.0, 'load': 0.6, 'invested': False, 'failed': False},
            1: {'capacity': 0.5, 'load': 0.2, 'invested': False, 'failed': False},
            2: {'capacity': 0.5, 'load': 0.2, 'invested': False, 'failed': False},
            3: {'capacity': 0.5, 'load': 0.2, 'invested': False, 'failed': False}
        })

    def play_game(self):
        self.record_state("Initial State")

        # Phase 1: Defenders invest
        self.defender_investments()

        # Phase 2: Attacker targets a node
        self.attacker_attack()

        # Phase 3: Cascade propagation
        self.propagate_failures()

    def defender_investments(self):
        for node in self.G.nodes:
            if random.random() < 0.5:  # 50% chance to invest
                self.G.nodes[node]['capacity'] += self.args.capacity_boost
                self.G.nodes[node]['invested'] = True
        self.record_state("After Defender Investments")

    def attacker_attack(self):
        """Simulate attacker targeting the most critical node."""
        target = max(self.G.degree, key=lambda x: x[1])[0]  # Node with max degree
        success_prob = 1 / (1 + self.G.nodes[target]['invested'])

        if random.random() < success_prob:
            self.G.nodes[target]['failed'] = True
        self.record_state(f"Attack on Node {target}")

    def propagate_failures(self):
        """Simulate cascading failures."""
        while True:
            new_failures = []
            for node in self.G.nodes:
                if self.G.nodes[node]['failed']:
                    for neighbour in self.G.neighbors(node):
                        if not self.G.nodes[neighbour]['failed']:
                            # Redistribute load
                            self.G.nodes[neighbour]['load'] += self.G.nodes[node]['load']
                            if self.G.nodes[neighbour]['load'] > self.G.nodes[neighbour]['capacity']:
                                new_failures.append(neighbour)

            if not new_failures:
                break

            for node in new_failures:
                self.G.nodes[node]['failed'] = True

            self.record_state("Cascade in Progress")

        self.record_state("Final State")

    def record_state(self, description):
        state = {
            'nodes': {n: data.copy() for n, data in self.G.nodes(data=True)},
            'edges': list(self.G.edges),
            'description': description
        }
        self.states.append(state)

    def update_plot(self, frame):
        plt.clf()
        state = self.states[frame]

        colours = [
            'red' if state['nodes'][n]['failed'] else 'green'
            for n in self.G.nodes
        ]
        sizes = [
            500 if state['nodes'][n]['invested'] else 300
            for n in self.G.nodes
        ]

        nx.draw(
            self.G, self.pos,
            with_labels=True,
            node_color=colours,
            node_size=sizes,
            edge_color='black'
        )
        plt.title(state['description'])

    def animate(self):
        fig = plt.figure(figsize=(8, 6))
        ani = FuncAnimation(fig, self.update_plot, frames=len(self.states), repeat=False)
        plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Parameters
alpha = 1.0  # Load penalty sensitivity
beta = 2.0   # Failure penalty
gamma = 2.0  # Attacker gain per failed node
Delta_values = np.linspace(0, 1, 50)  # Capacity boost
c_values = np.linspace(0, 5, 50)      # Investment cost
gamma_values = np.linspace(1, 5, 50)  # Attacker gain
beta_values = np.linspace(0, 5, 50)   # Failure penalty

# Example fixed values
L_h = 0.6  # Hub load
L_p = 0.2  # Peripheral load
C_h = 1.0  # Hub capacity
C_p = 0.5  # Peripheral capacity
k_h = 1.0  # Attack cost

# Defender Utility vs. c
defender_util_c = [-c - alpha * L_h - beta if L_h > C_h else -c for c in c_values]

# Attacker Utility vs. gamma
attacker_util_gamma = [gamma * 4 - k_h for gamma in gamma_values]  # Max failures (hub + 3 peripherals)

# Failed Nodes vs. Delta
failed_nodes_delta = [4 if (L_p + L_h / 3 > C_p + Delta) else 1 for Delta in Delta_values]

# Failed Nodes vs. beta
failed_nodes_beta = [4 if beta < alpha else 1 for beta in beta_values]


plt.figure(figsize=(14, 10))

# Plot 1: Defender Utility vs. c
plt.subplot(2, 2, 1)
plt.plot(c_values, defender_util_c, label="Defender Utility")
plt.axhline(0, color='red', linestyle='--', label="Break-even")
plt.title("Defender Utility vs. Investment Cost (c)")
plt.xlabel("Investment Cost (c)")
plt.ylabel("Utility")
plt.legend()

# Plot 2: Attacker utility vs. gamma
plt.subplot(2, 2, 2)
plt.plot(gamma_values, attacker_util_gamma, label="Attacker Utility", color="orange")
plt.title("Attacker Utility vs. Gain per Failed Node (gamma)")
plt.xlabel("Attacker Gain (gamma)")
plt.ylabel("Utility")
plt.legend()

# Plot 3: Failed nodes vs. Delta
plt.subplot(2, 2, 3)
plt.plot(Delta_values, failed_nodes_delta, label="Failed Nodes", color="green")
plt.title("Failed Nodes vs. Capacity Boost (Delta)")
plt.xlabel("Capacity Boost (Delta)")
plt.ylabel("Total Failed Nodes")
plt.legend()

# Plot 4: Failed nodes vs. beta
plt.subplot(2, 2, 4)
plt.plot(beta_values, failed_nodes_beta, label="Failed Nodes", color="purple")
plt.title("Failed Nodes vs. Failure Penalty (beta)")
plt.xlabel("Failure Penalty (beta)")
plt.ylabel("Total Failed Nodes")
plt.legend()

plt.tight_layout()
plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascading Failure Game")
    parser.add_argument("--investment-cost", type=float, default=0.5, help="Cost of defender investment (c)")
    parser.add_argument("--capacity-boost", type=float, default=0.7, help="Capacity boost from investment (Δ)")
    parser.add_argument("--load-penalty", type=float, default=1.1, help="Load penalty sensitivity (α)")
    parser.add_argument("--attacker-gain", type=float, default=1.0, help="Attacker's gain per failed node (γ)")
    args = parser.parse_args()

    game = CascadingFailureGame(args)
    game.play_game()
    game.animate()
