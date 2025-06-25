# Run 10 Monte Carlo 

cd src\\coordination_game

for i in {11..40}
do
    echo "Running experiment $i..."
    python line_network.py --players 1 2 3 4 --costs 0.5 1 2  --experiment_id $i --provider mistral --cfp min safety peace --neip baseline
done


