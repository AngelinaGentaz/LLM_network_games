# Run 10 Monte Carlo 

cd src\\coordination_game

for i in {1..10}
do
    echo "Running experiment $i..."
    python simultaneous.py --players 1 2 3 4 --costs 0.5 1 2  --experiment_id $i --provider 'mistral'
done


