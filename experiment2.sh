# Run 10 Monte Carlo 

cd src\\charness

for i in {1..10}
do
    echo "Running experiment $i..."
    python complete_info_complements.py --players A B C D E --networks 'orange' 'green' 'purple'  --experiment_id $i --provider 'anthropic'
done
