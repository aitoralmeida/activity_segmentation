for run in {1..100}; do
  python3 transformer_flatten_next_action_prediction.py --optimizer adamax
done
for run in {1..100}; do
  python3 transformer_global_avg_next_action_prediction.py --optimizer adamax
done