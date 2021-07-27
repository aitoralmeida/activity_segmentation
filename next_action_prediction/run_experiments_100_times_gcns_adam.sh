for run in {1..100}; do
  python3 gcn_next_action_prediction.py
done
for run in {1..100}; do
  python3 gcn_cnn_next_action_prediction.py
done
for run in {1..100}; do
  python3 gcn_lstm_next_action_prediction.py
done