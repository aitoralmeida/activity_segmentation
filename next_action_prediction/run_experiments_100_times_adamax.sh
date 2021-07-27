# vanilla
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings False --graph_to_retrofit none --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings False --graph_to_retrofit none --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings False --graph_to_retrofit none --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings False --graph_to_retrofit none --optimizer adamax
done
# activities graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities --optimizer adamax
done
# locations graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations --optimizer adamax
done
# activities_locations graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations --optimizer adamax
done
# activities_from_data graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_from_data --optimizer adamax
done
# locations_from_data graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit locations_from_data --optimizer adamax
done
# activities_locations_from_data graph
for run in {1..100}; do
  python3 cnn_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 cnn_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations_from_data --optimizer adamax
done
for run in {1..100}; do
  python3 lstm_attention_next_action_prediction.py --retrofitted_embeddings True --graph_to_retrofit activities_locations_from_data --optimizer adamax
done