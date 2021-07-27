#!/bin/bash
for DATASET in "kasteren_house_a" "kasteren_house_b" "kasteren_house_c"
do
    MODEL_DIR="/activity_segmentation/results/LODO/${DATASET}/word2vec_models/vector_files/"
    for graph in "activities" "activities_from_data" "locations" "locations_from_data" "activities_locations" "activities_locations_from_data" "locations_context" "activities_locations_context"
    do
        files="$MODEL_DIR"*day.vector
        for entry in $files
        do
            filename="$(basename ${entry})"
            python3 retrofit.py -i "${entry}"  -l "lexicons/${DATASET}/actions_${graph}.edgelist" -n 10 -o "$MODEL_DIR${filename:0:10}_retrofitted_${graph}.vector"
        done
    done
done