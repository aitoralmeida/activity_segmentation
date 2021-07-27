#!/bin/bash
DATASET=$1
CONFIG=$2
NUM_EXEC=$3
for STRATEGY in "word2vec_context" "word2vec_context_desc"
do
    MODEL_DIR="/activity_segmentation/results/${DATASET}/${STRATEGY}/"
    MODEL_FOLDER="${CONFIG}/train/word2vec_models/"
    for graph in "activities" "activities_from_data" "locations" "locations_from_data" "activities_locations" "activities_locations_from_data"
    do
        let END=$NUM_EXEC
        for ((E=0;E<END;E++))
        do
            python3 retrofit.py -i "${MODEL_DIR}${MODEL_FOLDER}${E}_execution.vector"  -l "lexicons/${DATASET}/actions_${graph}.edgelist" -n 10 -o "${MODEL_DIR}${MODEL_FOLDER}${E}_execution_retrofitted_${graph}.vector"
        done
    done
done