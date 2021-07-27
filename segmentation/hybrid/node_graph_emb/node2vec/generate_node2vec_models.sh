#!/bin/bash

for filename in /node-graph-emb/node2vec/graph/kasteren_house_a/reduced/*.edgelist; do
 python src/main.py --input "$filename" --output "${filename%?????????}.emb" --dimensions 50
done

for filename in /node-graph-emb/node2vec/graph/kasteren_house_b/*.edgelist; do
 python src/main.py --input "$filename" --output "${filename%?????????}.emb" --dimensions 50
done

for filename in /node-graph-emb/node2vec/graph/kasteren_house_c/*.edgelist; do
 python src/main.py --input "$filename" --output "${filename%?????????}.emb" --dimensions 50
done
