#!/bin/bash


dataDir="/home/mandum/twoThirdsPNNL/applications/tokenJoin/datasets"

csv_files=(
  "aol_clean_1.csv"
  "flickr_clean.csv"
  "gdelt_clean.csv"
  "kosarak_clean.csv"
  "BMS-POS_clean.csv"
  "yelp_clean.csv"
  "dblp_clean.csv"
  "livejournal_clean.csv"
  "enron_clean.csv"
  "mind_clean.csv"
)
start=10000
limit=50000


for file in "${csv_files[@]}"; do
      echo "Processing: $file"
      python3 deltaVary.py ${dataDir}"/"${file} 50000 0 >> allData5.jaccard.delta.res
done

    