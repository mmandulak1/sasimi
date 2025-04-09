#!/bin/bash


dataDir="/home/mandum/twoThirdsPNNL/applications/tokenJoin/datasets"
file="enron_clean.csv"
start=10000
limit=40000

for i in $(seq $start 10000 $limit); do
    python3 test.py ${dataDir}"/"${file} $i >> ${file}.tjRes
done