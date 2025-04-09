import sys
import os
import re 

if len(sys.argv) != 2:
    print("python3 ", sys.argv[0], " [DIR]")
    exit()


directory = sys.argv[1]


# GRAPH, METHOD, #SETS, INIT, GEN, REF, VER, TOTAL, #SURV, RECALL, PREC, AVG%DIFF, MAX%DIFF
print("Graph,Method,#Sets,Init,Gen,Ref,Ver,Total,#Surv,Recall,Prec")


dataset = []
recalls = []
precisions = []

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    if os.path.isfile(filepath):  # Ensure it's a file
        with open(filepath, 'r', encoding='utf-8') as file:
            filename = filepath.split('/')[-1].split('.')[0]
            #print(filename)
            method = ""
            idx = 0
            dataname = ""
            for line in file:
                #print(line.strip())  # Process each line
                if '.csv' in line:
                    dataname = line.rstrip().split('.')[0]

                if 'Progress' in line:
                    numsets = int(line.rstrip().split('/')[-1].replace(',',""))
                if "HUNGARIAN" in line:
                    method = 'HG'
                elif "LOCALLY-DOMINANT" in line:
                    method = 'LD'
                elif "PS METHOD" in line:
                    method = 'PS'
                elif "GREEDY" in line:
                    method = 'GD'
                if 'Cand Ver' in line:
                    timings = [float(num) for num in re.findall(r"\d+\.\d+", line)]
                
                if 'Survived' in line:
                    numsurv = int(line.rstrip().split(':')[-1].replace(',',""))
                    dataset.append([dataname, method, numsets, timings[0],timings[1],timings[2],timings[3], timings[4],numsurv])

                if 'Recall' in line:
                    recalls.append(float(line.rstrip().split(' ')[-1]))
                
                if 'Precision' in line:
                    precisions.append(float(line.rstrip().split(" ")[-1]))
                
                idx += 1

recallIdx = 0
for idx in range(len(dataset)):
    if dataset[idx][1] == 'LD' or dataset[idx][1] == 'PS' or dataset[idx][1] == 'GD':
        dataset[idx].append(recalls[recallIdx])
        dataset[idx].append(precisions[recallIdx])
        recallIdx += 1 
    else:
        dataset[idx].append(0.0)
        dataset[idx].append(0.0)

for entry in dataset:
    print(",".join(map(str, entry))) 