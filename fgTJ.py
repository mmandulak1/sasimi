import pandas as pd
import sys
import resource
import time
import tracemalloc
#from edit.join_topk import TokenJoin as KTokenJoin
#from edit.join_delta import TokenJoin
from jaccard.join_delta import TokenJoin
from utils.verification import graphN 
from utils.verification import graphM
from utils.verification import countMatchInst
from jaccard.join_delta import numVerified
from jaccard.join_delta import countSets
from jaccard.join_delta import totalEpS

def average(lst):
    return sum(lst) / len(lst) if lst else 0

def calc_recall(dforig, dfnew, text):
    dforig['id_tup'] = list(zip(dforig.l_id, dforig.r_id))
    dfnew['id_tup'] = list(zip(dfnew.l_id, dfnew.r_id))
    compSetnew = set(dfnew['id_tup'])
    compSetold = set(dforig['id_tup'])
    #print(set(df1['id_tup']))
    tp = 0
    fn = 0
    for val in dforig['id_tup']:
        if val in compSetnew:
            tp += 1
        else:
            fn += 1
    print(text," Recall: ", (tp/(tp+fn)))

def calc_precision(dforig, dfnew, text):
    dforig['id_tup'] = list(zip(dforig.l_id, dforig.r_id))
    dfnew['id_tup'] = list(zip(dfnew.l_id, dfnew.r_id))
    compSetnew = set(dfnew['id_tup'])
    compSetold = set(dforig['id_tup'])
    tp = 0
    fp = 0
    for val in dforig['id_tup']:
        if val in compSetnew:
            tp += 1
    for val in dfnew['id_tup']:
        if val not in compSetold:
            fp += 1
    print(text, " Precision: ", (tp/(tp+fp)))


if len(sys.argv) < 3:
    print("python3 ", sys.argv[0], " [File] [|D|% ( (0.0,1.0] ) ] [Opt. - Delta]")
    exit()


file = sys.argv[1]
print(file.split('/')[-1])

vsetOut = 0
#vsetOut = file.split('/')[-1] + ".vout"

if len(sys.argv) == 4:
    delta = float(sys.argv[3])
else:
    delta = 0.7

print("Delta: ",delta)

df = pd.read_csv(file, header=None)
df = df.reset_index(drop=False)
df.columns = ['id', 'text']
df.text = df.text.apply(lambda x: str(x).split(';'))
df.text = df.text.apply(lambda x: list(set(x)))

samples = int(float(sys.argv[2]) * len(df.index))
print("Samples: #:",samples, " %:",float(sys.argv[2])*100)
if sys.argv[2] != 1.0:
    df = df.sample(samples,random_state=1928).reset_index(drop=True)

#MATCH ALGS: 0 = Hungarian, 1 = LD, 2 = Streaming



#tracemalloc.start()
#print("HUNGARIAN")
#df_hung = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=0,printSets=vsetOut)
#current, peak = tracemalloc.get_traced_memory()
#print(peak/10**3,'KB')
#tracemalloc.clear_traces()


#print("EV METHOD")
#df_ev = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=2,matchAlg=0,printSets=vsetOut)
#current, peak = tracemalloc.get_traced_memory()
#print(peak/10**3,'KB')
#tracemalloc.clear_traces()

#print("GREEDY")
#df_gd = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=3,printSets=0)
#print("LOCALLY-DOMINANT")
#df_ld = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=1,printSets=0)


print("PS METHOD")
df_stream = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=5,printSets=0)
print("EpS of After Cand Generation: ",totalEpS[0]/countSets[0])
#current, peak = tracemalloc.get_traced_memory()
#print(peak/10**3,'KB')

#print(df_hung,df_stream)

#tracemalloc.stop()

#print("Average Graph Sizes: N:", average(graphN[0]), " M:", average(graphM[0]))
#print("Matching Instances Alg: ",countMatchInst[0], " | Percent: ", (countMatchInst[0]/numVerified[0])*100)
#print("Non-Matching Instances: ",(countMatchInst[-1]/numVerified[0])*100)

#print("Matching Instances: ",len(graphN[0]), " | Percent: ", (len(graphN[0])/numVerified[0])*100)

#calc_recall(df_hung,df_gd,"GD Matching")
#calc_precision(df_hung,df_gd,"GD Matching")
#calc_recall(df_hung,df_ld,"LD Matching")
#calc_precision(df_hung,df_ld,"LD Matching")
#calc_recall(df_hung,df_stream,"PS Matching")
#calc_precision(df_hung,df_stream,"PS Matching")

#calc_recall(df_hung,df_stream)
#output_df_1 = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=1)
#output_df_1 = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=0)
#output_df_1 = TokenJoin().tokenjoin_self(df, id='id', join='text',k=2500,delta_alg=1,matchAlg=1)
#output_df_1 = TokenJoin().tokenjoin_self(df, id='id', join='text',k=2500,delta_alg=1,matchAlg=0)
#output_df_1 = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=0,k=2500,delta_alg=1)




