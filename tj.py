import pandas as pd
import sys
#from edit.join_topk import TokenJoin as KTokenJoin
#from edit.join_delta import TokenJoin
from jaccard.join_delta import TokenJoin
from utils.verification import graphN 
from utils.verification import graphM
from utils.verification import countMatchInst
from jaccard.join_delta import numVerified

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
    #df = df.sample(samples).reset_index(drop=True)
    df = df.sample(50000,random_state=1).reset_index(drop=True)



#MATCH ALGS: 0 = Hungarian, 1 = LD, 2 = PS, 3 = Greedy, 4 = LD_opt, 5 = PS_opt, 6 = GD_opt

#print("HUNGARIAN")
#df_hung = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=0,printSets=vsetOut)
print("EV METHOD")
df_ev = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=2,matchAlg=0,printSets=vsetOut)

print("GREEDY")
df_gd = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=6,printSets=0)
print("LOCALLY-DOMINANT")
df_ld = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=4,printSets=0)

print("PS METHOD")
df_stream = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=5,printSets=0)
print("PS2 METHOD")
df_stream_2 = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=7,printSets=0)

print("PSSHIFT METHOD")
df_stream_shift = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=8,printSets=0)
print("PS2SHIFT METHOD")
df_stream_2_shift = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=9,printSets=0)

print("GDSHIFT METHOD")
df_gd_shift = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=10,printSets=0)
print("LDSHIFT METHOD")
df_ld_shift = TokenJoin().tokenjoin_self(df, id='id', join='text', posFilter=True, jointFilter=True,verification_alg=-1,matchAlg=11,printSets=0)

#print("Average Graph Sizes: N:", average(graphN[0]), " M:", average(graphM[0]))
#print("Matching Instances Alg: ",countMatchInst[0], " | Percent: ", (countMatchInst[0]/numVerified[0])*100)
#print("Non-Matching Instances: ",(countMatchInst[-1]/numVerified[0])*100)

#print("Matching Instances: ",len(graphN[0]), " | Percent: ", (len(graphN[0])/numVerified[0])*100)


#calc_recall(df_hung,df_ld,"LD Matching")
#calc_precision(df_hung,df_ld,"LD Matching")
calc_recall(df_ev,df_stream,"PS Matching")
calc_precision(df_ev,df_stream,"PS Matching")
calc_recall(df_ev,df_stream_2,"PS2 Matching")
calc_precision(df_ev,df_stream_2,"PS2 Matching")

calc_recall(df_ev,df_stream_shift,"PSShift Matching")
calc_precision(df_ev,df_stream_shift,"PSShift Matching")
calc_recall(df_ev,df_stream_2_shift,"PS2Shift Matching")
calc_precision(df_ev,df_stream_2_shift,"PS2Shift Matching")

calc_recall(df_ev,df_ld,"LD Matching")
calc_precision(df_ev,df_ld,"LD Matching")
calc_recall(df_ev,df_ld_shift,"LDSHIFT Matching")
calc_precision(df_ev,df_ld_shift,"LDSHIFT Matching")

calc_recall(df_ev,df_gd,"GD Matching")
calc_precision(df_ev,df_gd,"GD Matching")
calc_recall(df_ev,df_gd_shift,"GDSHIFT Matching")
calc_precision(df_ev,df_gd_shift,"GDSHIFT Matching")




