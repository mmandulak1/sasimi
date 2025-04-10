from math import floor, ceil
import pandas as pd
from time import time
from utils.verification import verification, verification_opt, jaccard, verification_ps
from utils.utils import binary_search, binary_search_dupl, post_basic, post_positional
from jaccard.join_utils import transform_collection, build_stats_for_record, build_index


hg_scores = []
ld_scores = []
ps_scores = []

verSet = []

numVerified = []

def min_index(lst):
    return lst.index(min(lst)) if lst else None 

def max_index(lst):
    return lst.index(max(lst)) if lst else None 

def average(lst):
    return sum(lst) / len(lst) if lst else 0

def percent_difference(value1, value2):
    if (value1 + value2)/2 == 0:
        return 0
    return (abs(value1 - value2) / ((value1 + value2) / 2)) * 100


def post_joint(R, S, tokens, idx, pers_delta, util_gathered, sum_stopped, pos_tok):
    for (tok, tok_info) in tokens[pos_tok:]:
        sum_stopped -= tok_info['utility']
        if tok in idx[S]:
            tok_info_S = idx[S][tok]
            util_gathered += tok_info['utility']

            if pers_delta - (util_gathered + sum_stopped) > .0000001:
                return (util_gathered + sum_stopped)
            
            if pers_delta - (util_gathered + tok_info_S['rest']) > .0000001:
                return (util_gathered + tok_info_S['rest'])
        else:
            if pers_delta - (util_gathered + sum_stopped) > .0000001:
                return (util_gathered + sum_stopped)
      
    total = util_gathered + sum_stopped
    
    for (tok, tok_info) in tokens:
        if tok in idx[S]:
            total -= tok_info['utility']
            
            tok_info_S = idx[S][tok]
            minLen = min(len(tok_info['utilities']), len(tok_info_S['utilities'])) - 1
            util_score = min(tok_info['utilities'][minLen], tok_info_S['utilities'][minLen])
            total += util_score

        if pers_delta - total > .0000001:
            return total
        
    return total    

def simjoin(collection1, collection2, delta, idx, lengths_list, jointFilter, posFilter, verification_alg, log, matchAlg,printSets):

    selfjoin = collection1 == collection2
    
    init_time = candgen_time = candref_time = candver_time = 0
    no_candgen = no_candref = no_candver = no_candres = 0
    output = []
    for R, (R_id, R_rec) in enumerate(collection1):
        
        if R % 100 == 0:
            print("\rProgress {:,}/{:,}".format(R, len(collection1)), end='')
            log[f'no_{R}'] = no_candres
        
        t1 = time()
        ## Starting Initialization ##
        RLen = len(R_rec)
        #print(RLen,R_rec)
        
        if selfjoin:
            tokens = idx[R]
        else:
            tokens = build_stats_for_record(R_rec)
        
        tokens = sorted(tokens.items(), key=lambda x: x[0])
        sum_stopped = RLen
        
        RLen_max = floor(RLen / delta)
        
        if selfjoin:
            theta = 2 * delta / (1 + delta) * RLen
        else:
            theta = delta * RLen
            RLen_min = ceil(RLen * delta)

        ## Ending Initialization ##
        t2 = time()
        init_time += t2-t1
        
        
        t1 = time()
        cands_scores = {}
        ## Starting Candidate Generation ##
#        for pos_tok, (tok, tok_info) in enumerate(tokens):
#            if theta - sum_stopped > 0.0000001:
#                break
        pos_tok = 0
        while sum_stopped - theta > 0.0000001:
            if pos_tok >= len(tokens):
                break
            (tok, tok_info) = tokens[pos_tok]
            pos_tok += 1
                
            sum_stopped -= tok_info['utility']
            
            if tok < 0:
                continue

            if selfjoin:
                true_min = binary_search(lengths_list[tok], R)

                for S in lengths_list[tok][true_min:]:
                    if R == S:
                        continue

                    if len(collection2[S][1]) > RLen_max:
                        break

                    if S not in cands_scores:
                        cands_scores[S] = 0
                    cands_scores[S] += tok_info['utility']

            else:
                
                true_min = binary_search_dupl(lengths_list[tok], RLen, collection2)
                for S in lengths_list[tok][true_min:]:
                    if len(collection2[S][1]) > RLen_max:
                        break

                    if S not in cands_scores:
                        cands_scores[S] = 0
                    cands_scores[S] += tok_info['utility']
                
                true_min -= 1   # true_min examined in previous increasing parsing
                if true_min >= 0:    # reached start of inv list and -1 will go circular
                    for S in lengths_list[tok][true_min::-1]:
                        if len(collection2[S][1]) < RLen_min:
                            break
        
                        if S not in cands_scores:
                            cands_scores[S] = 0
                        cands_scores[S] += tok_info['utility']        
                '''
                for S in lengths_list[tok]:
                    if RLen_min > len(collection2[S][1]) > RLen_max:
                        continue

                    if S not in cands_scores:
                        cands_scores[S] = 0
                    cands_scores[S] += tok_info['utility']
                '''
                
        ## Ending Candidate Generation ##
        t2 = time()
        candgen_time += t2-t1
        no_candgen += len(cands_scores)
        
        ## Starting Candidate Refinement ##
        for S, util_gathered in cands_scores.items():
            t1 = time()
            (S_id, S_rec) = collection2[S]
            SLen = len(S_rec)

            pers_delta = delta / (1.0 + delta) * (RLen + SLen);
            total = sum_stopped + util_gathered

            if pers_delta - total > .0000001:
                t2 = time()
                candref_time += t2-t1
                continue
                
            no_candref += 1                

            if jointFilter:
                UB = post_joint(R, S, tokens, idx, pers_delta, util_gathered, sum_stopped, pos_tok)
            elif posFilter:
                UB = post_positional(R, S, tokens, idx, pers_delta, util_gathered, sum_stopped, pos_tok)
            else:
                UB = post_basic(R, S, tokens, idx, pers_delta, total, pos_tok)

            if pers_delta - UB > .0000001:
                t2 = time()
                candref_time += t2-t1                
                continue

            no_candver += 1


            if matchAlg == 0:
                verSet.append((R_rec,S_rec))

            t1 = time()
            #score = verification(R_rec, S_rec)
            if verification_alg >= 0:
                if RLen < SLen:
                    score = verification_opt(R_rec, S_rec, jaccard, pers_delta, verification_alg)
                else:
                    score = verification_opt(S_rec, R_rec, jaccard, pers_delta, verification_alg)            
            else:
                if matchAlg == 2:
                    if RLen < SLen:
                        score = verification_ps(R_rec, S_rec, jaccard, pers_delta)
                    else:
                        score = verification_ps(S_rec, R_rec, jaccard, pers_delta)
                else:
                    if RLen < SLen:
                        score = verification(R_rec, S_rec, jaccard, pers_delta, matchAlg)
                    else:
                        score = verification(S_rec, R_rec, jaccard, pers_delta, matchAlg)
            t2 = time()
            candver_time += t2-t1

            if matchAlg == 0:
                hg_scores.append(score)
            elif matchAlg == 1:
                #if RLen < SLen:
                #    print(R_rec,score)
                #else:
                #    print(S_rec,score)
                ld_scores.append(score)
            elif matchAlg == 2:
                ps_scores.append(score)

            if delta - score > 0.000000001:
                continue

            no_candres += 1
            output.append((R_id, S_id, score))

        ## Ending Candidate Refinement ##
        
    log['init_time'] = init_time
    log['candgen_time'] = candgen_time
    log['candref_time'] = candref_time
    log['candver_time'] = candver_time
    log['no_candgen'] = no_candgen
    log['no_candref'] = no_candref
    log['no_candver'] = no_candver
    log['no_candres'] = no_candres
        
        
    print('\nTime elapsed: Init: {:.2f}, Cand Gen: {:.2f}, Cand Ref: {:.2f}, Cand Ver: {:.2f} Total: {:.2f}'.format(init_time, candgen_time, candref_time, candver_time,init_time+candgen_time+candref_time+candver_time))
    print('Candidates Generated: {:,}, Refined: {:,}, Verified: {:,}, Survived: {:,}'.format(no_candgen, no_candref, no_candver, no_candres))
    numVerified.append(no_candver)
    


    if matchAlg == 0 and printSets != 0:
        print("HG Scores: ", len(hg_scores))
        with open(printSets,"a") as file:
            for i in verSet:
                #print(i)
                form_out = "$".join(
                    [";".join([",".join(map(str, sublist)) for sublist in lists]) for lists in i]
                )
                #form_out = ";".join([",".join(map(str, sublist)) for sublist in i[0]])
                #print(form_out)
                file.write(form_out + '\n')
            
    '''elif matchAlg == 1:
        pds = []
        for idx in range(len(hg_scores)):
            pds.append(percent_difference(hg_scores[idx],ld_scores[idx]))
        print("LD Average Percent Diff: ", average(pds))
        print("LD Max Percent Diff: ", max(pds), " Max Comp (HG vs. LD): ", hg_scores[max_index(pds)], " ", ld_scores[max_index(pds)])
        print("LD Min Percent Diff: ", min(pds), " Min Comp (HG vs. LD): ", hg_scores[min_index(pds)], " ", ld_scores[min_index(pds)])
        
    elif matchAlg == 2:
        pds = []
        for idx in range(len(hg_scores)):
            pds.append(percent_difference(hg_scores[idx],ps_scores[idx]))
        print("PS Average Percent Diff: ", average(pds))
        print("PS Max Percent Diff: ", max(pds), " Max Comp (HG vs. PS): ", hg_scores[max_index(pds)], " ", ps_scores[max_index(pds)])
        print("PS Min Percent Diff: ", min(pds), " Min Comp (HG vs. PS): ", hg_scores[min_index(pds)], " ", ps_scores[min_index(pds)])
    '''
    return output

class TokenJoin():
    
    #def tokenjoin(left_df, right_df, left_id, right_id, left_join, right_join, left_attr, right_attr, left_prefix='l_', right_prefix='r_'):
    def tokenjoin_self(self, df, id, join, attr=[], left_prefix='l_', right_prefix='r_', delta=0.7, jointFilter=False, posFilter=False, verification_alg=0, keepLog=False, matchAlg = 0, printSets = 0):
        total_time = time()
        log = {}
        collection = transform_collection(df[join].values)
        idx, lengths_list = build_index(collection)
        
        output = simjoin(collection['collection'], collection['collection'], delta, idx,
                         lengths_list, jointFilter, posFilter, verification_alg, log, matchAlg,printSets)
        
        output_df = pd.DataFrame(output, columns=[left_prefix+id, right_prefix+id, 'score'])
        for col in attr+[join, id]:
            #output_df[left_prefix+col] = df.set_index(id).loc[output_df[left_prefix+id], col].values
            output_df[left_prefix+col] = df.iloc[output_df[left_prefix+id]][col].values
        for col in attr+[join, id]:
            #output_df[right_prefix+col] = df.set_index(id).loc[output_df[right_prefix+id], col].values    
            output_df[right_prefix+col] = df.iloc[output_df[right_prefix+id]][col].values    
        
        total_time = time() - total_time
        log['total_time'] = total_time
        if keepLog:
            return output_df, log
        return output_df
    
    
    def tokenjoin_foreign(self, left_df, right_df, left_id, right_id, left_join, right_join, left_attr=[], right_attr=[], left_prefix='l_', right_prefix='r_', delta=0.7, jointFilter=False, posFilter=False, verification_alg=0, keepLog=False):
        total_time = time()
        log = {}
        right_collection = transform_collection(right_df[right_join].values)
        idx, lengths_list = build_index(right_collection)
        
        left_collection = transform_collection(left_df[left_join].values, right_collection['dictionary'])
        
        output = simjoin(left_collection['collection'], right_collection['collection'],
                         delta, idx, lengths_list, jointFilter, posFilter, verification_alg, log)
        
        output_df = pd.DataFrame(output, columns=[left_prefix+left_id, right_prefix+right_id, 'score'])
        for col in left_attr+[left_join, left_id]:
            #output_df[left_prefix+col] = left_df.set_index(left_id).loc[output_df[left_prefix+left_id], col].values
            output_df[left_prefix+col] = left_df.iloc[output_df[left_prefix+left_id]][col].values
        for col in right_attr+[right_join, right_id]:
            #output_df[right_prefix+col] = right_df.set_index(right_id).loc[output_df[right_prefix+right_id], col].values    
            output_df[right_prefix+col] = right_df.iloc[output_df[right_prefix+right_id]][col].values    
        
        total_time = time() - total_time
        log['total_time'] = total_time
        if keepLog:
            return output_df, log
        return output_df
    
    def tokenjoin_prepare(self, right_df, right_id, right_join, right_attr=[], right_prefix='r_'):
        self.right_collection = transform_collection(right_df[right_join].values)
        self.idx, self.lengths_list = build_index(self.right_collection)
        self.right_df = right_df
        self.right_id = right_id
        self.right_join = right_join
        self.right_attr = right_attr
        self.right_prefix = right_prefix
        
    
    def tokenjoin_query(self, left_df, left_id, left_join, left_attr=[], left_prefix='l_', delta=0.7, jointFilter=False, posFilter=False, verification_alg=0, keepLog=False):
        log = {}
        left_collection = transform_collection(left_df[left_join].values, self.right_collection['dictionary'])
        
        output = simjoin(left_collection['collection'], self.right_collection['collection'],
                         delta, self.idx, self.lengths_list, jointFilter, posFilter, verification_alg, log)
        
        output_df = pd.DataFrame(output, columns=[left_prefix+left_id, self.right_prefix+self.right_id, 'score'])
        for col in left_attr+[left_join, left_id]:
            output_df[left_prefix+col] = left_df.iloc[output_df[left_prefix+left_id]][col].values
        for col in self.right_attr+[self.right_join, self.right_id]:
            output_df[self.right_prefix+col] = self.right_df.iloc[output_df[self.right_prefix+self.right_id]][col].values    
        
        if keepLog:
            return output_df, log
        return output_df     
