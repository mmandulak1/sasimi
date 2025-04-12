import networkx as nx
import editdistance
import heapq

'''
def jaccard(r, s):
    rr = set(r)
    ss = set(s)
    return len(rr & ss) / len(rr | ss)
'''

graphN = [[],[],[],[]]
graphM = [[],[],[],[]]
countMatchInst = [0,0,0,0,0,0,0,0,0]




def neds(r, s):
    return 1-editdistance.eval(r, s) / max((len(r), len(s)))

def jaccard(r, s):
    olap = pr = ps = 0
    maxr = len(r) - pr + olap;
    maxs = len(s) - ps + olap;

    while maxr > olap and maxs > olap :
        if r[pr] == s[ps] :
            pr += 1
            ps += 1
            olap += 1
        elif r[pr] < s[ps]: 
            pr += 1
            maxr -= 1
        else:
            ps += 1
            maxs -= 1

    return olap / (len(r) + len(s) - olap)


def deduplicate(r, s):
    olap = pr = ps = 0
    maxr = len(r) - pr + olap;
    maxs = len(s) - ps + olap;
    r_inds = []
    s_inds = []

    while maxr > olap and maxs > olap :
        #print("COMP CHECK",r[pr])
        if r[pr] == s[ps] :
            r_inds.append(pr)
            s_inds.append(ps)            
            pr += 1
            ps += 1
            olap += 1
        elif r[pr] < s[ps]: 
            pr += 1
            maxr -= 1
        else:
            ps += 1
            maxs -= 1

    return olap, set(r_inds), set(s_inds)



def build_csr(edgeList,n):
    sortedEdges = sorted(edgeList)
    indices = [0] * (n+1)
    edges = []
    weights = []
    for idx, e in enumerate(sortedEdges):
        u = e[0]
        v = e[1]
        w = e[-1]
        indices[u+1] += 1
        edges.append(v)
        weights.append(w)
    for i in range(1,n+1):
        indices[i] = indices[i-1] + indices[i]
    return indices, edges, weights

#DO THE COMPONENT WISE TIMING
def verification(R_record, S_record, phi, pers_delta,alg):
    
    # Start Element deduplication
    orRLen = len(R_record)
    orSLen = len(S_record)
    add, r_inds, s_inds = deduplicate(R_record, S_record)
    #print("DEDUP ADD VAL: ",add)
    R_record = [r for no, r in enumerate(R_record) if no not in r_inds]
    S_record = [s for no, s in enumerate(S_record) if no not in s_inds]
    if (len(R_record)) == 0:
        score = add / (orRLen + orSLen - add)
        countMatchInst[alg+3] += 1
        return score        
    # End Element deduplication
    '''
    add = 0
    '''
    
    UB = orRLen
    edges = []
    for nor, r in enumerate(R_record):
        max_NN = 0
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            #print("JSCORE:",score)
            max_NN = max((max_NN, score))
            edges.append((f'r_{nor}', f's_{nos}', score))
            
        UB -= 1 - max_NN
        if pers_delta - UB > 0.0000001:
            countMatchInst[alg+3] += 1
            return UB / (orRLen + orSLen - UB)
    #print(edges)

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    graphN[alg].append(len(G.nodes))
    graphM[alg].append(len(G.edges))
    countMatchInst[alg] += 1
    #print(G)
    #print(nx.max_weight_matching(G))
    if alg == 0:
        matching = add
        for e in nx.max_weight_matching(G):
            matching += G.edges[e]['weight']
    elif alg == 1:
        matching = add
        matching += ldMatching(G)
    elif alg == 2:
        matching = add 
        matching += psMatching(G)
    elif alg == 3:
        matching = add
        matching += greedyMatching(G)




    score = matching / (orRLen + orSLen - matching)
    return score 



def verification_ld(R_record, S_record, phi, pers_delta):
    
    # Start Element deduplication
    orRLen = len(R_record)
    orSLen = len(S_record)
    add, r_inds, s_inds = deduplicate(R_record, S_record)
    #print("DEDUP ADD VAL: ",add)
    R_record = [r for no, r in enumerate(R_record) if no not in r_inds]
    S_record = [s for no, s in enumerate(S_record) if no not in s_inds]
    if (len(R_record)) == 0:
        score = add / (orRLen + orSLen - add)
        return score        
    # End Element deduplication
    '''
    add = 0
    '''
    n = len(R_record) + len(S_record)
    UB = orRLen
    edges = [[] for _ in range(n)]
    for nor, r in enumerate(R_record):
        max_NN = 0
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            max_NN = max((max_NN, score))
            edges[nor].append((len(R_record) + nos, score))
            edges[len(R_record) + nos].append((nor, score))
            
        UB -= 1 - max_NN
        if pers_delta - UB > 0.0000001:
            return UB / (orRLen + orSLen - UB)



    matchScore = add
    matching = list(range(0,n))
    numUpdates = 1
    totalWeight = 0
    #SET POINTERS
    while numUpdates != 0:
        pointers = list(range(0,n))
        pointerWeights = [0] * n
        numUpdates = 0
        for u in range(0,n):
            if matching[u] != u:
                continue

            deg = len(edges[u])

            bestN = u
            bestW = float('-inf')
            for v in range(0,deg):
                cW = edges[u][v][-1]
                if cW > bestW:
                    bestN = edges[u][v][0]
                    bestW = cW
            pointers[u] = bestN
            pointerWeights[u] = bestW
        #MUTUAL CHECK
        for u in range(0,n):
            if pointers[u] == u or matching[u] != u:
                continue
            if u == pointers[pointers[u]]:
                matching[u] = pointers[u]
                matching[pointers[u]] = u
                totalWeight += pointerWeights[u]
                numUpdates+=2
    #totalWeight /= 2
    matchScore += totalWeight

    score = matchScore / (orRLen + orSLen - matchScore)
    return score 




def verification_gd(R_record, S_record, phi, pers_delta):
    
    # Start Element deduplication
    orRLen = len(R_record)
    orSLen = len(S_record)
    add, r_inds, s_inds = deduplicate(R_record, S_record)
    #print("DEDUP ADD VAL: ",add)
    R_record = [r for no, r in enumerate(R_record) if no not in r_inds]
    S_record = [s for no, s in enumerate(S_record) if no not in s_inds]
    if (len(R_record)) == 0:
        score = add / (orRLen + orSLen - add)
        return score        
    # End Element deduplication
    '''
    add = 0
    '''
    
    UB = orRLen
    edges = []
    for nor, r in enumerate(R_record):
        max_NN = 0
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            max_NN = max((max_NN, score))
            edges.append((nor, len(R_record) + nos, score))
            
        UB -= 1 - max_NN
        if pers_delta - UB > 0.0000001:
            return UB / (orRLen + orSLen - UB)

    matchScore = add
    matching = list(range(0,len(R_record) + len(S_record)))
    se = sorted(edges, key=lambda x: x[-1],reverse=True)
    totalWeight = 0
    for e in se:
        u = e[0]
        v = e[1]
        if matching[u] == u and matching[v] == v:
            matching[u] = v
            matching[v] = u 
            totalWeight += e[-1]
    matchScore += totalWeight


    score = matchScore / (orRLen + orSLen - matchScore)
    return score 




# Indexing = len(R) + currS 
def verification_ps(R_record, S_record, phi, pers_delta):
    # Start Element deduplication
    orRLen = len(R_record)
    orSLen = len(S_record)
    add, r_inds, s_inds = deduplicate(R_record, S_record)
    #print("DEDUP ADD VAL: ",add)
    R_record = [r for no, r in enumerate(R_record) if no not in r_inds]
    S_record = [s for no, s in enumerate(S_record) if no not in s_inds]
    if (len(R_record)) == 0:
        score = add / (orRLen + orSLen - add)
        return score        
    # End Element deduplication
    '''
    add = 0
    '''
    UB = orRLen
    matchScore = add
    mWeight = 0
    eps = 0.01
    n = len(R_record) + len(S_record)
    matching = list(range(0,n))
    S = []
    dual = [0] * (len(R_record) + len(S_record))

    for nor, r in enumerate(R_record):
        max_NN = 0
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            max_NN = max((max_NN, score))
            u = nor
            v = len(R_record) + nos
            w_e = score
            e = (u,v,w_e)
            #edge = (nor, len(R) + nos, score)
            if w_e > (1 + eps)*(dual[u] + dual[v]):
                w_p = w_e - (dual[u] + dual[v])
                dual[u] = dual[u] + w_p/2
                dual[v] = dual[v] + w_p/2
                S.append(e)
        UB -= 1 - max_NN
        if pers_delta - UB > 0.0000001:
            return UB / (orRLen + orSLen - UB)
    while len(S) != 0:
        e = S[-1]
        u = e[0]
        v = e[1]
        S.pop()
        if matching[u] == u and matching[v] == v:
            matching[u] = v
            matching[v] = u
            mWeight += e[2]
    #mWeight/=2
    matchScore += mWeight
    return matchScore/(orRLen + orSLen - matchScore)



def verification_opt(R_record, S_record, phi, pers_delta, alg):
    
    # Start Element deduplication
    orRLen = len(R_record)
    orSLen = len(S_record)
    
    add, r_inds, s_inds = deduplicate(R_record, S_record)
    
    if alg == 2 and add - pers_delta > 0.0000001:
        return add / (orRLen + orSLen - add);

    R_record = [r for no, r in enumerate(R_record) if no not in r_inds]
    S_record = [s for no, s in enumerate(S_record) if no not in s_inds]
    
    if (len(R_record)) == 0:
        score = add / (orRLen + orSLen - add)
        return score        
    # End Element deduplication
    
    
    RLen = len(R_record)
    SLen = len(S_record)
    
    square = False
    if RLen == SLen: # square matrix
        colMin = [1.0 for _ in range(SLen)]
        square = True
    else:
        colMin = [0.0 for _ in range(SLen)]
        RLen = SLen # square matrix
        square = False
    
    UB = add + RLen
    nnEdges = [0 for _ in range(RLen)]
    hits2 = [[0 for _ in range(SLen)] for _ in range(RLen)]
    if alg == 2:
        pq = [[] for _ in range(RLen)]
    for nor, r in enumerate(R_record):
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            nnEdges[nor] = max((nnEdges[nor], score))
            hits2[nor][nos] = score
    
            if alg == 2:
                heapq.heappush(pq[nor], (-score, nos))  #descending order to use pop
            
        UB -= 1 - nnEdges[nor]
        if pers_delta - UB > 0.0000001:
            return UB / (orRLen + orSLen - UB)    
    
    
    pi = [[0 for _ in range(SLen)] for _ in range(RLen)]
    
    # initialize: inverse and subtract row minima
    for r in range(RLen):
        for s in range(SLen):
            # print(len(pi), len(hits2), len(nnEdges), "\t", r, s, "\t", len(pi[r]), len(hits2[r]))
            pi[r][s] = nnEdges[r] - hits2[r][s]
            if square:
                colMin[s] = min((colMin[s], pi[r][s]))
    
    # initialize: subtract column minima
    if square:
        for s in range(SLen):
            if colMin[s] == 0: #there will be no change in this column
                continue
            for r in range(RLen):
                pi[r][s] = pi[r][s] - colMin[s]
    
    sumMatching = 0
    if alg == 0:
    	sumMatching = findMatching(pi, add, hits2)
    elif alg == 1:
     	sumMatching = findMatchingUB(pi, add, hits2, nnEdges, pers_delta)
    elif alg == 2:
     	sumMatching = findMatchingULB(pi, add, hits2, nnEdges, pers_delta, pq)
    score = sumMatching / (orRLen + orSLen - sumMatching)
    return score
    
def get_lower_bound(R_record, S_record, phi):
    
    RLen = len(R_record)
    SLen = len(S_record)
    
    pq = []
    for nor, r in enumerate(R_record):
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            heapq.heappush(pq, (-score, nor, nos))  #descending order to use pop
            
            
    sumMatching = 0
    rVertices = set()
    sVertices = set()
    while len(rVertices) != RLen :
        (score, nor, nos) = heapq.heappop(pq)
        if nor in rVertices: # matched
            continue
        if nos in sVertices: # matched
            continue

        rVertices.add(nor) # mark as matched
        sVertices.add(nos) # mark as matched
        sumMatching += -score;

    return sumMatching / (RLen + SLen - sumMatching)


def get_upper_bound(R_record, S_record, phi):
    
    sumMatching = 0
    for nor, r in enumerate(R_record):
        max_r = -1
        for nos, s in enumerate(S_record):
            score = phi(r, s)
            max_r = max(score, max_r)
        sumMatching += max_r
    return sumMatching / (len(R_record) + len(S_record) - sumMatching)


def greedyMatching(G):
    G = nx.convert_node_labels_to_integers(G)
    n = len(G.nodes)
    m = len(G.edges)
    matching = list(range(0,n))
    mweight = 0
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'],reverse=True)
    for e in sorted_edges:
        u = e[0]
        v = e[1]
        eweight = e[2]['weight']
        #print(u,v,eweight)
        if matching[u] == u and matching[v] == v:
            matching[u] = v
            matching[v] = u
            mweight += eweight
    return mweight

    



# For nodes e.g. r_0 or r_10, idx = _X (0/10)
# For nodes e.g. s_0 or s_10, idx = (N/2)+1 + _X (if N = 24, 13/23)
# nvm i'll just use networkx labeling since graph is tossed after

def ldMatching(G):


    G = nx.convert_node_labels_to_integers(G)
    n = len(G.nodes)
    m = len(G.edges)
    matching = list(range(0,n))
    numUpdates = 1
    #SET POINTERS
    while numUpdates != 0:
        pointers = list(range(0,n))
        numUpdates = 0
        for u in G.nodes:
            if matching[u] != u:
                continue
            bestN = u
            bestW = float('-inf')
            for v in G[u]:
                #print("EDGE",u,v,G[u][v]['weight'])
                cW = G[u][v]['weight']
                if cW > bestW:
                    bestN = v
                    bestW = cW
            pointers[u] = bestN
        #MUTUAL CHECK
        for u in G.nodes:
            if pointers[u] == u:
                continue
            if u == pointers[pointers[u]]:
                matching[u] = pointers[u]
                matching[pointers[u]] = u
                numUpdates+=2

    mWeight = 0
    nxWeight = 0
    for u in G.nodes:
        if matching[u] != u:
            mWeight += G[u][matching[u]]['weight']
    mWeight /= 2

    return mWeight


def psMatching(G):
    eps = 0.01
    G = nx.convert_node_labels_to_integers(G)
    n = len(G.nodes)
    m = len(G.edges)
    matching = list(range(0,n))
    S = []
    dual = [0] * n
    for e in G.edges:
        #print(e,e[0])
        u = e[0]
        v = e[1]
        w_e = G[u][v]['weight']
        if w_e > (1 + eps)*(dual[u] + dual[v]):
            w_p = w_e - (dual[u] + dual[v])
            dual[u] = dual[u] + w_p/2
            dual[v] = dual[v] + w_p/2
            S.append(e)
    while len(S) != 0:
        e = S[-1]
        u = e[0]
        v = e[1]
        S.pop()
        if matching[u] == u and matching[v] == v:
            matching[u] = v
            matching[v] = u

    mWeight = 0
    nxWeight = 0
    for u in G.nodes:
        if matching[u] != u:
            mWeight += G[u][matching[u]]['weight']
    
    
    mWeight /= 2

    return mWeight




    
def findMatching(pi, add, hits2):
    # Set<Edge> M = new HashSet<Edge>();
    M = set()
    rLen = len(pi)
    sLen = len(pi) # square matrix
    
    g = Graph(pi, rLen, sLen)
    
    ROffset = 0
    SOffset = rLen + ROffset
    sumMatching = 0
    
    while len(M) != rLen :
     	# find augmenting path
        pred = [0 for _ in range(g.V)]
        visited = [False for _ in range(g.V)]
    
        finalNode = g.BFS(g.src, g.dest, pred, visited)
    
        if finalNode == g.dest: # successful augmenting path
            crawl = pred[finalNode]
            P = set()
            while pred[crawl] != g.src:
                e = tuple(sorted((crawl, pred[crawl])))
                P.add(e)
                crawl = pred[crawl]
    
    		# search succesfful -> augment Matching
            M2 = set()
            for e in M:
                if e not in P:
                    M2.add(e)

            for e in P:
                if e not in M:
                    M2.add(e)

            for e in M:
                g.revertEdge(e[0], e[1]); # remove reversal of edges
            M = M2;
            sumMatching = add
            for e in M:
                g.updateEdge(e[0], e[1]);
                r = e[0] - ROffset
                s = e[1] - SOffset
                sumMatching += hits2[r][s]
    
        else: # landed in left Partite, we have a collision, variables need adjustment
    
    		# find delta
            delta = 1.1
    
    		# search delta (min pi) in Marked R -> Unmarked S
            for r in range(rLen):
                if visited[r]: # if R is marked
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            delta = min((delta, pi[r][s]))
    
    		# reduce Marked R -> Unmarked S, to enable more edges
            for r in range(rLen):
                if visited[r]:
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            pi[r][s] = pi[r][s] - delta
                            if pi[r][s] == 0.0:
                                g.addEdge(r, s)
    
    		# increase unmarked R -> marked S, to discourage colliding edges
            for r in range(rLen):
                if not visited[r]: # if R is unmarked
                    for s in range(sLen):
                        if visited[s + SOffset]: # if S is marked
                            pi[r][s] = pi[r][s] + delta
                            if pi[r][s] != 0.0:
                                g.removeEdge(r, s)
    
    return sumMatching;    

def findMatchingUB(pi, add, hits2, nnEdges, pers_delta):
    # Set<Edge> M = new HashSet<Edge>();
    M = set()
    rLen = len(pi)
    sLen = len(pi) # square matrix
    
    g = Graph(pi, rLen, sLen)
    
    ROffset = 0
    SOffset = rLen + ROffset
    sumMatching = 0
    
    while len(M) != rLen :
     	# find augmenting path
        pred = [0 for _ in range(g.V)]
        visited = [False for _ in range(g.V)]
    
        finalNode = g.BFS(g.src, g.dest, pred, visited)
    
        if finalNode == g.dest: # successful augmenting path
            crawl = pred[finalNode]
            P = set()
            while pred[crawl] != g.src:
                e = tuple(sorted((crawl, pred[crawl])))
                P.add(e)
                crawl = pred[crawl]
    
    		# search succesfful -> augment Matching
            M2 = set()
            for e in M:
                if e not in P:
                    M2.add(e)

            for e in P:
                if e not in M:
                    M2.add(e)

            for e in M:
                g.revertEdge(e[0], e[1]); # remove reversal of edges
            M = M2;
            sumMatching = add
            for e in M:
                g.updateEdge(e[0], e[1]);
                r = e[0] - ROffset
                s = e[1] - SOffset
                sumMatching += hits2[r][s]
                nnEdges[r] = hits2[r][s];

            UB = sumMatching
            for r in  g.adj[g.src]:
                UB += nnEdges[r]

            if pers_delta - UB > 0.0000001:
                return UB;
    
        else: # landed in left Partite, we have a collision, variables need adjustment
    
    		# find delta
            delta = 1.1
    
    		# search delta (min pi) in Marked R -> Unmarked S
            for r in range(rLen):
                if visited[r]: # if R is marked
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            delta = min((delta, pi[r][s]))
    
    		# reduce Marked R -> Unmarked S, to enable more edges
            for r in range(rLen):
                if visited[r]:
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            pi[r][s] = pi[r][s] - delta
                            if pi[r][s] == 0.0:
                                g.addEdge(r, s)
    
    		# increase unmarked R -> marked S, to discourage colliding edges
            for r in range(rLen):
                if not visited[r]: # if R is unmarked
                    for s in range(sLen):
                        if visited[s + SOffset]: # if S is marked
                            pi[r][s] = pi[r][s] + delta
                            if pi[r][s] != 0.0:
                                g.removeEdge(r, s)
    
    return sumMatching;  


def findMatchingULB(pi, add, hits2, nnEdges, pers_delta, pq):
    # Set<Edge> M = new HashSet<Edge>();
    M = set()
    rLen = len(pi)
    sLen = len(pi) # square matrix
    
    g = Graph(pi, rLen, sLen)
    
    ROffset = 0
    SOffset = rLen + ROffset
    sumMatching = 0
    
    LBEdges = [-1 for _ in range(rLen)] # store assignments for LB
    
    while len(M) != rLen :
     	# find augmenting path
        pred = [0 for _ in range(g.V)]
        visited = [False for _ in range(g.V)]
    
        finalNode = g.BFS(g.src, g.dest, pred, visited)
    
        if finalNode == g.dest: # successful augmenting path
            crawl = pred[finalNode]
            P = set()
            while pred[crawl] != g.src:
                e = tuple(sorted((crawl, pred[crawl])))
                P.add(e)
                crawl = pred[crawl]
    
    		# search succesfful -> augment Matching
            M2 = set()
            for e in M:
                if e not in P:
                    M2.add(e)

            for e in P:
                if e not in M:
                    M2.add(e)

            for e in M:
                g.revertEdge(e[0], e[1]); # remove reversal of edges
            M = M2;
            sumMatching = add
            for e in M:
                g.updateEdge(e[0], e[1]);
                r = e[0] - ROffset
                s = e[1] - SOffset
                sumMatching += hits2[r][s]
                nnEdges[r] = hits2[r][s];

            UB = sumMatching
            for r in g.adj[g.src]:
                UB += nnEdges[r]

            if pers_delta - UB > 0.0000001:
                return UB
                
            LB = sumMatching;
            for r in g.adj[g.src]:
            	# if pq[r] == None:  dummy left node
            	# 	continue;
            
                if len(g.adj[SOffset + LBEdges[r]]) == 0: # this LB is still free
                    continue

                while len(pq[r]) != 0 and len(g.adj[SOffset + heapq.nsmallest(1, pq[r])[0][1]]) > 0 : # iterate, until first best unmatched
                    heapq.heappop(pq[r])

                if len(pq[r]) != 0:
                    s_elem = heapq.nsmallest(1, pq[r])[0]
                    LB += -(s_elem[0])
                    LBEdges[r] = s_elem[1]
                
            if LB - pers_delta > 0.0000001:
                return LB
    
        else: # landed in left Partite, we have a collision, variables need adjustment
    
    		# find delta
            delta = 1.1
    
    		# search delta (min pi) in Marked R -> Unmarked S
            for r in range(rLen):
                if visited[r]: # if R is marked
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            delta = min((delta, pi[r][s]))
    
    		# reduce Marked R -> Unmarked S, to enable more edges
            for r in range(rLen):
                if visited[r]:
                    for s in range(sLen):
                        if not visited[s + SOffset]: # if S is unmarked
                            pi[r][s] = pi[r][s] - delta
                            if pi[r][s] == 0.0:
                                g.addEdge(r, s)
    
    		# increase unmarked R -> marked S, to discourage colliding edges
            for r in range(rLen):
                if not visited[r]: # if R is unmarked
                    for s in range(sLen):
                        if visited[s + SOffset]: # if S is marked
                            pi[r][s] = pi[r][s] + delta
                            if pi[r][s] != 0.0:
                                g.removeEdge(r, s)
    
    return sumMatching;  

class Graph:

	#int V; // No. of vertices
	#TIntSet adj[]; // Adjacency Lists
	#int ROffset, SOffset, totalOffset;
	#int src, dest;

    def __init__(self, e, rLen, sLen):
        self.ROffset = 0
        self.SOffset = rLen + self.ROffset
        totalOffset = sLen + self.SOffset

        self.V = rLen + sLen + 2; # R + S + (T, S)
        self.adj = [set() for _ in range(self.V)]

        self.dest = totalOffset;
        self.src = totalOffset + 1;

        for r in range(rLen):
            self.adj[self.src].add(r + self.ROffset)
            for s in range(sLen):
                if e[r][s] == 0.0:
                    self.adj[r + self.ROffset].add(s + self.SOffset)
                if len(self.adj[s + self.SOffset]) == 0:
                    self.adj[s + self.SOffset].add(self.dest)


    def addEdge(self, r, s):
        self.adj[r + self.ROffset].add(s + self.SOffset)

    def removeEdge(self, r, s):
        self.adj[r + self.ROffset].discard(s + self.SOffset)
        self.adj[s + self.SOffset].discard(r + self.ROffset) # could be in a matching already

    def updateEdge(self, esrc, edest):
        self.adj[self.src].discard(esrc)
        self.adj[esrc].discard(edest)
        self.adj[edest].add(esrc)
        self.adj[edest].discard(self.dest)

    def revertEdge(self, esrc, edest):
        self.adj[self.src].add(esrc)
        self.adj[esrc].add(edest)
        self.adj[edest].discard(esrc)
        self.adj[edest].add(self.dest)

	#a modified version of BFS that stores predecessor of each vertex in array pred
    def BFS(self, src, dest, pred, visited):
        queue = []

		# initially all vertices are unvisited so v[i] for all i is false and as no path is yet constructed
        for i in range(self.V):
            pred[i] = -1

		# now source is first to be visited and distance from source to itself should be 0
        visited[src] = True
        queue.append(src)

        u, v = -1, -1
		# bfs Algorithm
        while len(queue) != 0:
            u = queue.pop(0)

            for v in self.adj[u]:
                if visited[v] == False:
                    visited[v] = True
                    pred[v] = u
                    queue.append(v)

					# stopping condition (when we find our destination)
                    if v == dest:
                        return v

        if u < self.SOffset: # landed in Left Partite, we have a collision
            return u
        return -1


