import numpy as np
import itertools
import fastcluster
import hcluster
import networkx

from learning.utils.json_util2 import get_value
from collections import defaultdict, Counter
from networkx import Graph
from networkx.algorithms.components.connected import connected_components

import sys

def flatten( rec ):
    return ''.join(rec.values()).lower().replace(' ','')

def ngrams( sval, sizes ):
    return [(''.join(x),pos) for size in sizes for pos,x in enumerate(zip(*[sval[i:] for i in xrange(size)]))]

def build_index( recs, ngram_sizes ):
    index = defaultdict(list)
    for i, rec in enumerate( recs ):
        ngs = ngrams( flatten(rec), sizes=ngram_sizes )
        for ng,pos in ngs:
            index[ng].append((i,pos))
    return index
        
def prune_index( index, remove_frequent=True ):
    del_keys = [ key for key, value in index.iteritems() if len(set(item[0] for item in value)) <= 1 ]
    for key in del_keys:
        del(index[key])

    frequencies = Counter()
    for key, indices in index.iteritems():
        frequencies[key] = len(indices)

    if remove_frequent:
        mc = frequencies.most_common()[:1000]
        for key, c in mc:
            del(index[key])

    return index, frequencies

def score_records( record1, record2, index, frequencies, ngram_sizes=range(7,29,2), position_decay=20.0, frequency_decay=50.0 ):
    ng1 = dict( ngrams( flatten( record1 ), sizes=ngram_sizes ) )
    ng2 = dict( ngrams( flatten( record2 ), sizes=ngram_sizes ) )

    score = 0.0
    intersection = set(ng1.keys()).intersection(set(ng2.keys()))

    for key in intersection:
        key_frequency = float(frequencies[key])
        key_rarity = len(key) * np.exp( -key_frequency/frequency_decay )
        t = ng1[key] + ng2[key]
        score += (key_rarity * np.exp(-t/position_decay))

    return score

def cluster( recs,
             frequency_recs=None,
             frequencies=None,
             thresholds=None,
             position_decay=20.0,
             frequency_decay=50.0,
             ngram_sizes=range(7,29,2),
             keep_index=False,
             remove_frequent=True ):
    n = len(recs)
    index = build_index( recs, ngram_sizes )

    if frequency_recs is None:
        frequency_recs = recs
    if frequencies is None:
        frequency_index = build_index( frequency_recs, ngram_sizes )
        _, frequencies = prune_index( frequency_index, remove_frequent=remove_frequent )

    if n == 1:
        clusters = {}
        for threshold in thresholds:
            clusters[threshold] = []
        if keep_index:
            return (clusters,index,frequencies)
        else:
            return clusters

    coc = np.zeros( (int((n**2-n)/2.0),) )
    cd_index = lambda i,j: i*n + j - i*(i+1)/2 - i - 1
    def sf_index(i):
        b = 1 -2*n
        x = int( np.floor((-b - np.sqrt(b**2 - 8*i))/2) )
        y = int( i + x*(b + x + 2)/2 + 1 )
        return (x,y)

    for i, (key, indices) in enumerate(index.iteritems()):
        key_frequency = float(frequencies[key])
        key_rarity = len(key) * np.exp( -key_frequency/frequency_decay )
        for j, k in itertools.combinations(indices,2):
            t = j[1] + k[1]
            low,high = min(j[0],k[0]), max(j[0],k[0])
            ci = cd_index( low, high )
            coc[ci] += (key_rarity * np.exp(-t/position_decay))

    round_key = thresholds is None
    if thresholds is None:
        hist = np.histogram( coc, 50 )[1]
        low_threshold = hist[8]
        high_threshold = hist[12]
        thresholds = [ low_threshold, high_threshold ]

    clusters = {}
    for threshold in thresholds:
        g = Graph()
        matches = np.where( coc >= threshold )[0].tolist()
        for match in matches:
            edge = sf_index(match)
            g.add_edge( *edge )

        this_key = threshold
        if round_key:
            this_key = round(threshold,2)
        clusters[this_key] = [ i for i in connected_components(g) ]

    if keep_index:
        return (clusters,index,frequencies)
    else:
        return clusters
            
def hierarchical_cluster( clusters, threshold ):
    threshold = 1 - threshold
    score_dtype = [('pairs', 'i4', 2), ('score', 'f4', 1)]

#    lclassifier.predict_proba(distances)[0][1] > threshold

    dupe_graph = networkx.Graph()
    dupe_graph.add_weighted_edges_from((x[0], x[1], y) for (x, y) in clusters)

    dupe_sub_graphs = connected_components(dupe_graph)

    clustering = {}
    cluster_scores = {}
    cluster_id = 0
    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 2:
            pair_gen = ((sorted(x[0:2]), x[2]['weight'])
                        for x in dupe_graph.edges_iter(sub_graph, data=True))

            pairs = np.fromiter(pair_gen, dtype=score_dtype)
            pairlist = list(pairs)

            (i_to_id, condensed_distances) = condensedDistance(pairs)
            linkage = fastcluster.linkage(condensed_distances,
                                          method='centroid',
                                          preserve_input=False)

            partition = hcluster.fcluster(linkage,
                                          threshold,
                                          criterion='distance')

            for (i, sub_cluster_id) in enumerate(partition):
                clustering.setdefault(cluster_id + sub_cluster_id, []).append(i_to_id[i])

            cluster_id += max(partition)
        elif len(sub_graph) == 2:
            clustering[cluster_id] = sub_graph
            cluster_id += 1

    clusters = [set(l) for l in clustering.values() if len(l) >= 2 ]
    return (clusters,cluster_scores)
    
