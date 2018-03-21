#!/usr/bin/env python
from pymongo import MongoClient
from collections import defaultdict
from progress import console_progress

import numpy as np
from scipy.spatial.distance import jaccard, pdist

content_similarity_matrix = defaultdict(list)

def _main():
    client = MongoClient()
    all_content = {}
    all_content_features = {}
    print( 'loading all content...' )
    cursor = client.real_weddings_qa.photos.find( { 'terms.0' : { '$exists' : True } }, { '_id' : 1, 'terms' : 1 } )
    for record in cursor:
        all_content[record['_id']] = record['terms']
    print( 'done.' )

    print( 'loading all content features...' )
    cursor = client.recommendation.photo_features.find( {}, { '_id' : 1, 'features' : 1 } )
    for record in cursor:
        all_content_features[ record['_id'] ] = None
        features = record.get( 'features', None )
        if features:
            all_content_features[ record['_id'] ] = np.array(features)
    for record in cursor:
        all_content[record['_id']] = record['terms']
    print( 'done.' )

#    print( 'finding all terms...' )
#    all_terms = set([])
#    for terms in all_content.values():
#        all_terms = all_terms.union( set(terms) )
#    print( 'done.' )

    print( 'computing item/item similarities...' )
    similarities = defaultdict(list)
    last_marker = 0.0
    for total, (item1, vector1) in enumerate(all_content_features.iteritems()):
        (_,last_marker) = console_progress( total, len(all_content_features), print_count=True, next_marker=last_marker )
        for item2, vector2 in all_content_features.iteritems():
            if item1 == item2:
                continue

            similarity = 1.0-jaccard( vector1, vector2 )
            if similarity == 0:
                continue
            similarities[item1].append( ( similarity, item2 ) )
        
    print( 'done.' )
    return 0

if __name__ == "__main__":
    import sys
    sys.exit( _main() )
