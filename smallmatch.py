#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
smallmatch.py:  smallmatch application
'''
import argparse
import os
import ConfigParser
import itertools
import csv
import pickle
import json
import re
from bson import json_util
from random import shuffle
import random

import cluster
from cluster import hierarchical_cluster
import fastcluster
import hcluster
import networkx
from networkx.algorithms.components.connected import connected_components

from string import maketrans
from pymongo import MongoClient

from learning.utils.cpmongo import parse_mongo_spec
from learning.utils.progress import console_progress
from learning.utils.json_util2 import get_value, get_raw_value, mongo_sanitize
from learning.classifier.feature_vector import TermFeatureVector
from learning.classifier.classification import PerceptronClassifier, LogisticRegressionClassifier
from learning.classifier.term_classification import TermClassifier

from dedupe.distance.affinegap import normalizedAffineGapDistance

from smallmatch_etl import pre_process, translate_state
#from smallmatch_compare import *

from collections import OrderedDict, Counter
from unidecode import unidecode

from lr import lr
import numpy as np

class HBLogisticRegression:
    def __init__( self, labels, examples ):
        self.weights, self.bias = lr( labels, examples, 0.01, 500, 5000, 0.01 )
    def predict( self, distances ):
        logit = self.bias + np.dot( self.weights, distances )
        return 1.0/(1.0+np.exp(-logit))

#tfidf = None
#def tfidf_compare_name( s1, s2 ):
#    global tfidf
#    return 1.0-tfidf.compare(s1,s2)

def compare_postcode( str1, str2 ):
    if not str1 or not str2:
        return nan
    return denormalize_score( jaccard.jaccard_index( str1, str2 ) )

def compare_fax( str1, str2 ):
    if not str1 or not str2:
        return nan
    return compare_tel( str1, str2 )

def compare_tel( str1, str2 ):
    if not str1 or not str2:
        return nan
    if type(str1) == str:
        str1 = str1.translate( None, '-() ' )
        str2 = str2.translate( None, '-() ' )
    elif type(str1) == unicode:
        transtable = dict.fromkeys( [ ord('-'), ord('('), ord(')'), ord(' ') ] )
        str1 = str1.translate( transtable )
        str2 = str2.translate( transtable )
    else:
        return nan

    if not str1 or not str2:
        return nan

    if str1[0] == '1':
        str1 = str1[1:]
    if str2[0] == '1':
        str2 = str2[1:]

    if len(str1) != len(str2):
        if len(str1) > len(str2):
            str1 = str1[len(str1)-len(str2):]
        else:
            str2 = str2[len(str2)-len(str1):]

    return denormalize_score( int(str1[-7:] == str2[-7:]) )


reverse_fields = [ 'address', 'name', 'postcode' ]
#model_template = OrderedDict( {
#    'address' : { 'compare' : normalizedAffineGapDistance },
#    'reverse_address' : { 'compare' : normalizedAffineGapDistance },
#
#    'emailname' : { 'compare' : normalizedAffineGapDistance },
#    'emaildomain' : { 'compare' : normalizedAffineGapDistance },
#
#    'fax' : { 'compare' : compare_fax },
#    'locality' : { 'compare' : normalizedAffineGapDistance },
#
#    'name' : { 'compare' : normalizedAffineGapDistance },
#    'reverse_name' : { 'compare' : normalizedAffineGapDistance },
#
#    'postcode' : { 'compare' : compare_postcode },
#    'reverse_postcode' : { 'compare' : compare_postcode },
#
#    'region' : { 'compare' : normalizedAffineGapDistance },
#    'tel' : { 'compare' : compare_tel },
#    'website' : { 'compare' : normalizedAffineGapDistance },
#} )
model_template = OrderedDict( {
    'address_number' : { 'compare' : normalizedAffineGapDistance },
    'address' : { 'compare' : normalizedAffineGapDistance },
    'reverse_address' : { 'compare' : normalizedAffineGapDistance },

    'emailname' : { 'compare' : normalizedAffineGapDistance },
    'emaildomain' : { 'compare' : normalizedAffineGapDistance },

    'fax' : { 'compare' : compare_fax },
    'locality' : { 'compare' : normalizedAffineGapDistance },

#    'name' : { 'compare' : tfidf_compare_name },
    'name' : { 'compare' : normalizedAffineGapDistance },
    'reverse_name' : { 'compare' : normalizedAffineGapDistance },

    'postcode' : { 'compare' : compare_postcode },
    'reverse_postcode' : { 'compare' : compare_postcode },

    'region' : { 'compare' : normalizedAffineGapDistance },
    'tel' : { 'compare' : compare_tel },
    'website' : { 'compare' : normalizedAffineGapDistance },
} )

addr_reg = None
def encapsulate( record ):
    global addr_reg
    if not addr_reg:
        addr_reg = re.compile( r'^(\d+)\b' )

    records = []
    mrecord = {}
    mrecord['name'] = record.get( 'name', '' )
    mrecord['website'] = record.get( 'website', '' )
    mrecord['xoid'] = record.get('source', {}).get( 'id' )
    mrecord['applicationName'] = record.get( 'source', {}).get('applicationName')
    location = record.get('locations', [{}])[0]
    if location:
        address = location.get('address1','') + ' ' + location.get('address2','')
        digits = addr_reg.match(address)
        if digits:
            mrecord['address_number'] = digits.groups()[0]
            mrecord['address'] = address[len(mrecord['address_number']):]
        else:
            mrecord['address_number'] = ''
            mrecord['address'] = address

        mrecord['country'] = location.get('countryCode','')
        email = location.get('email','').split('@')
        if len(email) >= 2:
            mrecord['emailname'] = email[0]
            mrecord['emaildomain'] = email[1]
        else:
            mrecord['emailname'] = email[0]
            mrecord['emaildomain'] = ''
        mrecord['fax'] = location.get('fax','')
        mrecord['locality'] = location.get('city','')
        mrecord['postcode'] = location.get('postalCode','')
        mrecord['region'] = translate_state(location.get('state',''))
        mrecord['tel'] = location.get('phone','')

        reversed = { "reverse_" + k : v[::-1] for (k, v) in mrecord.iteritems() if k in reverse_fields }
        mrecord.update( reversed )

        cleaned = { pre_process(k) : pre_process(v) for (k, v) in mrecord.iteritems() }
        mrecord = cleaned

    difference = set(model_template.keys()) - set(mrecord.keys())
    if difference:
        for key in difference:
            mrecord[key] = ''

    return mrecord

def encapsulate_recs( input ):
    return [ encapsulate(rec) for rec in input ]


def url_etl( url ):
    if url.startswith('http://'):
        url = url.lstrip('http://')
    if url.startswith('www.'):
        url = url.lstrip('www.')

    index = url.find( '.' )
    if index != -1:
        url = url[:index]

    return url

def record_etl( record ):
    ordered_fields = [ 'name', 
                       'emailname', 
                       'address', 
                       'tel', 
                       'emaildomain', 
                       'website', 
                       'locality', 
                       'postcode', 
                       'region', 
                       'fax' ]

    location = record.get('locations', [{}])[0]
    location_rec = {}
    if location:
        email = location.get('email','').split('@')
        if len(email) >= 2:
            location_rec['emailname'] = email[0]
            location_rec['emaildomain'] = email[1].rstrip('.com')
        else:
            location_rec['emailname'] = email[0]
            location_rec['emaildomain'] = ''

        location_rec['tel'] = location.get('phone','').replace('-','').replace('(','').replace(')','')
        location_rec['address'] = location.get('address1','') + ' ' + location.get('address2','')
        location_rec['postcode'] = location.get('postalCode','')
        location_rec['locality'] = location.get('city','')
        location_rec['region'] = translate_state(location.get('state',''))
#        mrecord['country'] = location.get('countryCode','')
#        location_rec['country'] = ''
        location_rec['fax'] = location.get('fax','').replace('-','').replace('(','').replace(')','')

    mrecord = OrderedDict()

    mrecord['name'] = record.get( 'name', '' )
    if location_rec:
        mrecord['emailname'] = location_rec['emailname']
        mrecord['tel'] = location_rec['tel']
        mrecord['address'] = location_rec['address']
        mrecord['emaildomain'] = location_rec['emaildomain']
        mrecord['website'] = url_etl(record.get( 'website', '' ))
        mrecord['locality'] = location_rec['locality']
        mrecord['postcode'] = location_rec['postcode']
        mrecord['region'] = location_rec['region']
        mrecord['fax'] = location_rec['fax']
    else:
        mrecord['website'] = url_etl(record.get( 'website', '' ))

    return mrecord

def clean_recs( input ):
    return [ { k : ''.join(v.split()) for k,v in record_etl(rec).iteritems() } for rec in input ]

def model( records ):
    record1, record2 = records[0], records[1]
    distances = np.fromiter( ( logic['compare'](record1[field],record2[field]) for field,logic in model_template.iteritems() ), dtype=np.float64 )
    missing = np.isnan(distances)
    distances[missing] = -1
    length_deltas = np.fromiter( ( np.fabs(len(record1[field]))-np.fabs(len(record2[field])) 
                                   if len(record1[field]) and len(record2[field]) 
                                   else -1
                                   for field,_ in model_template.iteritems() ), dtype=np.float64 )
    distances = np.concatenate( (distances, length_deltas), axis=1 )
    return distances

def condensedDistance( dupes ):
    candidate_set = np.unique(dupes['pairs'])
    i_to_id = dict(enumerate(candidate_set))
    ids = candidate_set.searchsorted(dupes['pairs'])
    row = ids[:, 0]
    col = ids[:, 1]
    N = len(np.union1d(row, col))

    matrix_length = N * (N - 1) / 2
    row_step = (N - row) * (N - row - 1) / 2

    index = matrix_length - row_step + col - row - 1

    condensed_distances = np.ones(matrix_length, 'f4')
    condensed_distances[index] = 1 - dupes['score']

    return (i_to_id, condensed_distances)

def prune_clusters( raw_inputs, clusters, classifier ):
    result = []
    for cluster in clusters:
        match_count = Counter()
        for index1, index2 in itertools.combinations( cluster, 2 ):
            distances = model( (encapsulate(raw_inputs[index1]), encapsulate(raw_inputs[index2])) )
            if classifier(distances):
                match_count[index1] += 1
                match_count[index2] += 1
        small_cluster = []
        for item in cluster:
            if match_count[item] > 0:
                small_cluster.append(item)
        if small_cluster:
            result.append(small_cluster)
    return result

def diff_clusters( clusters1, clusters2 ):
    s2 = set( frozenset(cluster) for cluster in clusters2 )
    s1 = set( frozenset(cluster) for cluster in clusters1 )
    return [ list(cluster) for cluster in s2.difference(s1) ]

def convert_to_eval_rec(r):
    rec ={}
    if r:
        address = r.get('locations',[{}])[0].get('address1','') + r.get('locations',[{}])[0].get('address2','')
        digits = addr_reg.match(address)
        if digits:
            rec['address_number'] = digits.groups()[0]
            rec['address'] = address[len(rec['address_number']):]
        else:
            rec['address_number'] = ''
            rec['address'] = address

        rec['applicationname'] = r.get('source',{}).get('applicationName','')
        rec['country'] = r.get('locations',[{}])[0].get('countryCode','')

        email = r.get('locations',[{}])[0].get('email','').split('@')
        if len(email) >= 2:
            rec['emailname'] = email[0]
            rec['emaildomain'] = email[1]
        else:
            rec['emailname'] = email[0]
            rec['emaildomain'] = ''

        rec['fax'] = r.get('locations',[{}])[0].get('fax','')
        rec['locality'] = r.get('locations',[{}])[0].get('city','')
        rec['name'] = r.get('name','').strip()
        rec['contact_name'] = r.get('contact_name','').strip()
        rec['original_xoid'] = str(r.get('source',{}).get('id',''))
        rec['postcode'] = r.get('locations',[{}])[0].get('postalCode','')
        rec['region'] = r.get('locations',[{}])[0].get('state','')
        rec['status'] = ''
        rec['tel'] = r.get('locations',[{}])[0].get('phone','')
        rec['website'] = r.get('website','')
        rec['xoid'] = str(r.get('source',{}).get('id',''))

    return json.loads(unidecode.unidecode(json.dumps(rec)))

def generate_training(cl,recs,convert=convert_to_eval_rec,limit=5):

    rec_list = []

    cls = clus.get_clusters(cl)

    cluster_indicies = range(len(cls))
    random.shuffle(cluster_indicies)
        
    matches = []
    misses = []

    for ci in cluster_indicies[:limit]:
        cr = [convert(recs[ri]) for ri in cls[ci]]
        for p in itertools.combinations(cr,2):
            matches.append(list(p))

    clusters_indicies = range(len(cls))
    for i in xrange(limit/2):
        ci,cj = random.sample(cluster_indicies,2)
        cluster_indicies.remove(ci)
        cluster_indicies.remove(cj)
        cri = [convert(recs[ri]) for ri in cls[ci]]
        crj = [convert(recs[ri]) for ri in cls[cj]]
        for p in itertools.product(cri,crj):
            misses.append(list(p))

    training = {1:matches,0:misses}

    return training

def write_report( output_file, clusters, records, header=None ):
    if not clusters or not records:
        print( 'no clusters or records' )
        return

    if not header:
        header = records[0].keys()
    total = len(clusters)
    last_marker = 0.0
    with open( output_file, 'wb' ) as csvfile:
        writer = csv.writer( csvfile, dialect='excel', quoting=csv.QUOTE_MINIMAL )
        head = ['cluster']
        head.extend(header)
        writer.writerow( head )

        for idx, cluster in enumerate(clusters):
            cluster_recs = [ records[i] for i in cluster ]
            for record in cluster_recs:
                row = [ idx ]
                record_data = [ get_value( record, key.lower() ).translate( maketrans('\r\n', '  ') ) for key in header ]
                row.extend(record_data)
                writer.writerow( row )

            writer.writerow([])

            (_, last_marker) = console_progress( idx+1, total, True, on_increment=0.01, next_marker=last_marker )
        console_progress( total, total, True )
        print( 'done! report written to {}'.format( output_file ) )



#def init_tfidf( mongo_collection ):
#    global tfidf
#    names = [ unidecode(rec.get('name','')) for rec in mongo_collection.find( {}, { 'name' : 1 } ) ]
#    tfidf = TFIDFCompare( names )

def _main():
    config_parser = argparse.ArgumentParser( description=__doc__, 
                                             formatter_class=argparse.RawDescriptionHelpFormatter,
                                             add_help=False )
    config_parser.add_argument( '-c', '--config',
                                default='configs/default.ini',
                                help='app configuration file' )

    args, remaining_argv = config_parser.parse_known_args()
    defaults = { 'log_level' : 'INFO',
                 'threshold' : 0.5 }

    if args.config and os.path.exists( args.config ):
        config = ConfigParser.SafeConfigParser()
        config.read([args.config])
        defaults.update( dict(config.items("defaults")) )

    parser = argparse.ArgumentParser( description="smallmatch:  cluster and prune dredgeKnot clusters",
                                      usage="./smallmatch.py",
                                      parents=[config_parser] )
    
    parser.set_defaults( **defaults )

    parser.add_argument( '-l', '--log-level',
                         help='log level [INFO,DEBUG,CRITICAL,ERROR,FATAL,WARNING]' )
    parser.add_argument( '-p', '--prune',
                         action='store_true',
                         help='prune cluster output using machine learning stuff' )
    parser.add_argument( '-t', '--threshold',
                         type=float,
                         help='lr probability threshold' )
    parser.add_argument( '-i', '--input',
                         help='input mongo host:port/db.collection' )
    parser.add_argument( '-o', '--output',
                         help='output mongo location <host:port/db.collection>' )
    parser.add_argument( '--fast',
                         action='store_true',
                         help='go real fast' )
    parser.add_argument( '--auto-threshold',
                         action='store_true',
                         help='try to choose the best threshold' )
    parser.add_argument( '--test',
                         action='store_true',
                         help='load fewer records' )
    parser.add_argument( '--instrument',
                         action='store_true',
                         help=argparse.SUPPRESS )


    args = parser.parse_args()

    if args.instrument:
        import pdb; pdb.set_trace()
        print( "welcome" )

    threshold = float(args.threshold)

    (mongohost, mongoport, database_name, collection_name) = parse_mongo_spec( args.input )
    input_collection = MongoClient( mongohost, mongoport )[database_name][collection_name]

#    print( 'initializing tfidf on names from {}'.format( collection_name ) )
#    init_tfidf(input_collection)
#    print( 'done' )

    print( 'loading mongo recs' )
    (mongohost, mongoport, database_name, collection_name) = parse_mongo_spec( args.output )
    output_collection = MongoClient( mongohost, mongoport )[database_name][collection_name]

    if args.test:
        raw_inputs = [ rec for rec in input_collection.find().limit(1000) ]
    else:
        raw_inputs = [ rec for rec in input_collection.find() ]

    print( 'done' )
    
    print( 'encoding records..' )
    enc_recs = clean_recs(raw_inputs)
    print( 'clustering records..' )
    thresholds = None
    if not args.auto_threshold:
        train_threshold = 600
        run_threshold = 500
        thresholds = [ train_threshold, run_threshold ]
    
    if args.fast:
        print( 'fast requested, using fewer ngrams' )
#        all_clusters, index, frequencies = cluster.cluster(enc_recs, thresholds=thresholds, ngram_sizes=[7,15,23,31], keep_index=True)
        all_clusters = cluster.cluster(enc_recs, thresholds=thresholds, ngram_sizes=[7,15,23,31])
    else:
        all_clusters = cluster.cluster(enc_recs, thresholds=thresholds)

    if args.auto_threshold:
        train_threshold = max( all_clusters.keys()[0], all_clusters.keys()[1] )
        run_threshold = min( all_clusters.keys()[0], all_clusters.keys()[1] )

#    cluster.score_records( enc_recs[0], enc_recs[100], index, frequencies )
    
    print( 'cluster result thresholded at {} and {}'.format( run_threshold, train_threshold ) ) 

    train_clusters = [ c for c in all_clusters[train_threshold] if len(c) < 6 ]
    run_clusters = all_clusters[run_threshold]

#    print( 'generating dredgeKnot clusters' )
#    clusters = [ c for c in dredgeKnot.cluster( raw_inputs ) if len(c) > 1 ]

    print( 'done' )
    run_total = int( sum( (len(c)**2.0-len(c))/2.0 for c in run_clusters ) )
    train_total = int( sum( (len(c)**2.0-len(c))/2.0 for c in train_clusters ) )
    print( 'generated {}/{} train/run clusters with {}/{} train/run records'.format(len(train_clusters),len(run_clusters),
                                                                                    train_total,run_total) )

    custom_header = [ 'source.id',
                      'source.applicationName',
                      'source.tblID',
                      'attributes.servicesCode.0',
                      'name',
                      'website',
                      'locations.0.address1',
                      'locations.0.address2',
                      'locations.0.city',
                      'locations.0.state',
                      'locations.0.postalCode',
                      'locations.0.countryCode',
                      'locations.0.phone',
                      'locations.0.fax',
                      'locations.0.email' ] # aw yeah.

    write_report( '/var/tmp/raw_report.csv', run_clusters, raw_inputs, header=custom_header )
    write_report( '/var/tmp/raw_train.csv', train_clusters, raw_inputs, header=custom_header )

    print( 'culling training samples' )

    training_inputs = [ raw_inputs[i] for c in train_clusters for i in c ]
    new_train_clusters = []
    i = 0
    for c in train_clusters:
        new_train_clusters.append( range(i, i+len(c)) )
        i += len(c)

    train_clusters = new_train_clusters

    empty_record = {u'website': u'',
                    u'name': u'',
                    u'locations': [{u'city': u'',
                                    u'fax': u'',
                                    u'countryCode': u'',
                                    u'address1': u'',
                                    u'address2': u'',
                                    u'phone': u'',
                                    u'state': u'',
                                    u'postalCode': u'',
                                    u'email': u''}],
                    u'unused': {u'local market': u'',
                                u'gdocid': u''},
                    u'source': {u'applicationName': u'CRMMarch',
                                u'applicationId': u'44',
                                u'id': u'',
                                u'objectType': u'CRMConcierge'},
                    u'attributes': {u'services': [u''],
                                    u'servicesCode': [u''],
                                    u'marketCode': [u''],
                                    u'market': [u'']},
                    u'contact_name': u'' }
    for index in random.sample( xrange(len(train_clusters)), int(len(train_clusters)*0.1) ):
        i = random.choice( train_clusters[index] )
        nrec = empty_record
        nrec['name'] = training_inputs[i]['name']
        training_inputs[i] = nrec
        
    training = generate_training( train_clusters, training_inputs, convert=lambda x:x, limit=int(0.5*len(train_clusters)) )
    positives = [ ( encapsulate(rec1), encapsulate(rec2) )
                  for rec1,rec2 in training[1] ]
    shuffle(positives)
    negatives = [ ( encapsulate(rec1), encapsulate(rec2) )
                  for rec1,rec2 in training[0] ]
    shuffle(negatives)
    print( 'training logistic regression on {}/{} pos/neg samples'.format(len(positives),len(negatives)) )

    classifier = LogisticRegressionClassifier( model,
                                               positives,
                                               negatives,
                                               percentage_training=1.0,
                                               preserve_samples=False )
    
    lclassifier = classifier.get_classifier()
    print( 'done' )

    cluster_recs = []
    distances = []
    print( 'building for reclustering...' )
    for c in run_clusters:
        if len(c) < 3:
            continue
#        print( 'computing {} pairs'.format( (len(c)**2 - len(c))/2.0 ) )
        encoded_cluster = { i : encapsulate(raw_inputs[i]) for i in c }
        for index1, index2 in itertools.combinations( c, 2 ):
            distances.append( model( (encoded_cluster[index1], encoded_cluster[index2]) ) )
            cluster_recs.append( (index1,index2) )
#        print( 'done' )

    print( 'done' )
    print( 'probabilities and reclustering...' )
    probabilities = lclassifier.predict_proba( distances )[::,1]
    indices = (probabilities >= threshold).nonzero()[0]
    probabilities = probabilities[indices]
    search_indices = set( indices.tolist() )
    cluster_recs = [ pair for i, pair in enumerate(cluster_recs) if i in search_indices ]
    
    (new_clusters, cluster_scores) = hierarchical_cluster( zip( cluster_recs, probabilities ), threshold )

    print( 'done' )

    two_clusters = [ c for c in run_clusters if len(c) < 3 ]
    new_clusters = [ list(c) for c in new_clusters ]
    new_clusters.extend( two_clusters )
    new_clusters = [ c for c in reversed( sorted( new_clusters, key=lambda x: len(x) ) ) ]
    write_report( '/var/tmp/report.csv', new_clusters, raw_inputs, header=custom_header )

    if args.prune:
        training_file = '/var/tmp/smallmatch_training.json'
        if not os.path.exists( training_file ):
            print( 'culling training samples' )
            training = generate_training( train_clusters, training_inputs, convert=lambda x:x, limit=len(train_clusters) )
            positives = [ ( encapsulate(rec1), encapsulate(rec2) ) 
                          for rec1,rec2 in training[1] ]
            negatives = [ ( encapsulate(rec1), encapsulate(rec2) ) 
                          for rec1,rec2 in training[0] ]
            print( 'done' )
        
            with open( training_file, 'w' ) as fil:
                json.dump( { '1' : positives, '0' : negatives }, fil, indent=4, default=json_util.default, separators=(',',':  ') )
                print( 'wrote {}'.format(training_file) )
        else:
            with open( training_file, 'r' ) as fil:
                training = json.load(fil)
                positives = training['1']
                negatives = training['0']
                print( 'loaded {}'.format(training_file) )
    
        print( 'training homebrew lr on {}/{} pos/neg samples'.format(len(positives),len(negatives)) )
        labels = np.concatenate( (np.ones(len(positives),), np.zeros(len(negatives),)) )
        examples = np.concatenate( ([ model(record) for record in positives ], [ model(record) for record in negatives ]) )
    
        pickle_file = '/var/tmp/hblr.pickle'
        if os.path.exists( pickle_file ):
            with open( pickle_file, 'rb' ) as input:
                classifier = pickle.load( input )
        else:
            classifier = HBLogisticRegression( labels, examples )
            with open( pickle_file, 'wb' ) as input:
                pickle.dump( classifier, input )
    
        print( 'done' )
    
        pruned_clusters_1 = prune_clusters( raw_inputs, run_clusters, lambda distances: classifier.predict(distances) > threshold )
        print( 'pruned_clusters_1: {}'.format(len(pruned_clusters_1)) )
        diff_clusters_1 = diff_clusters( run_clusters, pruned_clusters_1 )
    
        print( 'training peceptron on {}/{} pos/neg samples'.format(len(positives),len(negatives)) )
        classifier = PerceptronClassifier( model,
                                           positives,
                                           negatives,
                                           percentage_training=1.0,
                                           preserve_samples=False )
    
        pclassifier = classifier.perceptron_classifier()
        print( 'done' )
        pruned_clusters_2 = prune_clusters( raw_inputs, run_clusters, lambda distances: pclassifier.predict(distances) )
        print( 'pruned_clusters_2: {}'.format(len(pruned_clusters_2)) )
        diff_clusters_2 = diff_clusters( run_clusters, pruned_clusters_2 )
    
        print( 'training lr model on {}/{} pos/neg samples'.format(len(positives),len(negatives)) )
        classifier = LogisticRegressionClassifier( model,
                                                   positives,
                                                   negatives,
                                                   percentage_training=1.0,
                                                   preserve_samples=False )
        
        lclassifier = classifier.get_classifier()
        print( 'done' )
        pruned_clusters_3 = prune_clusters( raw_inputs, run_clusters, lambda distances: lclassifier.predict_proba(distances)[0][1] > threshold )
        print( 'pruned_clusters_3: {}'.format(len(pruned_clusters_3)) )
        diff_clusters_3 = diff_clusters( run_clusters, pruned_clusters_3 )
    
        write_report( '/var/tmp/report_hblr_diff.csv', diff_clusters_1, raw_inputs, header=custom_header )
        write_report( '/var/tmp/report_perceptron_diff.csv', diff_clusters_2, raw_inputs, header=custom_header )
        write_report( '/var/tmp/report_lr_diff.csv', diff_clusters_3, raw_inputs, header=custom_header )

    return 0


if __name__ == '__main__':
    import sys
    sys.exit( _main() )
