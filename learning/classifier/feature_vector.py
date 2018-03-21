#!/usr/bin/env python
'''
feature_vector.py:  Calculate and generate features given a profile
'''
from __future__ import print_function, division
import operator
import subprocess as sub
import math
import jellyfish
import re
from classification import NaiveBayesClassifier
from collections import OrderedDict, defaultdict
import os
import cloud.serialization.cloudpickle as pickle
import string
import urllib
import urllib2
import json
from fuzzywuzzy import process
from nltk import sent_tokenize
from utils.json_util2 import get_value, get_raw_value, stringify
from dedupe.distance.affinegap import normalizedAffineGapDistance
from feature_base import Feature, BinaryFeature, DiscreteFeature, RealValuedFeature
from nltk import wordpunct_tokenize
import hashlib

class TermFeature( BinaryFeature ):
    def __init__( self, term ):
        Feature.__init__( self, term, 0.0 )
    def resolve( self, content ):
        success = Feature.resolve( self, content )
        if success is None:
            self.score = int( self.name in content.get( 'terms', [] ) )
            content = Feature.update( self, content )
        return content

class ClassifierFeature( BinaryFeature ):
    def __init__( self, name, fieldname, bayes_classifier ):
        Feature.__init__( self, name, 0.0 )
        self.fieldname = fieldname
        self.classifier = bayes_classifier

    def resolve( self, content ):
        success = Feature.resolve( self, content )
        if success is None:
            self.score = 0.0
            fieldvalue = get_value( content, self.fieldname ).lower()
            if fieldvalue:
                self.score = self.classifier.naive_bayes_classifier().classify( self.classifier.feature_generator()( fieldvalue ) )
            content = Feature.update( self, content )

        return content

def load_featurevector( filename ):
    with open(filename, 'rb') as fil:
        whole = fil.read()
        fv = pickle.loads( whole )
        return fv

def dump_featurevector( filename, fv ):
    with open(filename, 'wb') as fil:
        whole = pickle.dumps( fv, protocol=1 )
        fil.write(whole)
        fil.flush()

class TermFeatureVector:

    def __init__( self, terms ):
        self.feature_vec = OrderedDict()

        for term in sorted(terms):
            feature = TermFeature( term )
            self.feature_vec[feature.name] = feature

    def feature_names( self ):
        return self.feature_vec.keys()

    def resolve( self, content, feature_predicate=None ):
        if not feature_predicate:
            feature_predicate = lambda x: True

        for (name,feature) in self.feature_vec.iteritems():
            if feature_predicate(feature):
                content = feature.resolve( content )

        return (content, content['features']['scores'])
        
def _main():
    import logging
    from pprint import pprint as pp
    import cProfile

    logging.basicConfig(level='DEBUG')

    fv_pickle = '/var/tmp/.feature_vector.pickle'
    if os.path.exists( fv_pickle ):
        fv = load_featurevector( fv_pickle )
    else:
        import merge.features.feature_vector
        fv = merge.features.feature_vector.FeatureVector()
        dump_featurevector( fv_pickle, fv )

    fv.initialize()

#    (profile, features) = fv.resolve( test_profile )
#    print( 'features are: %s' % profile['features']['scores'] )
#    for (feature_class, feature_scores) in features.iteritems():
#        print( "%s" % feature_class )
#        for (name, score) in feature_scores.iteritems():
#            print( "%s:  %f" % (name, score) )

    return 0

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
