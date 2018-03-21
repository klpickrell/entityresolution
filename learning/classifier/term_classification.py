#!/usr/bin/env python

import nltk
from nltk import wordpunct_tokenize
from sklearn import linear_model
import collections
import re
import random
import numpy as np
from collections import OrderedDict
import json
from utils.json_util2 import get_value
from feature_base import Feature, BinaryFeature, DiscreteFeature, RealValuedFeature

class TermClassifier:
    def __init__( self,
                  classifier_type,
                  feature_names,
                  training,
                  feature_resolver = None,
                  percentage_training = 0.8,
                  preserve_samples = False ):

        self.feature_names = list(feature_names)
        self.training = training
        self.feature_resolver = feature_resolver

        if self.feature_resolver:
            self.classifier_g = classifier_type( self.feature_generator, training['positives'], training['negatives'], percentage_training, preserve_samples )
        else:
            self.classifier_g = classifier_type( None, training['positives'], training['negatives'], percentage_training, preserve_samples )
        self.classifier = self.classifier_g.get_classifier()

    def sample_count( self ):
        return self.classifier_g.sample_count()

    def get_classifier( self ):
        return self.classifier

    def feature_names( self ):
        if self.feature_names:
            return self.feature_names
        elif self.feature_resolver:
            return self.feature_resolver.feature_names()

    def feature_generator( self, content ):
        (content, scores) = self.feature_resolver.resolve( content )
        all_scores = OrderedDict()
        if not self.feature_names:
            for feature_type in [ 'BinaryFeature', 'DiscreteFeature', 'RealValuedFeature' ]:
                if scores.has_key( feature_type ):
                    all_scores.update( scores[feature_type] )
        else:
            for feature_type in [ 'BinaryFeature', 'DiscreteFeature', 'RealValuedFeature' ]:
                if scores.has_key( feature_type ):
                    all_scores.update( { key : value for (key,value) in scores[feature_type].iteritems() if key in self.feature_names } )
            # reorder all_scores based on feature order in feature_names
            ordered_all_scores = OrderedDict()
            for feature in self.feature_names:
                ordered_all_scores[feature] = all_scores[feature]
            all_scores = ordered_all_scores

        return all_scores

    def resolve_and_predict( self, content ):
        if not self.feature_resolver:
            raise Exception( 'need a feature resolver to resolve' )
        if type(content) is list:
            content_list = content
        if type(content) is dict:
            content_list = [ content ]

        samples = []
        for content in content_list:
            (_, scores) = self.feature_resolver.resolve( content )
            all_scores = OrderedDict()
            if not self.feature_names:
                for feature_type in [ 'BinaryFeature', 'DiscreteFeature', 'RealValuedFeature' ]:
                    if scores.has_key( feature_type ):
                        all_scores.update( scores[feature_type] )
            else:
                for feature_type in [ 'BinaryFeature', 'DiscreteFeature', 'RealValuedFeature' ]:
                    if scores.has_key( feature_type ):
                        all_scores.update( { key : value for (key,value) in scores[feature_type].iteritems() if key in self.feature_names } )
     
                # reorder all_scores based on feature order in feature_names
                ordered_all_scores = OrderedDict()
                for feature in self.feature_names:
                    ordered_all_scores[feature] = all_scores[feature]
                all_scores = ordered_all_scores

            samples.append( all_scores.values() )

        result = self.classifier.predict_proba( np.array(samples) )
        return result[::,1]

    def predict( self, samples ):
        result = self.classifier.predict_proba( np.array(samples) )
        return result[::,1]


    def test_all( self ):
        print( "="*30 + "Term Classification" + "="*30 )
        print( "="*30+self.classifier_g.__class__.__name__+"="*30 )
        self.classifier_g.test()

def _main():
    return 0

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
