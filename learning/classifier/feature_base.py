#!/usr/bin/env python
from collections import OrderedDict
class Feature:
    def __init__( self, name, score ):
        self.name = name
        self.default_score = score
        self.score = self.default_score

    def get_score( self, profile ):
        return profile.get( 'features', {} ).get( 'scores', {} ).get( self.feature_class(), {} ).get( self.name, None )
    
    def resolve( self, profile ):
        feature_score = self.get_score( profile )
        if feature_score is not None:
            self.score = feature_score
            return self.update( profile )
        else:
            return None

    def update( self, profile ):
        if not profile.has_key( 'features' ):
            profile['features'] = OrderedDict()
        feature_class = self.feature_class()
        profile['features'].setdefault( 'scores', OrderedDict() )
        profile['features']['scores'].setdefault( feature_class, OrderedDict() )
        profile['features']['scores'][feature_class][self.name] = self.score
        return profile

    def reset( self ):
        self.score = self.default_score

class BinaryFeature(Feature):
    def feature_class( self ):
        return "BinaryFeature"

class DiscreteFeature(Feature):
    def feature_class( self ):
        return "DiscreteFeature"

class RealValuedFeature(Feature):
    def feature_class( self ):
        return "RealValuedFeature"
