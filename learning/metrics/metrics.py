#!/usr/bin/env python
from nltk import word_tokenize

class MetricWrapper:
    '''
    Add a getSimilarity method
    '''
    def __init__( self, metric ):
        self.metric = metric
    def getSimilarity( self, str1, str2 ):
        return self.metric( str1, str2 )

def denormalize_score( score, low=5.5, high=0.5 ):
    '''
    move score from range 0-1 to low-high
    '''
    return ((1-score)*(low-high) + high)

def normalize_score( score, low=5.5, high=0.5 ):
    '''
    move score from range low-high to 0-1
    '''
    return 1-(score-high)/(low-high)

def hybrid_metric( compare, hybrid, str1, str2, token_threshold=0.9 ):
    if str1 == str2:
        return 1.0
    tokens_in1 = word_tokenize(str1)
    tokens_in2 = word_tokenize(str2)
    current_token = 'A'
    token_string1 = []
    token_string2 = []

    while tokens_in1 and tokens_in2:
        token1 = tokens_in1.pop(0)
        if token1 in tokens_in2:
            token_string1.append( current_token )
            token_string2.append( current_token )
            current_token = chr( ord(current_token) + 1 )
            tokens_in2.pop(tokens_in2.index(token1))
        else:
            popit = None
            for idx, token2 in enumerate( tokens_in2 ):
                if compare( token1, token2 ) > token_threshold:
                    token_string1.append( current_token )
                    token_string2.append( current_token )
                    current_token = chr( ord(current_token) + 1 )
                    popit = idx
                    break
            if popit is not None:
                tokens_in2.pop( popit )
            else:
                token_string1.append( current_token )
                current_token = chr( ord(current_token) + 1 )

    for token in tokens_in2:
        token_string2.append( current_token )
        current_token = chr( ord(current_token) + 1 )

    return compare( str1, str2 ) * hybrid.getSimilarity( str("".join(token_string1)), str("".join(token_string2)) )
