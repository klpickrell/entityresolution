#!/usr/bin/env python
import pyximport
pyximport.install()

from learning.metrics import jaccard
from learning.metrics.metrics import denormalize_score
from numpy import nan

import os
from gensim import corpora, models, similarities
from nltk.tokenize import wordpunct_tokenize

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

class TFIDFCompare:
    def __init__( self, documents ):
        corpus_file = '/var/tmp/smallmatch_corpus.mm'
        dict_file = '/var/tmp/smallmatch_dict.mm'
        if os.path.exists( corpus_file ) and os.path.exists( dict_file ):
            self.corpus = corpora.MmCorpus( corpus_file )
            self.dictionary = corpora.Dictionary.load( dict_file )
        else:
            self.tokenized_names = [ wordpunct_tokenize(item.lower()) for item in documents ]
            self.dictionary = corpora.Dictionary( self.tokenized_names )
            corpus = [ self.dictionary.doc2bow( item ) for item in self.tokenized_names ]
            corpora.MmCorpus.serialize( corpus_file, corpus )
            self.corpus = corpora.MmCorpus( corpus_file )
            self.dictionary.save( dict_file )

        self.tfidf = models.TfidfModel( self.corpus )

    def compare( self, s1, s2 ):
        e1 = self.dictionary.doc2bow(wordpunct_tokenize(s1.lower()))
        e2 = self.dictionary.doc2bow(wordpunct_tokenize(s2.lower()))
        t1 = self.tfidf[e1]
        t2 = self.tfidf[e2]
        index = similarities.MatrixSimilarity( [t1], num_features=self.corpus.num_terms )
        return index[t2][0]
        
