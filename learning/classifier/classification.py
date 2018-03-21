#!/usr/bin/env python

import nltk
from nltk import wordpunct_tokenize
from sklearn import linear_model, tree, svm
import collections
from collections import OrderedDict
import re
import random
import numpy as np

class BinaryClassifier:
    def __init__( self, feature_generator, positives, negatives, percentage_training=0.8 ):
        self.memo = dict()
        self._feature_generator = feature_generator
        if type( positives ) == str:
            r_positives = [ line.rstrip() for line in open( positives ).readlines() ]
        else:
            r_positives = positives
        if type( negatives ) == str:
            r_negatives = [ line.rstrip() for line in open( negatives ).readlines() ]
        else:
            r_negatives = negatives
        self.memo['training_set'] = ( [ (item, 1) for item in r_positives ] +
                                      [ (item, 0) for item in r_negatives ] )
        self.training_fold = int( len( self.memo['training_set'] ) * percentage_training )
        random.shuffle( self.memo['training_set'] )
        self.preserve_samples = not np.isclose( percentage_training, 1.0 )

    def sample_count( self ):
        return self.training_fold

    def feature_generator( self ):
        return self._feature_generator

    def test( self, classifier ):
        pass

    def cleanup( self ):
        self.memo = dict()


class NaiveBayesClassifier(BinaryClassifier):
    def __init__( self, feature_generator, positives, negatives, percentage_training=0.8 ):
        BinaryClassifier.__init__( self, feature_generator, positives, negatives, percentage_training )
        self.naivebayes_c = None
        self.naivebayes_c = self.get_classifier()
        if not self.preserve_samples:
            self.cleanup()

    def get_classifier( self ):
        if self.naivebayes_c:
            return self.naivebayes_c

        if self.feature_generator():
            self.memo['feature_sets'] = [ (self.feature_generator()(item), label) for (item, label) in self.memo['training_set'] ]
        else:
            self.memo['feature_sets'] = [ (item, label) for (item, label) in self.memo['training_set'] ]
        train = self.memo['feature_sets'][:self.training_fold]
        self.naivebayes_c = nltk.NaiveBayesClassifier.train( train )
        return self.naivebayes_c

        
    def test( self, show_informative=None ):
        if self.naivebayes_c and self.memo and self.memo.has_key( 'feature_sets' ):
            test = self.memo['feature_sets'][self.training_fold:]
            reference_set = collections.defaultdict(set)
            test_set = collections.defaultdict(set)
            for i, (feats, label) in enumerate(test):
                reference_set[label].add(i)
                observed = self.naivebayes_c.classify(feats)
                test_set[observed].add(i)

            if show_informative:
                print self.naivebayes_c.show_most_informative_features( show_informative )

            print 'nltk accuracy:\t', nltk.classify.accuracy( self.naivebayes_c, test )
            print 'pos precision:\t', nltk.metrics.precision(reference_set[1], test_set[1])
            print 'pos recall:\t', nltk.metrics.recall(reference_set[1], test_set[1])
            print 'pos F-measure:\t', nltk.metrics.f_measure(reference_set[1], test_set[1])
            print 'neg precision:\t', nltk.metrics.precision(reference_set[0], test_set[0])
            print 'neg recall:\t', nltk.metrics.recall(reference_set[0], test_set[0])
            print 'neg F-measure:\t', nltk.metrics.f_measure(reference_set[0], test_set[0])
        
            fp1 = test_set[1].intersection( reference_set[1] )
            fp2 = test_set[0].intersection( reference_set[0] )
            precision = (len(fp1) + len(fp2)) / float(len(test_set[1]) + len(test_set[0]))
            
            print 'overall precision:\t', precision
        else:
            print( "No classifier or feature_sets memoized for NaiveBayesClassifier.test" )

class LogisticRegressionClassifier(BinaryClassifier):
    def __init__( self, feature_generator, positives, negatives, percentage_training=0.8 ):
        BinaryClassifier.__init__( self, feature_generator, positives, negatives, percentage_training )
        self.logreg_c = None
        self.logreg_c = self.get_classifier()
        if not self.preserve_samples:
            self.cleanup()

    def get_classifier( self ):
        if self.logreg_c:
            return self.logreg_c

        if self.feature_generator():
            self.memo['domain'] = np.array([ self.feature_generator()(item[0]) for item in self.memo['training_set'] ])
        else:
            self.memo['domain'] = np.array([ vector for vector,label in self.memo['training_set'] ])
        self.memo['target'] = np.array( [ t[1] for t in self.memo['training_set'] ] )
        self.logreg_c = linear_model.LogisticRegression()
        self.logreg_c.fit(self.memo['domain'][:self.training_fold], self.memo['target'][:self.training_fold])
        return self.logreg_c

    def test( self ):
        if self.logreg_c and self.memo and self.memo.has_key( 'domain' ) and self.memo.has_key( 'target' ):
            print( 'Mean distance from hyperplane:\t' + str(self.logreg_c.score(self.memo['domain'][self.training_fold:], self.memo['target'][self.training_fold:])) )
            probs = self.logreg_c.predict_proba( self.memo['domain'][self.training_fold:] )[::,1]
            targets = self.memo['target'][self.training_fold:]
            print( 'Mean proximity to true label:\t' + str((1-(sum( (targets-probs)**2)/len(probs)))) )
            print( 'MSE:\t\t\t\t' + str( (sum( (targets-probs)**2)/len(probs))) )
            print( 'RMSE:\t\t\t\t' + str( np.sqrt((sum( (targets-probs)**2)/len(probs)))) )
            print( 'samples:\t\t\t\t{}'.format( len(probs) ) )

        else:
            print( "No classifier or domain and target memoized for LogisticRegressionClassifier.test" )


class PerceptronClassifier(BinaryClassifier):
    def __init__( self, feature_generator, positives, negatives, percentage_training=0.8 ):
        BinaryClassifier.__init__( self, feature_generator, positives, negatives, percentage_training )
        self.perceptron_c = None
        self.perceptron_c = self.get_classifier()
        if not self.preserve_samples:
            self.cleanup()

    def get_classifier( self ):
        if self.perceptron_c:
            return self.perceptron_c

        if self.feature_generator():
            self.memo['domain'] = np.array([ self.feature_generator()(item[0]) for item in self.memo['training_set'] ])
        else:
            self.memo['domain'] = np.array([ vector for vector,label in self.memo['training_set'] ])
#        self.memo['feature_sets'] = [ (self.feature_generator()(item), label) for (item, label) in self.memo['training_set'] ]
#        self.memo['domain'] = np.array([ self.feature_generator()(item[0]) for item in self.memo['training_set'] ])
        self.memo['target'] = np.array( [ t[1] for t in self.memo['training_set'] ] )
        self.perceptron_c = linear_model.perceptron.Perceptron()
        self.perceptron_c.fit(self.memo['domain'][:self.training_fold], self.memo['target'][:self.training_fold])
        return self.perceptron_c

    def test( self ):
        if self.perceptron_c and self.memo and self.memo.has_key( 'domain' ) and self.memo.has_key( 'target' ):
            test = self.memo['domain'][self.training_fold:]
            targets = self.memo['target'][self.training_fold:]
            reference_set = collections.defaultdict(set)
            test_set = collections.defaultdict(set)
            for i, (feats, label) in enumerate( zip(test, targets) ):
                reference_set[label].add(i)
                observed = self.perceptron_c.predict(feats)[0]
                test_set[observed].add(i)
        
            print 'Perceptron:'
            print 'pos precision:\t', nltk.metrics.precision(reference_set[1], test_set[1])
            print 'pos recall:\t', nltk.metrics.recall(reference_set[1], test_set[1])
            print 'pos F-measure:\t', nltk.metrics.f_measure(reference_set[1], test_set[1])
            print 'neg precision:\t', nltk.metrics.precision(reference_set[0], test_set[0])
            print 'neg recall:\t', nltk.metrics.recall(reference_set[0], test_set[0])
            print 'neg F-measure:\t', nltk.metrics.f_measure(reference_set[0], test_set[0])
        
            fp1 = test_set[1].intersection( reference_set[1] )
            fp2 = test_set[0].intersection( reference_set[0] )
            precision = (len(fp1) + len(fp2)) / float(len(test_set[1]) + len(test_set[0]))
            
            print 'precision (perceptron):\t', precision
        else:
            print( "No classifier or feature_sets memoized for PerceptronClassifier.test" )

class MultiClassClassifier:
    def __init__( self, feature_generator, samples, percentage_training=0.8 ):
        self.memo = dict()
        self._feature_generator = feature_generator
        self.memo['training_set'] = samples

        self.training_fold = int( len( self.memo['training_set'] ) * percentage_training )
        random.shuffle( self.memo['training_set'] )
        self.preserve_samples = not np.isclose( percentage_training, 1.0 )

    def sample_count( self ):
        return self.training_fold

    def feature_generator( self ):
        return self._feature_generator

    def test( self, classifier ):
        pass

    def cleanup( self ):
        self.memo = dict()


class DecisionTree(MultiClassClassifier):
    def __init__( self, feature_generator, samples, percentage_training=0.8 ):
        MultiClassClassifier.__init__( self, feature_generator, samples, percentage_training )
        self.decision_tree = None
        self.decision_tree = self.get_classifier()
        if not self.preserve_samples:
            self.cleanup()

    def get_classifier( self ):
        if self.decision_tree:
            return self.decision_tree

        if self.preserve_samples:
            self.memo['feature_sets'] = [ (self.feature_generator()(item), label) for (item, label) in self.memo['training_set'] ]

        self.decision_tree = tree.DecisionTreeClassifier( criterion='entropy' )

        self.memo['domain'] = np.array([ self.feature_generator()(item).values() for (item, label) in self.memo['training_set'] ])
        self.memo['target'] = np.array( [ t[1] for t in self.memo['training_set'] ] )
        self.decision_tree.fit(self.memo['domain'][:self.training_fold], self.memo['target'][:self.training_fold])
        return self.decision_tree

    def test( self ):
        if self.decision_tree and self.memo and self.memo.has_key( 'feature_sets' ):
            difference = 0.0
            difference_squared = 0.0
            test = self.memo['feature_sets'][self.training_fold:]
            reference_set = collections.defaultdict(set)
            test_set = collections.defaultdict(set)
            for test_id, (feats, label) in enumerate(test):
                reference_set[label].add(test_id)
                observed = self.decision_tree.predict(feats.values())
                test_set[observed[0]].add(test_id)
                residual = np.fabs( float(observed) - float(label) )
                difference += residual
                difference_squared += residual**2
        
#            fp1 = test_set[1].intersection( reference_set[1] )
#            fp2 = test_set[0].intersection( reference_set[0] )
#            accuracy = (len(fp1) + len(fp2)) / float(len(test_set[1]) + len(test_set[0]))

            successful = 0.0
            total = 0.0
            for label, samples in test_set.iteritems():
                common = samples.intersection( reference_set[label] )
                successful += len( common )
                total += len( samples )
            
            if total == 0:
                accuracy = "-1"
                total = 0.0000001
            else:
                accuracy = successful/total

            print( 'accuracy:\t\t%f' % float(accuracy) )
            print( 'mean error:\t\t%f' % (difference/total) )
            print( 'mean squared error:\t%f' % (difference_squared/total) )

#            print( 'pos precision:\t%f' % nltk.metrics.precision(reference_set[1], test_set[1]) )
#            print( 'pos recall:\t%f' % nltk.metrics.recall(reference_set[1], test_set[1]) )
#            print( 'pos F-measure:\t%f' % nltk.metrics.f_measure(reference_set[1], test_set[1]) )
#            print( 'neg precision:\t%f' % nltk.metrics.precision(reference_set[0], test_set[0]) )
#            print( 'neg recall:\t%f' % nltk.metrics.recall(reference_set[0], test_set[0]) )
#            print( 'neg F-measure:\t%f' % nltk.metrics.f_measure(reference_set[0], test_set[0]) )
        
        else:
            print( "No classifier or feature_sets memoized for DecisionTree.test" )

class SVMClassifier(MultiClassClassifier):
    def __init__( self, feature_generator, samples, percentage_training=0.8 ):
        MultiClassClassifier.__init__( self, feature_generator, samples, percentage_training )
        self.svm = None
        self.svm = self.get_classifier()
        if not self.preserve_samples:
            self.cleanup()

    def get_classifier( self ):
        if self.svm:
            return self.svm

        if self.preserve_samples:
            self.memo['feature_sets'] = [ (self.feature_generator()(item), label) for (item, label) in self.memo['training_set'] ]

        self.svm = svm.SVC()

        self.memo['domain'] = np.array([ self.feature_generator()(item).values() for (item, label) in self.memo['training_set'] ])
        self.memo['target'] = np.array( [ t[1] for t in self.memo['training_set'] ] )
        self.svm.fit(self.memo['domain'][:self.training_fold], self.memo['target'][:self.training_fold])
        return self.svm

    def test( self ):
        if self.svm and self.memo and self.memo.has_key( 'feature_sets' ):
            difference = 0.0
            difference_squared = 0.0
            test = self.memo['feature_sets'][self.training_fold:]
            reference_set = collections.defaultdict(set)
            test_set = collections.defaultdict(set)
            for test_id, (feats, label) in enumerate(test):
                reference_set[label].add(test_id)
                observed = self.svm.predict(feats.values())
                test_set[observed[0]].add(test_id)
                residual = np.fabs( float(observed) - float(label) )
                difference += residual
                difference_squared += residual**2
        
#            fp1 = test_set[1].intersection( reference_set[1] )
#            fp2 = test_set[0].intersection( reference_set[0] )
#            accuracy = (len(fp1) + len(fp2)) / float(len(test_set[1]) + len(test_set[0]))

            successful = 0.0
            total = 0.0
            for label, samples in test_set.iteritems():
                common = samples.intersection( reference_set[label] )
                successful += len( common )
                total += len( samples )
            
            if total == 0:
                accuracy = "-1"
                total = 0.0000001
            else:
                accuracy = successful/total

            print( 'accuracy:\t\t%f' % float(accuracy) )
            print( 'mean error:\t\t%f' % (difference/total) )
            print( 'mean squared error:\t%f' % (difference_squared/total) )

#            print( 'pos precision:\t%f' % nltk.metrics.precision(reference_set[1], test_set[1]) )
#            print( 'pos recall:\t%f' % nltk.metrics.recall(reference_set[1], test_set[1]) )
#            print( 'pos F-measure:\t%f' % nltk.metrics.f_measure(reference_set[1], test_set[1]) )
#            print( 'neg precision:\t%f' % nltk.metrics.precision(reference_set[0], test_set[0]) )
#            print( 'neg recall:\t%f' % nltk.metrics.recall(reference_set[0], test_set[0]) )
#            print( 'neg F-measure:\t%f' % nltk.metrics.f_measure(reference_set[0], test_set[0]) )
        
        else:
            print( "No classifier or feature_sets memoized for SVMClassifier.test" )


def _main():

#    phones = GeneralLinearClassifier( phone_format_features, 
#                                      '../pymerge-data/phone_proper.txt',
#                                      '../pymerge-data/phone_improper.txt' )
#
#    print( "\n" + "="*30+"Phone Format"+"="*30 )
#
#    bayes = phones.naive_bayes_classifier()
#    lr    = phones.logistic_regression_classifier()
#    perceptron_c = phones.perceptron_classifier()
#    phones.test( bayes )
#    phones.test( lr )
#    phones.test( perceptron_c )

    return 0

if __name__ == '__main__':
    import sys
    sys.exit( _main() )
