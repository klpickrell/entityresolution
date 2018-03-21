#cython: boundscheck=False, wraparound=False, cdivision=True

import math
import numpy as np
import random


def lr(labels,
       examples,
       alpha,
       miniterations,
       maxiterations,
       tss_tolerance,
       shuffle=True):

    if shuffle:
        result = zip(labels,examples)
        random.shuffle(result)
        labels = np.array( [ item[0] for item in result ] )
        examples = np.array( [ item[1] for item in result ] )

    i, j, label, n_features, n_examples = (0,0,0,0,0)
    rate, rate_n, bias, predicted, update, error, logit = (0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    weight = 0.0

    n_examples = examples.shape[0]
    n_features = examples.shape[1]

    rate = 0.01

    weight = examples[0] * 0
    bias = 0
    n = maxiterations
    for i in range(maxiterations):
        rate_n = rate * (n-i)/n
        tss = 0.0
        for _ in range(n_examples) :
            j = random.randrange(n_examples)
            label = labels[j]

            logit = bias
            for k in range(n_features):
                logit += weight[k] * examples[j,k]

            predicted = 1.0 / (1.0 + math.exp(min(-logit,100)))

            error = label - predicted

            for k in range(n_features):
                update = error * examples[j,k] - (alpha * weight[k])
                weight[k] += rate_n * update

            bias += rate_n * error
            tss += error * error

        if i > miniterations and tss < tss_tolerance:
            print( "weights converged (%f < %f)" % (tss, tss_tolerance) )
            break
    print( "exit at iteration %d  tss=%f tolerance=%f" % (i, tss, tss_tolerance) )
        #print 'iteration', i, 'done.'
    return weight, bias 

