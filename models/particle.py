# Implementation of Anderson's rational model of Categorization:
# Local MAP: Anderson (1990) and Anderson (1991),
# It can be readily extended to be optimized by a particle filter or Gibbs sampler.
#
# Implemented by John McDonnell, 2010

import os
import numpy as np
import numpy.random as nprand
import random
from collections import Counter

# from itertools import izip
import scipy.stats.distributions as dist
import matplotlib.pyplot as plt
from typing import Tuple, List

# import pdb

VERBOSE = False

# Utility functions:


def tatval(df, mu, sigma, x):
    tdist = dist.t([df])
    return tdist.pdf((x - mu) / sigma)


def normatval(mu, sigma, x):
    normdist = dist.norm()
    return normdist.pdf((x - mu) / sigma)


def w_choice(lst):
    n = random.uniform(0, 1)
    for item, weight in lst:
        if n < weight:
            break
        n = n - weight
    return item


def stimsequal(stim1, stim2):
    return all([s[0] == s[1] for s in zip(stim1, stim2)])


# -----------------------------------------------------------------------------------
# RationalParticle()
# -----------------------------------------------------------------------------------
class RationalParticle:
    """
    A particle incrementally updates one hypothetical partitioning, integrating
    new observations as they are observed. A single particle run alone is
    essentially an implementation of Anderson's rational model. In a particle
    filter as described by Sanborn et al., many of these are run in parallel,
    and on a new observation the filter samples from the set of particles.

    Anderson used Maximum a Posteriori to assign stimuli to a partition, but we
    also include a softmax choice function, which is necessary for use in a
    particle filter (otherwise all particles would be identical).

    For information on Anderson's rational model, see Anderson (1990, 1991).
    For information on the particle filter model of categorization, see Sanborn
    et al. (2006)

    'Categories' were renamed 'clusters' in the documentation to avoid
    confusion with labels.

    This version only supports continuous-valued stimuli
    (but is probably close to supporting discrete values mixed in).
    Stimulus format is a list of floats.
    """

    def __init__(self, args, decision="Soft"):
        """
        args: c, mu0, sigma0, lambda0, a0, types, decision
            c: Coupling probability; controls likelihood of joining an existing
                        cluster
            mu0: prior on the means for each feature
            sigma0: prior on the variances for each feature
            lambda0: strength of prior on means (number of 'votes' the prior
                        gets)
            a0: strength of prior on variances (number of 'votes' the prior
                        gets)
            types: String giving the feature type for each feature.
              'd' means discrete ( currently not supported )
              'c' is continuous
              Example, 'cdc': Three features, first and last are continuous,
              middle is discrete.

        decision: Rule to use when deciding which cluster a stimulus will join.
          "Soft": joins a cluster with likelihood equal to posterior
                      likelihood. Default.
          "MAP": Always joins the maximum a posteriori cluster.
        """
        (
            self.c,
            self.mu0,
            self.sigma0sq,
            self.lambda0,
            self.a0,
            self.types,
            self.n_labels,
            self.feedback,
        ) = args
        self.alpha = np.ones(
            (len(self.types), 2)
        )  # a vector of two ones (assuming binary feature dims)
        self.alpha0 = sum(self.alpha.T)
        self.alpha_label = np.ones(self.n_labels)
        self.alpha0_label = np.array(self.n_labels * self.alpha_label)
        self.stims = []
        self.partition = []
        self.clusters = 0
        self.N = 0  # number of stimuli so far
        self.decision = decision
        self.label_probs = []

    def finddistribution(self, k, i, lambdai=None, ai=None, n=None):
        """
        For a given cluster computes the current mean and variance.
        """
        if lambdai is None and ai is None and n is None:  # compute in special case
            if k == self.clusters:  # Form new cluster?
                n = 0
            else:  # otherwise, n = total number of items in cluster
                n = sum([item == k for item in self.partition])

            # see Anderson, 1991 pg. 414 for this
            lambdai = self.lambda0[i] + n
            ai = self.a0[i] + n

        items = []
        for index in range(len(self.partition)):
            if self.partition[index] == k:
                items.append(self.stims[index][i])

        if n > 0:
            xbar = np.mean(items)
        else:
            xbar = 0
        if n > 1:
            var = np.var(items)
        else:
            var = 0

        mui = (self.lambda0[i] * self.mu0[i] + n * xbar) / float(ai)
        if n == 0:
            sigmaisq = self.sigma0sq[i]
        else:
            sigmaisq = (
                self.a0[i] * self.sigma0sq[i]
                + (n - 1.0) * var
                + (self.lambda0[i] * n / lambdai) * (self.mu0[i] - xbar) ** 2
            ) / float(ai)
        return mui, sigmaisq

    def probDensity(self, k, i, x):
        """
        For continuous-valued features, computes the probability of a feature value given
        a cluster membership.
        """
        if k == self.clusters:  # Form new cluster?
            n = 0
        else:
            n = sum([item == k for item in self.partition])

        # see Anderson, 1991 pg. 414 for this
        lambdai = self.lambda0[i] + n
        ai = self.a0[i] + n
        mui, sigmaisq = self.finddistribution(k, i, lambdai, ai, n)

        ret = tatval(ai, mui, np.sqrt(sigmaisq * (1.0 + (1.0 / lambdai))), x)
        return ret

    def probClustVal(self, k, i, j):
        """
        For discrete-valued features, computes the probability of a feature
        value given a cluster membership (P(j|k)).
        """
        if k == self.clusters:  # Form new cluster?
            cj = 0
            nk = 0
        else:
            cj = 0
            nk = 0
            for index in range(len(self.partition)):
                if self.partition[index] == k:
                    nk += 1
                    if self.stims[index][i] == j:
                        cj += 1

        pjk = (float(cj) + self.alpha[i][j]) / float((float(nk) + self.alpha0[i]))
        return pjk

    def valprob(self, k, i, val):
        """
        Given a cluster, a stimulus dimension, and a value along that dimension, compute the probability of occurence.
        """
        if self.types[i] == "c":  # if continuous, use t-distribution
            return self.probDensity(k, i, val)
        elif self.types[i] == "d":  # if discrete use binomial
            return self.probClustVal(k, i, val)
        else:
            raise Exception("Unrecognized dimension type.")

    def stimprob(self, stim, k, env=None):
        """
        Find P(F|k).
        Given a cluster (k), find the probability that the stimulus belongs to
        this cluster. This is done by breaking it down into feature values and
        treating their likelihood as independent.
        """
        pjks = []
        for i in range(len(self.stims[stim])):
            if env:  # consider only specifically "known" dimensions
                if env[i] == "k":
                    pjks.append(self.valprob(k, i, self.stims[stim][i]))
            else:  # assume all dimensions are known
                pjks.append(self.valprob(k, i, self.stims[stim][i]))

        # print "p(j|k): ", pjks
        # print "product: ", np.product(pjks)

        # assumes features are independent, and just multiplies to get current prob.
        return np.product(pjks)

    def labelprob(self):
        """Calculate conditional label probability given cluster

        return:
            pjk for the label
        """
        # we already know the cluster probs, but the feedback has not yet been presented
        unique_partitions = np.unique(self.partition)
        n_partitions = len(unique_partitions)
        n_unique_feedback = len(np.unique(self.feedback[0 : self.N]))
        print("n_partitions = ", n_partitions)
        print("unique partitions =", unique_partitions)
        if self.N == 1:
            counts_formatted = np.repeat(0, self.n_labels)
            objects_per_cluster = np.repeat(0, self.n_labels)
        else:
            lvls, counts = crosstab(
                self.partition[0 : (self.N - 1)], self.feedback[0 : (self.N - 1)]
            )

            counts_formatted = np.reshape(
                np.repeat(0, (len(np.unique(self.partition)) + 1) * self.n_labels),
                (n_partitions + 1, self.n_labels),
            )
            if n_unique_feedback < self.n_labels:
                for p in lvls[0]:
                    idx = 0
                    for l in lvls[1]:
                        counts_formatted[p, l - 1] += counts[p, idx]
                        idx += 1
            else:
                counts_formatted = counts
            objects_per_cluster = np.reshape(
                np.repeat(counts_formatted.sum(1), self.n_labels),
                (n_partitions + 1, self.n_labels),
            )

        # pjk for the label, therefore plk
        plk = (counts_formatted + self.alpha_label) / (
            objects_per_cluster + self.alpha0_label
        )
        return plk

    def computeposterior(self, stim, env=None, calcclust=False):
        """
        Take a given item and find the probability it belongs to each existing
        cluster (or a new cluster), p(k|f).
        """
        pk = np.empty(self.clusters + 1)
        pfk = np.empty(self.clusters + 1)

        for k in range(self.clusters):
            # This loops through the clusters, finding prior and conditional

            # Prior:
            pk[k] = (
                self.c
                * float(sum([item == k for item in self.partition]))
                / ((1.0 - self.c) + self.c * float(self.N))
            )

            # for given cluster, compute likelihood of current stimulus
            # pfk[k] = self.stimprob( stim, k )
            # Conditional:
            pfk[k] = self.stimprob(stim, k, env)

        # Prior / likelihood for new cluster
        pk[self.clusters] = (1.0 - self.c) / float(
            (1.0 - self.c) + self.c * float(self.N)
        )
        pfk[self.clusters] = self.stimprob(stim, self.clusters)

        # put it together
        num = pk * pfk
        denom = sum(num)
        if denom > 0:
            pkf = num / denom
        else:
            # raise Exception, "Could not find posterior"
            # I think this will only arise relatively harmlessly, when there's
            # been a numerical error"
            pkf = np.ones(len(pk)) / float(len(pk))

        # Calculate it just for clusters too, this is for filter sampling.
        if calcclust:
            clustnum = pk[: self.clusters] * pfk[: self.clusters]
            clustdenom = sum(clustnum)
            if clustdenom > 0:
                clustpkf = np.dot(pk, pfk)
            else:
                # raise Exception, "Could not find posterior"
                # I think this will only arise relatively harmlessly, when there's
                # been a numerical error"
                clustpkf = np.ones(len(pk)) / float(len(pk))

        if VERBOSE:
            print("p(k)s: ", pk)
            print("p(f|k)s: ", pfk)
            print("p(k|f): ", pkf)

        self.currentposterior = pkf
        if calcclust:
            self.clusterposterior = clustpkf
        self.laststim = self.stims[stim]
        if not calcclust:
            return pkf
        else:
            return clustpkf

    def getposterior(self, stim):
        """
        Gets the previously computed posterior. Avoids repeat calls, but is
        dangerous: the value could be old. It will raise an error if the
        identities have changed, but it's possible you could have the same stim
        twice in a row, which will not be caught.
        """
        # This probably means it was fresh, not a guaranteed check though.
        assert stimsequal(
            self.laststim, self.stims[stim]
        ), "getposterior() called before computeposterior()."
        return self.currentposterior

    def additemBayes(self, stim):
        """
        Adds a new item to the cluster according to Bayesian method (ie.,
        Sanborn's MORE Rational model).

        Softmax of P(k|F) + P(0|F)
        """
        if VERBOSE:
            print("Stim: ", self.stims[stim])
            print("Partition: ", self.partition)
            # print self.posterior(stim)

        # post = self.getposterior(stim)
        post = self.currentposterior

        needle = random.random()
        winner = int(np.sum(np.cumsum(post) < needle))
        if not winner in self.partition:
            self.clusters += 1
        self.partition[stim] = winner
        self.N += 1

    def additemMAP(self, stim):
        """
        Present an item, assigns it to the most likely cluster, and updates cluster.
        Argmax of P(k|F) + P(0|F)
        """

        # winner = np.argmax(self.getposterior(stim))
        winner = np.argmax(self.currentposterior)

        if VERBOSE:
            print("Stim: ", self.stims[stim])
            print(self.partition)  # "Partition: " +
            # print self.posterior(stim)

        if not winner in self.partition:
            self.clusters += 1
        self.partition[stim] = winner
        self.N += 1

    def register_item(self, stim, checkduplicate=False):
        """
        When the particle filter receives a new item, it  it appends it  to the
        list of stims. This function deals with that overhead.
        """
        stimnum = -1
        if checkduplicate:
            for i, oldstim in enumerate(self.stims):
                if stimsequal(stim, oldstim):
                    stimnum = i
                    break
        if stimnum == -1:
            self.stims.append(stim)
            self.partition.append(-1)
        return stimnum

    def additem_particle(self, stim, checkduplicate=False):
        """
        When the particle filter receives a new item, it appends it  to the
        list of stims. This function adds the item, registering it first.
        """
        stimnum = self.register_item(stim, checkduplicate=checkduplicate)
        # compute posterior over observed features
        self.computeposterior(stimnum)
        if self.decision == "Soft":
            self.additemBayes(stimnum)
        elif self.decision == "MAP":
            self.additemMAP(stimnum)
        else:
            raise Exception("Invalid decision rule. Valid values are 'Soft' and 'MAP'.")
        # make a prediction on the label for the currently observed stimulus
        plk = self.labelprob()
        print("plk = ", plk)
        print("posterior over clusters = ", self.currentposterior)
        K = len(self.currentposterior)
        plf = np.reshape(self.currentposterior, (1, K)) @ np.reshape(
            plk, (K, self.n_labels)
        )

        self.label_probs.append(plf)

    def findMAPval(self, stimulus, env):
        """Queried value should be -1."""
        qdim = [x[1] for x in zip(env, range(len(env))) if x[0] == "?"]
        if len(qdim) > 1:
            raise Exception("ERROR: Multiple dimensions queried.")
        if len(qdim) == 0:
            raise Exception("ERROR: No dimensions queried.")
        qdim = qdim[0]

        stimnum = -1
        for i, s in enumerate(self.stims):
            if stimsequal(stimulus, s):
                stimnum = i
                break
        if stimnum == -1:
            self.stims.append(
                stimulus
            )  # WARNING: this is a hackish solution, should be refactored.

        pkF = self.computeposterior(stimnum, env)
        pkF = pkF[:-1] / sum(pkF[:-1])  # eliminate `new cluster' prob

        pjF = []
        if self.types[qdim] == "c":
            for k in range(len(self.partition) - 1):
                mui = self.finddistribution(k, qdim)[0]
                pjF.append(mui * pkF[k])
        elif self.types[qdim] == "d":
            for k in range(len(np.unique(self.partition))):
                pjF.append(
                    np.argmax([self.valprob(k, qdim, j) for j in range(2)]) * pkF[k]
                )

        # if sum(pjF)==0:
        #    return 500
        # else:
        if stimnum == -1:
            del self.stims[-1]  # (reversing the stim-adding hack)
        return sum(pjF)


def plotsolution(post):
    import pylab

    X = np.unique(post[0])
    Y = np.unique(post[1])
    Z = np.zeros((len(X), len(Y)))
    sparseZ = dict(zip(map(tuple, post[:2].T), post[2]))
    for x in range(len(X)):
        for y in range(len(Y)):
            if (X[x], Y[y]) in sparseZ.keys():
                Z[x, y] = sparseZ[(X[x], Y[y])]
    pylab.matshow(Z)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# basic tests
# reference crosstab: https://asifr.com/groupby-count-numpy
def crosstab(*args) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """
    Contingency table of counts.

    Parameters
    ----------
    args : list of array-like
        Arrays of discrete categorical data.

    Returns
    -------
    actual_levels : Tuple[np.ndarray]
        The actual levels of the categorical variables.
    count : np.ndarray
        The counts of the categorical variables cross-tabulated.
    """
    levels, indices = zip(*[np.unique(a, return_inverse=True) for a in args])
    count = np.zeros(list(map(len, levels)), dtype=int)
    np.add.at(count, indices, 1)
    return levels, count


def get_zmstims(n):
    """
    Stimuli in Z&M 2009 were lines w/ varying length and angle.

    n needs to be even because there are two groups of items being combined.

    First dim is size, second is angle.
    For this example, we use only the size-bimodal version.
    """
    assert n % 2 == 0, "Size for the Z&M stims must be even (%d given)." % n
    Mu1 = [187.5, 45]
    Mu2 = [412.5, 45]
    Cov = np.multiply(np.eye(2), [12.5, 15])

    samples = np.vstack(
        [
            nprand.multivariate_normal(Mu1, Cov, int(n / 2)),
            nprand.multivariate_normal(Mu2, Cov, int(n / 2)),
        ]
    )
    # nprand.shuffle( samples ) # not shuffling for purposes of demo, stims should end up split in half.
    labels = np.hstack([np.zeros(int(n / 2)), np.ones(int(n / 2))])
    withlabels = np.column_stack([samples, labels])
    nprand.shuffle(withlabels)
    items = withlabels[:, :-1]
    labels = withlabels[:, -1]
    return items, labels


def testcontinuous():
    """
    This test plots six partitions made using a single particle each.
    """
    # stims = np.loadtxt( os.popen("awk 'NF==13 {print( $7, $8, $10 )}' data/3.dat") )
    # types = 'ccc'
    outfile = "/dev/null"
    plt.suptitle("Six partitions made using a single particle each.")
    for run in range(6):
        stims, labels = get_zmstims(400)
        types = "cc"

        Cparam = 0.65
        mu0 = np.mean(stims, 0)
        sigma0 = np.var(stims, 0)
        lambda0 = np.ones(len(stims[0]))
        a0 = np.ones(len(stims[0]))

        args = [Cparam, mu0, sigma0, lambda0, a0, types]

        # args[-2][-1] = .1
        # args[-1][-1] = .1
        model = RationalParticle(args)

        stimset = stims  # [:64]
        # shuffle(stimset)
        for s in stimset:
            model.additem_particle(s)
        if VERBOSE:
            print(model.partition)
            print(len(model.partition) - 1, " cluster(s) formed.")

        order = np.lexsort((model.partition, labels))
        cols = np.column_stack([labels, model.partition])
        print("Actual labels (left) vs. model partition (right)")
        print(cols[order])

        # Plotting the outcome:
        for clust in range(model.clusters):
            thesepoints = []
            for i in range(len(model.partition)):
                if model.partition[i] == clust:
                    thesepoints.append(model.stims[i])
            thesepoints = np.array(thesepoints)
            plt.subplot(2, 3, run + 1)
            plt.plot(thesepoints[:, 0], thesepoints[:, 1], "o")

        # env = 'kk?'
        # X = []; Y = []; Z = []
        # X,Y = stims[:64,:2].T
        # X = s.T[0]
        # Y = s.T[1]
        # Z = [ model.findMAPval( stim, env) for stim in np.vstack((X,Y)).T ]
        # for s in stims[:64]:
        #    X.append(s[0])
        #    Y.append(s[1])
        #    Z.append( model.findMAPval(s, env))

        # plotsolution( np.vstack( (X,Y,Z) ) )
        # np.savetxt( outfile, (X,Y,Z))

        # print "Prob vals for ", stims[0][:-1], ": ", model.query(stims[0][:-1] + [-1])
    plt.show()


def test_anderson_discrete():
    """
    Tests the Anderson's ratinal model using the Medin & Schaffer (1978) data.

    This script will print out the probability that each item belongs to each
    of the existing clusters or to a new cluster, and the model assign it to
    the most likely cluster. To see that the model is working correctly, you
    can follow along with Anderson (1991), which steps through in the same way.

    The classic Shepard tasks can also be commented out and run.
    """

    stims = [
        [1, 1, 1, 1, 1],  # Medin & Schaffer (1978)
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 0, 0],
    ]
    types = "ddddd"

    # Below are the classic Shepard Type II and Type IV datasets.  Uncomment
    # the one you want to try out; you might want to uncomment shuffling the
    # stims too if you don't care about order.
    # stims = ['0000', '0010', '1101', '1111', '1000', '1011', '0100', '0111'] # Type IV
    # stims = ['0000', '0010', '1100', '1110', '1001', '1011', '0101', '0111'] # Type II
    # stims = [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 1, 1]] # Type IV
    # stims = [
    #     [0, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [1, 1, 0, 0],
    #     [1, 1, 1, 0],
    #     [1, 0, 0, 1],
    #     [1, 0, 1, 1],
    #     [0, 1, 0, 1],
    #     [0, 1, 1, 1],
    # ]  # Type II
    # predict the label
    stims = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
    ]  # Type II
    types = "ddd"
    feedback = [1, 1, 1, 1, 2, 2, 2, 2]

    for _ in range(1):
        args = [
            0.5,
            np.mean(stims, 0),
            np.var(stims, 0),
            np.ones(len(stims[0])),
            np.ones(len(stims[0])),
            types,
            2,
            feedback,
        ]
        model = RationalParticle(args, decision="MAP")

        # random.shuffle(stims)
        for s in stims:
            model.additem_particle(s)
        print(model.partition)

        query = [0] * (len(stims[0]) - 1) + [-1]
        print("Prob vals for ", query)
        print(model.findMAPval(query, "k" * (len(stims[0]) - 1) + "?"))


def main():
    # testcontinuous()
    test_anderson_discrete()


# if __name__ != '__main__':
#    VERBOSE = False

if __name__ == "__main__":
    VERBOSE = True
    main()
