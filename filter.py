# Particle filter, posterior approximation method for the Anderson rational model.
#
# Implemented in python by John McDonnell, 2010
# Many thanks to Sanborn et al. for developing it and sharing their Matlab implementation.

from pylab import *
import numpy as np
import numpy.random as nprand
import copy
from random import randrange
import os

import particle as particle
from itertools import izip

argmax = lambda array: max(izip(array, xrange(len(array))))[1]


class particlefilter():
    """
    Implements the particle filter of sanborn et al. (2006)
    arguments: 
        args: particle filter arguments (see RationalParticle docstring)
        m:    number of clusters
    """
    def __init__(self, args, m):
        self.args = args
        self.m = m
        self.particles = [particle.RationalParticle(self.args, decision="Soft") for _ in xrange(self.m)]
    
    def findrespdist(self, stim, env):
        res= [ s.findmapval(stim, env) for s in self.particles]
        fulldist = np.mean (np.transpose(res), 0)
        return  fulldist
    
    def findmapval(self, stim, env):
        #print "my guess ", [ s.findmapval(stim, env) for s in self.particles] <- old way
        return np.mean([ s.findMAPval(stim, env) for s in self.particles])
    
    def getnclusters(self):
        #clusterdist = self.getclusterdist()
        print [s.getNClusters() for s in self.particles]
    
    def getclusterdist(self):
        maxclustertocount = 12
        res = [s.getNClusters() for s in self.particles]
        clustercount = np.zeros(maxclustertocount) 
        for line in res: 
            if line<maxclustertocount:
                clustercount[line]+=1
        return clustercount
    
    def getfullpartition(self):
        res = [s.partition for s in self.particles]
        return res
    
    def iterate(self, stim ):
        samples = [randrange(self.m) for _ in xrange(self.m) ]  # random sample (with replacement)
        samplesenum = [[i, samples.count(i)] for i in xrange(self.m)] # counts number of times each particle sampled
        newparticles = []
        for l in samplesenum:
            if l[1] > 0:
                # Computing posterior ahead of time:
                stimnum = self.particles[l[0]].register_item( stim )
                self.particles[l[0]].computeposterior( stimnum )
                for i in range(l[1]):
                    newparticles.append(copy.deepcopy(self.particles[l[0]]))
                    newparticles[-1].additemBayes( stimnum )
        self.particles = newparticles[:]

def get_zmstims(n):
    """
    Stimuli in Z&M 2009 were lines w/ varying length and angle.
    
    n needs to be even because there are two groups of items being combined.
    
    First dim is size, second is angle.
    For this example, we use only the size-bimodal version.
    """
    assert n%2 == 0, "Size for the Z&M stims must be even (%d given)." % n
    Mu1 = [187.5, 45]
    Mu2 = [412.5, 45]
    Cov = np.multiply( np.eye( 2 ), [12.5,15] )
    
    samples = np.vstack([nprand.multivariate_normal( Mu1, Cov, n/2 ),
                         nprand.multivariate_normal( Mu2, Cov, n/2 )])
    #nprand.shuffle( samples ) # not shuffling for purposes of demo, stims should end up split in half.
    labels = np.hstack([np.zeros(n/2),np.ones(n/2)])
    withlabels = np.column_stack([samples, labels ])
    nprand.shuffle( withlabels )
    items = withlabels[:,:-1]
    labels = withlabels[:,-1]
    return items, labels

def testparticle():
    #stims = loadtxt( os.popen("awk 'NF==13 {print( $7, $8, $10 )}' data/3.dat") )
    #types='ccc'
    outfile = "/dev/null"
    stims, labels = get_zmstims(100)
    types = 'cc'
    
    Cparam = .65
    mu0 = mean( stims, 0 )
    sigma0 = var( stims, 0 )
    lambda0 = ones( len(stims[0]) )
    a0 = ones( len(stims[0]) )
    
    args = [Cparam, mu0, sigma0, lambda0, a0, types]
    m = 6  # Just one particle for demonstration purposes.
    
    model = particlefilter(args, m)
    for s in stims:
        #model.findMAPval(stim, 'cc?')
        model.iterate(s)
        #model.getnclusters()
    
    import pylab as pl
    pl.suptitle( "Six particles from a single run." )
    partitions = model.getfullpartition()
    for particle in range( 6 ):
        thispartition = partitions[particle]
        order = np.lexsort(( thispartition, labels ))
        cols = np.column_stack([ labels, thispartition ])
        for clust in range( len(np.unique(thispartition)) ):
            thesepoints = []
            for i in xrange( len( thispartition ) ):
                if thispartition[i]==clust:
                    thesepoints.append( stims[i] )
            thesepoints = np.array( thesepoints )
            pl.subplot( 2, 3, particle+1 )
            pl.plot( thesepoints[:,0], thesepoints[:,1], 'o' )
        #print "Actual labels (left) vs. model partition (right)"
        #print cols[order]


def main():
    #testcontinuous()
    testparticle()

if __name__ == '__main__':
    main()


