# Gibbs Sampler.
#
# Implemented in python by John McDonnell, 2010
# Many thanks to Sanborn et al. for developing it and sharing their Matlab implementation.


import os
import random
import numpy as np
import numpy.random as nprand
from itertools import izip
import scipy.stats.distributions as dist
import particle

#import pylab
#from collections import deque

#Utility functions:

argmax = lambda array: max(izip(array, xrange(len(array))))[1]

def tatval(df, mu, sigma, x):
    tdist = dist.t([df])
    return tdist.pdf((x-mu)/sigma)

def normatval(mu, sigma, x):
    normdist = dist.norm()
    return normdist.pdf((x-mu)/sigma)

def w_choice(lst):
    n = random.uniform(0, 1)
    for item, weight in lst:
        if n < weight:
            break
        n = n - weight
    return item


class GibbsSamplerPart(particle.RationalParticle):
    def __init__(self, stims, args, sampleargs, initpartition):
        """
        Overrides RationalParticle intialization.
        
        stims: list of lists or nx2 array.
        args: same as in RationalParticle:
            c, mu0, sigma0, lambda0, a0, types, decision
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
        
        sampleargs have to do with the sampling method:
            nsamples: how many samples you want to end up with
            spacing: how many runs to throw out between samples
            burnin: how many runs to do before recording of samples begins.
        
        initpartition:
            The initial partition of items. Partitions must be identified as a
            set of ascending integers beginning with 0. Must be the same length
            as the items. Often an array of zeros is used initially.
        """
        self.c, self.mu0, self.sigma0sq, self.lambda0, self.a0, self.types = args
        self.alpha = np.ones((len(self.types), 2)) # a vector of two ones 
        self.alpha0 = sum(self.alpha.T)
        
        self.nsamples, self.spacing, self.burnin = sampleargs
        self.stims = stims
        self.partition = initpartition
        self.N = len( self.partition )
        self.clusters = len(np.unique(self.partition))
        assert all( [ clust in np.unique(self.partition) for clust in range(self.clusters)] ), "Violation of cluster naming convention."
        assert len(initpartition) == len(stims), \
                "initpartition has the wrong number of items: its length is \
                %d, there are %d items." % ( len(initpartition), len(stims)  )
    
    def sample(self, outfn):
        totalsamples = self.nsamples * self.spacing + self.burnin
        sofar = self.partition
        
        ofile = open( outfn, 'w' )
        
        for samp in range( totalsamples ):
            for stim in range(len(self.stims)):
                oldpartition = self.partition.copy()
                temppartition = oldpartition.copy()
                temppartition[stim] = -1 #np.nan
                self.N-=1
                if not self.partition[ stim ] in temppartition:
                    self.clusters-=1
                    for i in range(len(temppartition)):
                        if temppartition[i] > self.partition[stim]:
                            temppartition[i]-=1
                self.partition = temppartition
                
                post = self.computeposterior(stim)
                
                self.additemBayes( stim )
                if VERBOSE:
                    print "Iteration %d" % samp
             
            if samp > self.burnin and (samp - self.burnin)% self.spacing == 0:
             
                print self.partition
                print "Sample: ", (samp-self.burnin) // self.spacing
                ofile.write( str( self.partition )[1:-1] )
                ofile.flush()
        
        ofile.close()
        return


class GibbsSampler:
    def __init__(self, stims, args, sampleargs, initpartition):
        self.c, self.mu0, self.sigma0sq, self.lambda0, self.a0, self.types = args
        self.alpha = np.ones((len(self.types), 2)) # a vector of two ones 
        self.alpha0 = sum(self.alpha.T)
        
        self.nsamples, self.spacing, self.burnin = sampleargs
        self.stims = stims
        self.partition = initpartition
        self.N = len( self.partition )
        self.clusters = len(np.unique(self.partition))
        assert all( [ clust in np.unique(self.partition) for clust in range(self.clusters)] ), "Violation of cluster naming convention."
    
    def finddistribution(self, k, i, lambdai=None, ai=None, n=None):
        """
        For a given cluster computes the current mean and variance.
        """
        if lambdai is None and ai is None and n is None: # compute in special case
            if k==self.clusters: #Form new cluster?
                n = 0
            else: # otherwise, n = total number of items in cluster
                n = sum(self.partition==k)
        
            # see Anderson, 1991 pg. 414 for this
            lambdai = self.lambda0[i] + n 
            ai = self.a0[i] + n
        
        items = []
        for index in range(len(self.partition)):
            if self.partition[index] == k:
                items.append( self.stims[index][i] )
        
        xbar = np.mean( items )
        var = np.var( items )
        if n <= 0:
            xbar = 0
        if n <= 1:
            var = 0 
        
        mui = (self.lambda0[i]*self.mu0[i] + n * xbar) / float(ai)
        if n == 0:
            sigmaisq=self.sigma0sq[i]
        else:
            sigmaisq = ( self.a0[i]*self.sigma0sq[i] + (n-1.0) * var \
                + (self.lambda0[i]*n/lambdai) * (self.mu0[i] - xbar)**2 ) / float(ai)
        return mui, sigmaisq
    
    def probDensity(self, k, i, x):
        """
        Find f_i ( x | k ) (the probability density for a given feature value given a cluster (continuous case).
        
        """
        if k==self.clusters: #Form new cluster?
            n = 0
        else: # otherwise, n = total number of items in cluster
            n = sum(self.partition==k)
        
        # see Anderson, 1991 pg. 414 for this
        lambdai = self.lambda0[i] + n 
        ai = self.a0[i] + n
        mui, sigmaisq = self.finddistribution( k, i, lambdai, ai, n)
        ret = tatval( ai, mui, np.sqrt( sigmaisq * (1.0 + (1.0/lambdai))), x ) # give the value of the t-distribution at a particular val.
        return ret
    
    def probClustVal(self, k, i, j):
        """
        Find P(j|k), that is, the probability of a given feature value given a
        cluster (discrete case).
        """
        if k==self.clusters: #Form new cluster?
            cj = 0
            nk = 0
        else:
            cj = 0
            nk = 0
            for index in range(len(self.partition)):
                if self.partition[index] == k:
                    nk+=1
                    if self.stims[index][i] == j:
                        cj+=1
        
        pjk = (float(cj) + self.alpha[i][j]) / \
                float((float(nk) + self.alpha0[i]))
        return pjk
    
    def valprob(self, k, i, val):
        """
        Given a value, cluster, and dimension compute the probability of this value along this dimension in the cluster.
        """
        if self.types[i] == 'c': # if continuous, use t-distribution
            return self.probDensity(k, i, val)
        elif self.types[i] == 'd': # if discrete use binomial or whatever
            return self.probClustVal(k, i, val)
        else:
            raise Exception, "Unrecognized dimension type."
    
    def stimprob(self, stim, k, env=None ):
        """
        Find P(F|k). 
        Given a cluster (k), find the probability that the stimulus belongs to this cluster.
        This is done by breaking it down into feature values and treating their likelihood as independent.
        """
        pjks = []
        for i in range(len(self.stims[stim])):
            if env: # consider only specifically "known" dimensions
                if env[i] == 'k':
                     pjks.append( self.valprob( k, i, self.stims[stim][i]) )
            else: # assume all dimensions are known
                pjks.append( self.valprob( k, i, self.stims[stim][i]) )
        
        # print "p(j|k): ", pjks
        # print "product: ", np.product(pjks)
        return np.product( pjks )  # assumes features are independent, and just multiplies to get current prob.
    
    def computeposterior( self, stim ):
        """
        Take a given item and find the probability it belongs to each existing cluster (or a new cluster)
        """
        pk  = np.empty( self.clusters+1 )
        pfk = np.empty( self.clusters+1 )
        
        # existing clusters:
        for k in range(self.clusters):
            # This loops through the clusters, finding prior and conditional
            
            # Prior, from Dirichlet process:
            
            pk[k] = self.c * float(sum(self.partition==k)) / ((1.0-self.c) +
                                                              self.c *
                                                              float(self.N))
            
            # Conditional:
            pfk[k] = self.stimprob( stim, k ) # for given cluster, compute likelihood of current stimulus
        
        pk[self.clusters] = (1.0-self.c) / float(( 1.0-self.c ) + self.c * float(self.N))
        pfk[self.clusters] = self.stimprob( stim, self.clusters ) 
        
        # put it together
        if sum(pk*pfk) > 0:
            pkf = (pk*pfk) / float(sum( pk*pfk ))
        else:
            raise "Could not compute posterior"
            #pkf = np.ones(len(pk))/float(len(pk))  # 1/# of clusters ... shouldn't this never happen?
        
        if VERBOSE:
            print "p(k)s: ", pk
            print "p(f|k)s: ", pfk
            print "p(k|f): ", pkf
        
        self.currentposterior = pkf
        self.laststim = stim
        return pkf
    
    def getposterior (self, stim):
        """
        Gets the previously computed posterior. Avoids repeat calls, but is dangerous: the value
        could be old. It will raise an error if the identities have changed, but it's possible
        you could have the same stim twice in a row, which will not be caught.
        """
        # This probably means it was fresh, not a guaranteed check though.
        assert (self.laststim == stim), "getposterior() called before computeposterior()."
        return self.currentposterior
    
    def additemBayes(self, stim):
        """
        Present an item, assigns it to a cluster based on likelihood, and updates cluster.
        Softmax of P(k|F) + P(0|F)
        """
        if VERBOSE:
            print "Stim: ", stim
            print "Partition: ",  self.partition
            #print self.posterior(stim)
        
        # choose a sample at random from this distribution and add the cluster here
        post = self.getposterior(stim)
        
        needle = random.random()
        winner = int(np.sum(np.cumsum(post)<needle))
        if not winner in self.partition:
            self.clusters += 1
        self.partition[ stim ] = winner
        self.N += 1
    
    def additemMAP(self, stim):
        """
        Present an item, assigns it to the most likely cluster, and updates cluster.
        Argmax of P(k|F) + P(0|F)
        """
        winner = argmax( self.getposterior() )
        if VERBOSE:
            print "Stim: ", stim
            print "Partition: ", self.partition
            print self.posterior(stim)
        
        if not winner in self.partition:
            self.clusters += 1
        self.partition[ stim ] = winner
        self.N +=1
     
    def sample(self, outfn):
        totalsamples = self.nsamples * self.spacing + self.burnin
        sofar = self.partition
        
        ofile = open( outfn, 'w' )
        
        for samp in range( totalsamples ):
            for stim in range(len(self.stims)):
                oldpartition = self.partition.copy()
                temppartition = oldpartition.copy()
                temppartition[stim] = -1 #np.nan
                self.N-=1
                if not self.partition[ stim ] in temppartition:
                    self.clusters-=1
                    for i in range(len(temppartition)):
                        if temppartition[i] > self.partition[stim]:
                            temppartition[i]-=1
                self.partition = temppartition
                
                post = self.computeposterior(stim)
                
                self.additemBayes( stim )
                if VERBOSE:
                    print "Iteration %d" % samp
             
            if samp > self.burnin and (samp - self.burnin)% self.spacing == 0:
             
                print self.partition
                print "Sample: ", (samp-self.burnin) // self.spacing
                ofile.write( str( self.partition )[1:-1] )
                ofile.flush()
        
        ofile.close()
        return


def AnalyzeExp():
    files = [os.path.join('data', fn) for fn in os.listdir('data') if fn[-4:]==".dat" ]
    types = 'ccc'
    
    nsamples = 50
    spacing = 10
    burnin = 200
    sampleargs = [nsamples, spacing, burnin]
    
    for filename in files:
        print filename
        cond = int(open( filename ).readlines()[1].split()[1])
        if cond == 0: # ignore 'FR's
            continue
        subj = int( os.path.basename(filename)[:-4] )
        for respcol in [10]: # 9 is participant response, 10 is 'reinforced' response
            stims = np.loadtxt( os.popen("awk 'NF==13 {print( $7, $8, $%i )}' %s" % (respcol, filename)) )[160:]
            condition = open(filename)
            
            cparam = .3
            mu0 = np.mean( stims, 0 )
            sigma0 = np.var( stims, 0 )*.5
            lambda0 = np.ones( len(stims[0]) )
            a0 = np.ones( len(stims[0]) )*.5
            args = [ cparam, mu0, sigma0, lambda0, a0, types ]
            
            initpartition = np.zeros(len(stims), dtype="int")   # This puts everything in one cluster named 0.
            sampler = GibbsSampler( stims, args, sampleargs, initpartition )
            if respcol == 9:
                outfn = "gibbsdataMarch10/responses_subj%d_cond%d.dat" % ( subj, cond )
            elif respcol == 10:
                outfn = "gibbsdataMarch10/answers_subj%d_cond%d.dat" % ( subj, cond )
            sampler.sample( outfn )


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

def demo():
    #stims = np.loadtxt( os.popen("awk 'NF==13 {print( $7, $8, $10 )}' data/2.dat") )[:64]
    #types='ccc'
    outfile = "/dev/null"
    stims, labels = get_zmstims(100)
    types = 'cc'
    
    nsamples = 100
    spacing = 5
    burnin = 150
    
    Cparam = .25
    mu0 = np.mean( stims, 0 )
    sigma0 = np.var( stims, 0 )
    lambda0 = np.ones( len(stims[0]) )
    a0 = np.ones( len(stims[0]) )
    
    args = [Cparam, mu0, sigma0, a0, lambda0, types]
    
    sampleargs = [nsamples, spacing, burnin]
    
    # Put everything in one cluster named 0.
    initpartition = np.zeros(len(stims))   
    
    #sampler = GibbsSampler( stims, args, sampleargs, initpartition )
    sampler = GibbsSamplerPart( stims, args, sampleargs, initpartition )
    partitions = sampler.sample( outfile )

def main():
    demo()

if __name__ == "__main__":
    VERBOSE = True
    main()
