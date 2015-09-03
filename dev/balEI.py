#import brian_no_units
from numpy import array
from brian import *
import time

from brian.library.IF import *
from brian.library.synapses import *

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import pickle
def minimal_example():
    # Neuron model parameters
    Vr = -70 * mV
    Vt = -55 * mV
    taum = 10 * ms
    taupsp = 0.325 * ms
    weight = 4.86 * mV
    # Neuron model
    equations = Equations('''
        dV/dt = (-(V-Vr)+x)*(1./taum)                            : volt
        dx/dt = (-x+y)*(1./taupsp)                               : volt
        dy/dt = -y*(1./taupsp)+25.27*mV/ms+(39.24*mV/ms**0.5)*xi : volt
        ''')

    # Neuron groups
    P = NeuronGroup(N=2000, model=equations,
                  threshold=Vt, reset=Vr, refractory=1 * ms)
#    P = NeuronGroup(N=1000, model=(dV,dx,dy),init=(0*volt,0*volt,0*volt),
#                  threshold=Vt,reset=Vr,refractory=1*ms)

    Pinput = PulsePacket(t=10 * ms, n=85, sigma=1 * ms)
    # The network structure
    Pgp = [ P.subgroup(100) for i in range(20)]
    C = Connection(P, P, 'y',delay=5*ms)
    for i in range(19):
        C.connect_full(Pgp[i], Pgp[i + 1], weight)
    Cinput = Connection(Pinput, P, 'y')
    Cinput.connect_full(Pinput, Pgp[0], weight)
    # Record the spikes
    Mgp = [SpikeMonitor(p, record=True) for p in Pgp]
    Minput = SpikeMonitor(Pinput, record=True)
    monitors = [Minput] + Mgp
    # Setup the network, and run it
    P.V = Vr + rand(len(P)) * (Vt - Vr)
    run(90 * ms)
    # Plot result
    raster_plot(showgrouplines=True, *monitors)
    show()


# DEFAULT PARAMATERS FOR SYNFIRE CHAIN
# Approximates those in Diesman et al. 1999
model_params = Parameters(
    # Simulation parameters
    dt=0.1 * ms,
    duration=90 * ms,
    # Neuron model parameters
    taum=10 * ms,
    taupsp=0.325 * ms,
    Vt= -55 * mV,
    Vr= -70 * mV,
    abs_refrac=1 * ms,
    we=34.7143,
    wi= -34.7143,
    psp_peak=0.14 * mV,
    # Noise parameters
    noise_neurons=20000,
    noise_exc=0.88,
    noise_inh=0.12,
    noise_exc_rate=2 * Hz,
    noise_inh_rate=12.5 * Hz,
    computed_model_parameters="""
    noise_mu = noise_neurons * (noise_exc * noise_exc_rate - noise_inh * noise_inh_rate ) * psp_peak * we
    noise_sigma = (noise_neurons * (noise_exc * noise_exc_rate + noise_inh * noise_inh_rate ))**.5 * psp_peak * we
    """
    )

# MODEL FOR SYNFIRE CHAIN
# Excitatory PSPs only
def Model(p):
    equations = Equations('''
        dV/dt = (-(V-p.Vr)+x)*(1./p.taum)                          : volt
        dx/dt = (-x+y)*(1./p.taupsp)                               : volt
        dy/dt = -y*(1./p.taupsp)+25.27*mV/ms+(39.24*mV/ms**0.5)*xi : volt
        ''')
    return Parameters(model=equations, threshold=p.Vt, reset=p.Vr, refractory=p.abs_refrac)

default_params = Parameters(
    # Network parameters
    num_layers=10,
    neurons_per_layer=125, #change this to obtain figure 4(a:80,b:90,c:100,d:110)
    neurons_in_input_layer=100,
    # Initiating burst parameters
    initial_burst_t=50 * ms,
    initial_burst_a=85,
    initial_burst_sigma=1 * ms,
    # these values are recomputed whenever another value changes
    computed_network_parameters="""
    total_neurons = neurons_per_layer * num_layers
    """,
    # plus we also use the default model parameters
    ** model_params
    )

# DEFAULT NETWORK STRUCTURE
# Single input layer, multiple chained layers
class DefaultNetwork(Network):
    def __init__(self, p):
        # define groups
        chaingroup = NeuronGroup(N=p.total_neurons, **Model(p))
        inputgroup = PulsePacket(p.initial_burst_t, p.neurons_in_input_layer, p.initial_burst_sigma)
        layer_E = [ chaingroup.subgroup(int(p.neurons_per_layer * 0.88)) for i in range(p.num_layers) ]
        layer_I = [ chaingroup.subgroup(int(p.neurons_per_layer * 0.12)) for i in range(p.num_layers) ]
        # connections
        chainconnect = Connection(chaingroup, chaingroup, 2,delay=5*ms)
        for i in range(p.num_layers - 1):
            chainconnect.connect_full(layer_E[i], layer_E[i + 1], p.psp_peak * p.we)
            chainconnect.connect_full(layer_E[i], layer_I[i + 1], p.psp_peak * p.we)
            chainconnect.connect_full(layer_I[i], layer_E[i + 1], p.psp_peak * p.wi)
            chainconnect.connect_full(layer_I[i], layer_I[i + 1], p.psp_peak * p.wi)        
        inputconnect = Connection(inputgroup, chaingroup, 2)
        inputconnect.connect_full(inputgroup, layer_E[0], p.psp_peak * p.we)
        inputconnect.connect_full(inputgroup, layer_I[0], p.psp_peak * p.we)
        # monitors
        chainmon_E = [SpikeMonitor(g, True) for g in layer_E]
        chainmon_I = [SpikeMonitor(g, True) for g in layer_I]
        inputmon = SpikeMonitor(inputgroup, True)
        mon_E = [inputmon] + chainmon_E
        mon_I = [inputmon] + chainmon_I
        # network
        Network.__init__(self, chaingroup, inputgroup, chainconnect, inputconnect, mon_E, mon_I)
        # add additional attributes to self
        self.mon_E = mon_E
        self.mon_I = mon_I
        self.inputgroup = inputgroup
        self.chaingroup = chaingroup
        self.layer_E = layer_E
        self.layer_I = layer_I
        self.params = p

    def prepare(self):
        Network.prepare(self)
        self.reinit()

    def reinit(self, p=None):
        Network.reinit(self)
        q = self.params
        if p is None: p = q
        self.inputgroup.generate(p.initial_burst_t, p.initial_burst_a, p.initial_burst_sigma)
        self.chaingroup.V = q.Vr + rand(len(self.chaingroup)) * (q.Vt - q.Vr)

    def run(self):
        Network.run(self, self.params.duration)

    def plot(self):
        raster_plot(ylabel="Layer", title="Synfire chain raster plot",
                   color=(1, 0, 0), markersize=3,
                   showgrouplines=False, spacebetweengroups=0.2, grouplinecol=(0.5, 0.5, 0.5),
                   *self.mon_E)
        show()
        raster_plot(ylabel="Layer", title="Synfire chain raster plot",
                   color=(0, 1, 0), markersize=3,
                   showgrouplines=False, spacebetweengroups=0.2, grouplinecol=(0.5, 0.5, 0.5),
                   *self.mon_I)

def estimate_params(mon, time_est):
    # Quick and dirty algorithm for the moment, for a more decent algorithm
    # use leastsq algorithm from scipy.optimize.minpack to fit const+Gaussian
    # http://www.scipy.org/doc/api_docs/SciPy.optimize.minpack.html#leastsq
    i, times = zip(*mon.spikes)
    times = array(times)
    times = times[abs(times - time_est) < 15 * ms]
    if len(times) == 0:
        return (0, 0 * ms)
    better_time_est = times.mean()
    times = times[abs(times - time_est) < 5 * ms]
    if len(times) == 0:
        return (0, 0 * ms)
    return (len(times), times.std())

def single_sfc():
    net = DefaultNetwork(default_params)
    net.run()
    net.plot()

def state_space(grid, neuron_multiply, verbose=True):
    amin = 0
    amax = 100
    sigmamin = 0. * ms
    sigmamax = 4. * ms

    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = int(params.neurons_per_layer * neuron_multiply)

    net = DefaultNetwork(params)

    i = 0
    # uncomment these 2 lines for TeX labels
    #import pylab
    #pylab.rc_params.update({'text.usetex': True})
    if verbose:
        print "Completed:"
    start_time = time.time()
    figure()
    for ai in range(grid + 1):
        for sigmai in range(grid + 1): #
            a = int(amin + (ai * (amax - amin)) / grid)
            if a > amax: a = amax
            sigma = sigmamin + sigmai * (sigmamax - sigmamin) / grid
            params.initial_burst_a, params.initial_burst_sigma = a, sigma
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon_E[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
            col = (float(ai) / float(grid), float(sigmai) / float(grid), 0.5)
            plot([sigma / ms, newsigma / ms], [a, newa], color=[0,0,0])
            plot([sigma / ms], [a], marker='.', color=[0,0,0], markersize=15)
            i += 1
            if verbose:
                print str(int(100. * float(i) / float((grid + 1) ** 2))) + "%",
        if verbose:
            print
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel('sigma (ms)')
    ylabel('a')
    title('Synfire chain state space')
    axis([sigmamin / ms, sigmamax / ms, 0, 120])

def probability_vs_a(neuron_multiply = 1, verbose=True):
    '''Generates figure 2C'''
    amin = 0
    amax = 100
    sigmamin = 0
    sigmamax = 5
    
    npts = 50
    step = 100/int(npts)
    
    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = params.neurons_per_layer * neuron_multiply

    net = DefaultNetwork(params)
    if verbose:
        print "Spike probability is being calculated for input volleys of varying neuronal number and dipersion:"
    start_time = time.time()
    figure()
    for sigmai in range(sigmamax+1):
        alpha = []
        for ai in xrange(0,amax + step,step): #
            params.initial_burst_a, params.initial_burst_sigma = ai, sigmai * ms
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
            alpha.append(newa/float(amax))
        plot(xrange(0,amax + step,step),alpha)
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel('a (neurons)')
    ylabel('alpha (probability)')
    title('spike probability vs input spike number and sigma')
    axis([amin,amax,0,1])

def newsigma_vs_sigmain(neuron_multiply = 1, verbose=True):
    '''Generates figure 2d'''
    amin = 0
    amax = 100
    sigmamin = 0
    sigmamax = 5
    
    npts = 5
    
    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = params.neurons_per_layer * neuron_multiply

    net = DefaultNetwork(params)
    if verbose:
        print "\nSynchrony is being calculated for input volleys of varying neuronal number and dipersion:"
    start_time = time.time()
    figure()
    for ai in [45,65,100]:
        sigmaOut = []
        for sigmai in linspace(sigmamin,sigmamax,npts): #
            params.initial_burst_a, params.initial_burst_sigma = ai, sigmai * ms
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon_E[-1], params.initial_burst_t)
            sigmaOut.append(newsigma)
        plot(linspace(sigmamin,sigmamax,npts),sigmaOut)
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel('sigma in (in ms)')
    ylabel('sigma out (in s)')
    title('synchrony vs input spike number and sigma')
    axis([sigmamin,sigmamax,sigmamin * ms,sigmamax * ms])

def aout_vs_ain(neuron_multiply = 1, verbose=True):
    '''Generates figure 4a'''
    amin = 0
    amax = 100
    sigmamin = 0
    sigmamax = 5
    
    npts = 10
    step = 100/int(npts)
    
    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = params.neurons_per_layer * neuron_multiply

    net = DefaultNetwork(params)
    if verbose:
        print "Final spike number is being calculated for input volleys of varying neuronal number and dipersion:"
    start_time = time.time()
    figure()
    for sigmai in range(sigmamax+1):
        aout = []
        for ai in xrange(0,amax + step,step): #
            params.initial_burst_a, params.initial_burst_sigma = ai, sigmai * ms
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon_E[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
            aout.append(newa)
        plot(xrange(0,amax + step,step),aout)
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel('a_in (spikes)')
    ylabel('a_out (spikes)')
    title('a_out vs input spike number a_in and sigma')
    axis([amin,amax,amin,amax])

def isoclines(grid, neuron_multiply, verbose=True):
    amin = 0
    amax = 100
    sigmamin = 0. * ms
    sigmamax = 4. * ms
    dsigma = 1. * ms
    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = int(params.neurons_per_layer * neuron_multiply)
    net = DefaultNetwork(params)
    i = 0
    
    if verbose:
        print "Completed:"
    start_time = time.time()
    figure()
    
    aouta = []
    aouts = []
    souta = []
    souts = []
    ovrlp = {}
    
    newsigma = 0. * ms
    for ai in range(grid + 1):
        ovrlp_s = []
        for sigmai in range(grid + 1):
            a = int(amin + (ai * (amax - amin)) / grid)
            if a > amax: a = amax
            sigma = sigmamin + sigmai * (sigmamax - sigmamin) / grid
            params.initial_burst_a, params.initial_burst_sigma = a, sigma
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon_E[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
       
            #col = (float(ai) / float(grid), float(sigmai) / float(grid), 0.5)
            plot([sigma / ms, newsigma / ms], [a, newa], color=[0,0,0])
            plot([sigma / ms], [a], marker='.', color=[0,0,0], markersize=15)
            if (newa-a >= 0): 
                aouta.append(a)
                aouts.append(sigma / ms)  
                plot([sigma / ms], [a], marker='.', color='b', markersize=15) 
            if (newsigma*1000 - sigma / ms) < 0.01:
                souta.append(a)
                souts.append(sigma / ms)
                plot([sigma / ms], [a], marker='.', color='r', markersize=15) 
            if (newa-a >= 0) and (newsigma*1000 - sigma / ms) < 0.01:
                if a > 10:
                    ovrlp_s.append(sigma / ms)
                    ovrlp.update({a:ovrlp_s})
                plot([sigma / ms], [a], marker='.', color='g', markersize=15)                        
            i += 1
            if verbose:
                print str(int(100. * float(i) / float((grid + 1) ** 2))) + "%",
        if verbose:
            print
                    

    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"

    print "\nThe points of intersection are:\n"

    xlabel('sigma (ms)')
    ylabel('a')
    title('Isoclines')
    axis([sigmamin / ms, sigmamax / ms, 0, 100])       

    print "stable fixed point at ",max(ovrlp.keys()),ovrlp[max(ovrlp.keys())][0]
    print "\nSaddle node at ",min(ovrlp.keys()),ovrlp[min(ovrlp.keys())][-1]
    print "\n"
    
def fp_vs_inh(grid, neuron_multiply, weight, verbose=True):
    amin = 0
    amax = 100
    sigmamin = 0. * ms
    sigmamax = 4. * ms
    dsigma = 1. * ms
    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = int(params.neurons_per_layer * neuron_multiply)
    net = DefaultNetwork(params)
    i = 0
    
    if verbose:
        print "Completed:"
    start_time = time.time()
    figure()
    
    aouta = []
    aouts = []
    souta = []
    souts = []
    ovrlp = {}
    
    newsigma = 0. * ms
    for ai in range(grid + 1):
        ovrlp_s = []
        for sigmai in range(grid + 1):
            a = int(amin + (ai * (amax - amin)) / grid)
            if a > amax: a = amax
            sigma = sigmamin + sigmai * (sigmamax - sigmamin) / grid
            params.initial_burst_a, params.initial_burst_sigma = a, sigma
            params.wi = weight
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon_E[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
       
            #col = (float(ai) / float(grid), float(sigmai) / float(grid), 0.5)
            plot([sigma / ms, newsigma / ms], [a, newa], color=[0,0,0])
            plot([sigma / ms], [a], marker='.', color=[0,0,0], markersize=15)
            if (newa-a >= 0): 
                aouta.append(a)
                aouts.append(sigma / ms)  
                plot([sigma / ms], [a], marker='.', color='b', markersize=15) 
            if (newsigma*1000 - sigma / ms) < 0.01:
                souta.append(a)
                souts.append(sigma / ms)
                plot([sigma / ms], [a], marker='.', color='r', markersize=15) 
            if (newa-a >= 0) and (newsigma*1000 - sigma / ms) < 0.01:
                if a > 10:
                    ovrlp_s.append(newsigma*1000)
                    ovrlp.update({newa:ovrlp_s})
                plot([sigma / ms], [a], marker='.', color='g', markersize=15)                        
            i += 1
            if verbose:
                print str(int(100. * float(i) / float((grid + 1) ** 2))) + "%",
        if verbose:
            print
                    
    #plot(aouts,aouta,'b-')
    #plot(souts,souta,'r-')
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"

    print "\nThe points of intersection are:\n"
    '''
    foo = 1
    while(foo):
        if min(ovrlp.keys()) < 10:
            try:
                del ovrlp[min(ovrlp.keys())]
            except KeyError:
                foo = 0
    '''

    xlabel('sigma (ms)')
    ylabel('a')
    title('Isoclines')
    axis([sigmamin / ms, sigmamax / ms, 0, 120])     
    
    savefig(("wi{0}.png").format(params.wi), bbox_inches='tight')
    
    print "\nweight ",weight  

     
    if ovrlp.keys()!=[]:
        print "\nstable fixed point at ",max(ovrlp.keys()),ovrlp[max(ovrlp.keys())][0]
        print "\nSaddle node at ",min(ovrlp.keys()),ovrlp[min(ovrlp.keys())][-1]
        print "\n"       
        return array([(max(ovrlp.keys()),ovrlp[max(ovrlp.keys())][0]),(min(ovrlp.keys()),ovrlp[min(ovrlp.keys())][-1])])

    else:
        print "None\n"
        return ((None,None),(None,None))

def fpVsInhRun():
    params = default_params()
    wi = params.wi
    sfp = [] #Stable fixed point list
    sn = []  #Saddle node list
    ratio = []
    for i in linspace(2*wi,0,50):
        temp = fp_vs_inh(10,50,i,True)
        sfp.append(temp[0])
        sn.append(temp[1])
        ratio.append(i)

    sfp = array(sfp)
    sn = array(sn)
    ratio = array(ratio)

    z = ratio
    xst = sfp[:,0]
    yst = sfp[:,1]
    xsn = sn[:,0]
    ysn = sn[:,1] 

    # Save the data
    f_sfp = open("sfp.p","wb")
    f_sn  = open("sn.p","wb")
    pickle.dump(sfp,f_sfp) 
    pickle.dump(sn,f_sn)
    
def loadPlotData():
    params = default_params()
    wi = params.wi
    sfp = [] #Stable fixed point list
    sn = []  #Saddle node list
    ratio = linspace(2*wi,0,50)
    
    sfp = pickle.load(open("sfp.p",'r'))
    sn = pickle.load(open("sn.p",'r'))

    z = ratio
    xst = sfp[:,0]
    yst = sfp[:,1]
    xsn = sn[:,0]
    ysn = sn[:,1] 

    # Plot in 3D
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xst, yst, z, label='stable fixed points')
    ax.plot(xsn, ysn, z, label='saddle node points')
    ax.legend()
    plt.show()

    
##--------------------------------------------
## Uncomment below functions to generate state space
##--------------------------------------------
#print 'Computing SFC with multiple layers'
#print 'Plotting SFC state space'
#state_space(10,1)
#state_space(8,10)
#state_space(10,100)
#state_space(10,50)
#isoclines(10,50)

##--------------------------------------------
## Uncomment below function to run and plot fixed point vs inhibition
##--------------------------------------------
fpVsInhRun()
loadPlotData()


##--------------------------------------------
## Uncomment below functions to generate figures 2c,2d,3a,4a,4b,4c/3c and 4d
##--------------------------------------------
#probability_vs_a(1)
#newsigma_vs_sigmain(1)
#aout_vs_ain(1)
#show()
