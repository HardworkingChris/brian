#import brian_no_units
from brian import *
import time

from brian.library.IF import *
from brian.library.synapses import *

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
    neurons_per_layer=100, #change this to obtain figure 4(a:80,b:90,c:100,d:110)
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
        layer = [ chaingroup.subgroup(p.neurons_per_layer) for i in range(p.num_layers) ]
        # connections
        chainconnect = Connection(chaingroup, chaingroup, 2,delay=5*ms)
        for i in range(p.num_layers - 1):
            chainconnect.connect_full(layer[i], layer[i + 1], p.psp_peak * p.we)
        inputconnect = Connection(inputgroup, chaingroup, 2)
        inputconnect.connect_full(inputgroup, layer[0], p.psp_peak * p.we)
        # monitors
        chainmon = [SpikeMonitor(g, True) for g in layer]
        inputmon = SpikeMonitor(inputgroup, True)
        mon = [inputmon] + chainmon
        # network
        Network.__init__(self, chaingroup, inputgroup, chainconnect, inputconnect, mon)
        # add additional attributes to self
        self.mon = mon
        self.inputgroup = inputgroup
        self.chaingroup = chaingroup
        self.layer = layer
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
                   *self.mon)

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
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
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
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
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
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
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
    
    aout = []
    sigmaout = []    
    
    for ai in range(grid + 1):
        for sigmai in range(grid + 1): #
            a = int(amin + (ai * (amax - amin)) / grid)
            if a > amax: a = amax
            sigma = sigmamin + sigmai * (sigmamax - sigmamin) / grid
            params.initial_burst_a, params.initial_burst_sigma = a, sigma
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
            if abs(newa-a)<50 : aout.append([newa,newsigma])
            #if abs(newsigma - sigma)<dsigma : sigmaout.append([newa,newsigma])
            #col = (float(ai) / float(grid), float(sigmai) / float(grid), 0.5)
            #plot([sigma / ms, newsigma / ms], [a, newa], color=[0,0,0])
            plot([sigma / ms], [a], marker='.', color=[0,0,0], markersize=15)
            i += 1
            if verbose:
                print str(int(100. * float(i) / float((grid + 1) ** 2))) + "%",
        if verbose:
            print
    plot(aout[1],aout[0],'b-')
    #plot(sigmaout[1],sigmaout[0],'r.')
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel('sigma (ms)')
    ylabel('a')
    title('Isoclines')
    axis([sigmamin / ms, sigmamax / ms, 0, 120])
    
##--------------------------------------------
## Uncomment below functions to generate state space
##--------------------------------------------
#print 'Computing SFC with multiple layers'
#print 'Plotting SFC state space'
#state_space(10,1)
#state_space(8,10)
#state_space(10,100)
#state_space(10,50)
isoclines(10,1)
show()


##--------------------------------------------
## Uncomment below functions to generate figures 2c,2d,3a,4a,4b,4c/3c and 4d
##--------------------------------------------
#probability_vs_a(1)
#newsigma_vs_sigmain(1)
#aout_vs_ain(1)
#show()
