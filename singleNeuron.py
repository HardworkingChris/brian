from brian import *
# Neuron model parameters
Vr = -70 * mV
Vt = -55 * mV
taum = 10 * ms # was 10, 12.35 interesting TMM modifications 20140424
taupsp = 0.325 * ms
weight = 4.86 * mV
# Neuron model
eqs = Equations('''
dV/dt=(-(V-Vr)+x)*(1./taum) : volt
dx/dt=(-x+y)*(1./taupsp) : volt
dy/dt=-y*(1./taupsp)+25.27*mV/ms+\
    (39.24*mV/ms**0.5)*0 : volt
''')
# Neuron groups.  Modified to be larger TMM
P = NeuronGroup(N=3, model=eqs,
    threshold=Vt, reset=Vr, refractory=1 * ms)
# Input spike volley's a and sigma from user
print "\nInput no.of neurons 'a' and temporal dispersion of spikes 'sigma'"
user_n = int(raw_input("\na = "))
user_sigma = float(raw_input("sigma(in mSec) = "))
print "\n"
Pinput = PulsePacket(t=10 * ms, n=user_n, sigma=user_sigma * ms)
# The network structure.  Modified to be larger TMM
Pgp = [ P.subgroup(1) for i in range(3)]
C = Connection(P, P, 'y',delay=5*ms)
# modified to be larger TMM
for i in range(2):
    C.connect_full(Pgp[i], Pgp[i + 1], weight)
Cinput = Connection(Pinput, Pgp[0], 'y')
Cinput.connect_full(weight=weight)
# Record the spikes
Mgp = [SpikeMonitor(p) for p in Pgp]
Minput = SpikeMonitor(Pinput)
monitors = [Minput] + Mgp
# Setup the network, and run it
P.V = Vr + rand(len(P)) * (Vt - Vr)
# Track voltage 
trace_v = StateMonitor(P,'V',record=[1,2,3])
# Track conductance  
trace_x = StateMonitor(P,'x',record=[1,2,3]) 
# Track synapse
trace_y = StateMonitor(P,'y',record=[1,2,3]) 
# Run the simulation
run(40 * ms) # trace records the state variable change along the run
# Plot voltage trace
plot(trace_v.times/ms,trace[1]/mV)
plot(trace_v.times/ms,trace[2]/mV)
plot(trace_v.times/ms,trace[3]/mV)
show()
# Plot voltage trace
plot(trace_x.times/ms,trace[1]/mV)
plot(trace_x.times/ms,trace[2]/mV)
plot(trace_x.times/ms,trace[3]/mV)
show()
# Plot voltage trace
plot(trace_y.times/ms,trace[1]/mV)
plot(trace_y.times/ms,trace[2]/mV)
plot(trace_y.times/ms,trace[3]/mV)
show()
# Plot voltage change due to synaptic current from 1 neuron
plot(trace_v.times/ms - trace_v.times/ms ,trace[2]/mV)
show()
# Plot result
raster_plot(showgrouplines=True, *monitors)
show()
