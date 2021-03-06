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
dy/dt=-y*(1./taupsp)+25.27*mV/ms: volt
''')
# Neuron groups.  Modified to be larger TMM
P = NeuronGroup(N=400, model=eqs,
    threshold=Vt, reset=Vr, refractory=1 * ms)
# Input spike volley's a and sigma from user
print "\nInput no.of neurons 'a' and temporal dispersion of spikes 'sigma'"
user_n = int(raw_input("\na = "))
user_sigma = float(raw_input("sigma(in mSec) = "))
print "\n"
Pinput = PulsePacket(t=10 * ms, n=user_n, sigma=user_sigma * ms)
# The network structure.  Modified to be larger TMM
Pgp = [ P.subgroup(100) for i in range(4)]
C = Connection(P, P, 'y',delay=5*ms)
# modified to be larger TMM
for i in range(3):
    C.connect_full(Pgp[i], Pgp[i + 1], weight)
Cinput = Connection(Pinput, Pgp[0], 'y')
Cinput.connect_full(weight=weight)
# Record the spikes
Mgp = [SpikeMonitor(p) for p in Pgp]
Minput = SpikeMonitor(Pinput)
monitors = [Minput] + Mgp
# Setup the network, and run it
P.V = Vr #+ rand(len(P)) * (Vt - Vr)
# Track voltage 
trace_v = StateMonitor(P,'V',record=[1,201,301])
# Track conductance  
trace_x = StateMonitor(P,'x',record=[1,201,301]) 
# Track synapse
trace_y = StateMonitor(P,'y',record=[1,201,301]) 
# Run the simulation
run(50 * ms) # trace records the state variable change along the run
# Plot voltage trace
xlabel('time (in ms)')
ylabel('voltage (in mV)')
title('Leaky integrate and fire neuron voltage trace')
plot(trace_v.times/ms,trace_v[1]/mV)
#plot(trace_v.times/ms,trace_v[201]/mV)
#plot(trace_v.times/ms,trace_v[301]/mV)
show()
# Plot conductance trace
xlabel('time (in ms)')
ylabel('g_syn (in mV)')
title('Leaky integrate and fire neuron K conductance trace')
plot(trace_x.times/ms,trace_x[1]/mV)
#plot(trace_x.times/ms,trace_x[201]/mV)
#plot(trace_x.times/ms,trace_x[301]/mV)
show()

# Plot synapse trace
xlabel('time (in ms)')
ylabel('g_syn (in mV)')
title('Synaptic conductance trace')
plot(trace_y.times/ms,trace_y[1]/mV)
#plot(trace_y.times/ms,trace_y[201]/mV)
#plot(trace_y.times/ms,trace_y[301]/mV)
show()

# Plot voltage change due to synaptic current from 1 neuron
xlabel('time (in ms)')
ylabel('voltage (in mV)')
title('Change in voltage due to one spike')
plot(trace_v.times/ms,trace_v[1]/mV - trace_v[201]/mV)
show()

# Plot result
raster_plot(showgrouplines=True, *monitors)
show()
