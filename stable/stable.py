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
    (39.24*mV/ms**0.5)*xi : volt
''')
# Neuron groups.  Modified to be larger TMM
P = NeuronGroup(N=1000, model=eqs,
    threshold=Vt, reset=Vr, refractory=1 * ms)
# Input spike volley's a and sigma from user
print "\nInput no.of neurons 'a' and temporal dispersion of spikes 'sigma'"
user_n = int(raw_input("\na = "))
user_sigma = float(raw_input("sigma(in mSec) = "))
print "\n"
Pinput = PulsePacket(t=10 * ms, n=user_n, sigma=user_sigma * ms)
# The network structure.  Modified to be larger TMM
Pgp = [ P.subgroup(100) for i in range(10)]
C = Connection(P, P, 'y',delay=5*ms)
# modified to be larger TMM
for i in range(9):
    C.connect_full(Pgp[i], Pgp[i + 1], weight)
Cinput = Connection(Pinput, Pgp[0], 'y')
Cinput.connect_full(weight=weight)
# Record the spikes
Mgp = [SpikeMonitor(p) for p in Pgp]
Minput = SpikeMonitor(Pinput)
monitors = [Minput] + Mgp
# Setup the network, and run it
P.V = Vr + rand(len(P)) * (Vt - Vr)
# Plot voltage trace
trace = StateMonitor(P,'V',record=[1,101,201,301,401,501,601,701,801,999]) 

run(90 * ms) # trace records the state variable change along the run

plot(trace.times/ms,trace[1]/mV)
plot(trace.times/ms,trace[101]/mV)
plot(trace.times/ms,trace[201]/mV)
plot(trace.times/ms,trace[301]/mV)
plot(trace.times/ms,trace[401]/mV)
plot(trace.times/ms,trace[501]/mV)
plot(trace.times/ms,trace[601]/mV)
plot(trace.times/ms,trace[701]/mV)
plot(trace.times/ms,trace[801]/mV)
plot(trace.times/ms,trace[999]/mV)
show()
# Plot result
raster_plot(showgrouplines=True, *monitors)
show()
