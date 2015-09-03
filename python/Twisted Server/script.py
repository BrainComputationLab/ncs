#!/usr/bin/python
import os, sys
ncs_lib_path = ("../../../")
sys.path.append(ncs_lib_path)
import ncs

if __name__ == "__main__":
	sim = ncs.Simulation()
	parameters_0 = sim.addNeuron("Cell 3", "izhikevich", {'a': 0.2, 'c': -65.0, 'b': 0.2, 'd': 8.0, 'u': <ncs.Uniform instance at 0x7f477bd91098>, 'v': <ncs.Uniform instance at 0x7f477bd910e0>, 'threshold': 30.0})
	group_0 = sim.addNeuronGroup("group_0", 150, parameters_0, None)
	if not sim.init(sys.argv):
		print "failed to initialize simulation."
		return
	sim.run(duration=1.0)
