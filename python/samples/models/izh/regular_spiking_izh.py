#!/usr/bin/python

import os, sys
#specify the location of ncs.py in ncs_lib_path
ncs_lib_path = ('../../../../')
sys.path.append(ncs_lib_path)
import ncs

def run(argv):
	sim = ncs.Simulation()
	regular_spiking_parameters = sim.addNeuron("regular_spiking","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.2,
								 "c": -65.0,
								 "d": 8.0,
								 "u": -12.0,
								 "v": -60.0,
								 "threshold": 30
								})
	group_1=sim.addNeuronGroup("group_1",15,regular_spiking_parameters,None)

	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	input_parameters = {
				"amplitude":10
			   }

	sim.addStimulus("rectangular_current", input_parameters, group_1, 1, 0.01, 1.0)
	voltage_report=sim.addReport("group_1", "neuron", "neuron_voltage", 1, 0.0, 1.0)
	voltage_report.toAsciiFile("./regular_spiking_izh.txt")

	sim.run(duration=1.0)

	return

if __name__ == "__main__":
	run(sys.argv)
