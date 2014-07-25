#!/usr/bin/python

import os, sys
ncs_lib_path = ('../../../')
sys.path.append(ncs_lib_path)
import ncs

def run(argv):
	sim = ncs.Simulation()
	ib_parameters = sim.addNeuron("ib","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.2,
								 "c": -55.0,
								 "d": 4.0,
								 "u": -13.0,
								 "v": -65.0,
								 "threshold": 30,
								})
	group_1=sim.addNeuronGroup("group_1",1,ib_parameters,None)
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	sim.addStimulus("rectangular_current",{"amplitude":10,"width": 1, "frequency": 1},group_1,1,0.01,1.0)
	voltage_report=sim.addReport("group_1","neuron","neuron_voltage",1.0,0.0,1.0)
	voltage_report.toAsciiFile("./intrinsically_bursting_izh.txt")	
	sim.run(duration=1.0)

	return

if __name__ == "__main__":
	run(sys.argv)
