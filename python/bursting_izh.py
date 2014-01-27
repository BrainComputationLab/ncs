#!/usr/bin/python

import sys

import ncs

def run(argv):
	sim = ncs.Simulation()
	bursting_parameters = sim.addNeuron("bursting","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.3,
								 "c": -50.0,
								 "d": 4.0,
								 "u": -12.0,
								 "v": -65.0,
								 "threshold": 30,
								})
	group_1=sim.addNeuronGroup("group_1",2,bursting_parameters,None)
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	sim.addStimulus("rectangular_current",{"amplitude":18,"width": 1, "frequency": 1},group_1,1,0.01,1.0)
#	current_report=sim.addReport("group_1","neuron","synaptic_current",1.0)
#	current_report.toStdOut()
	voltage_report=sim.addReport("group_1","neuron","neuron_fire",1.0).toStdOut()
#voltage_report.toAsciiFile("./bursting_izh.txt")
	sim.step(10)

	return

if __name__ == "__main__":
	run(sys.argv)
