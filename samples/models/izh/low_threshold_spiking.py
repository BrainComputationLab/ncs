#!/usr/bin/python

import sys

import ncs

def run(argv):
	sim = ncs.Simulation()
	lts_parameters = sim.addModelParameters("lts","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.25,
								 "c": -65.0,
								 "d": 2.0,
								 "u": -16.0,
								 "v": -65.0,
								 "threshold": 30,
								})
	group_1=sim.addCellGroup("group_1",1,lts_parameters,None)
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	#sim.addInput("rectangular_current",{"amplitude": 10,"width": 1, "frequency": 1},group_1,1,0.01,1)
	sim.addInput("rectangular_current",{"amplitude":10,"width": 1, "frequency": 1},group_1,1,0.01,1.0)
	current_report=sim.addReport("group_1","neuron","synaptic_current",1.0)
	current_report.toStdOut()
	voltage_report=sim.addReport("group_1","neuron","neuron_voltage",1.0)
	voltage_report.toAsciiFile("./low_threshold_spiking_izh.txt")	
	#voltage_report.toStdOut()
	sim.step(1000)
#	if not sim.init(argv):
#                print "failed to initialize simulation."
#                return

	return

if __name__ == "__main__":
	run(sys.argv)
