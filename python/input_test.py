#!/usr/bin/python

import math
import sys

import ncs

def run(argv):
	sim = ncs.Simulation()
	bursting_parameters = sim.addModelParameters("bursting","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.3,
								 "c": -50.0,
								 "d": 4.0,
								 "u": -12.0,
								 "v": -65.0,
								 "threshold": 30,
								})
	group_1=sim.addCellGroup("group_1",2,bursting_parameters,None)
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	sim.addInput("sine_current",
               {
                 "amplitude_scale":10,
                 "time_scale": 200.0 / math.pi,
                 "phase": 0.0,
                 "amplitude_shift":10
               },
               group_1,
               1,
               0.01,
               1.0)
#	current_report=sim.addReport("group_1","neuron","synaptic_current",1.0)
#	current_report.toStdOut()
	voltage_report=sim.addReport("group_1","neuron","input_current",1.0).toStdOut()
#voltage_report.toAsciiFile("./bursting_izh.txt")
	sim.step(1000)

	return

if __name__ == "__main__":
	run(sys.argv)