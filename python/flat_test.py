#!/usr/bin/python

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
  group_1=sim.addCellGroup("group_1",1,bursting_parameters,None)
  group_2=sim.addCellGroup("group_2",1,bursting_parameters,None)
  flat_parameters = sim.addModelParameters("flat_synapse", "flat", 
                                           { "delay": 3,
                                           "current": 10.0
                                           })
  sim.connect("1_to_2", group_1, group_2, 1.0, flat_parameters)

  if not sim.init(argv):
    print "failed to initialize simulation."
    return

  sim.addInput("rectangular_current",{"amplitude":18,"width": 1, "frequency": 1},group_1,1,0.0,1.0)
#	current_report=sim.addReport("group_1","neuron","synaptic_current",1.0)
#	current_report.toStdOut()
  voltage_report=sim.addReport("group_2","neuron","synaptic_current",1.0).toStdOut()
#voltage_report.toAsciiFile("./bursting_izh.txt")
  sim.step(10)

  return

if __name__ == "__main__":
  run(sys.argv)
