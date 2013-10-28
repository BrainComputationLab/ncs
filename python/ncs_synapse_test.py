#!/usr/bin/python

import sys

import ncs

def Run(argv):
  sim = ncs.Simulation()
  excitatory_parameters = sim.addModelParameters("label_excitatory",
                                                 "izhikevich",
                                                 { 
                                                  "a": 0.2,
                                                  "b": 0.2,
                                                  "c": -65.0,
                                                  "d": ncs.Uniform(7.0, 9.0),
                                                  "u": [-15.0, -11.0], # this also makes a uniform
                                                  "v": ncs.Normal(-60.0, 5.0),
                                                  "threshold": 30,
                                                 }
                                                )
  group_1 = sim.addCellGroup("group_1", 100, "label_excitatory", None) # last param is geometry
  group_2 = sim.addCellGroup("group_2", 100, excitatory_parameters)

  all_cells = sim.addCellAlias("all_cells", [group_1, "group_2"])
  sim.addCellAlias("all", all_cells)
  sim.addCellAlias("all_2", "all_cells")

  flat_parameters = sim.addModelParameters("synapse", 
                                           "ncs", 
                                           { "utilization": ncs.Normal(0.5, 0.05),
                                             "redistribution": 1.0,
                                             "last_prefire_time": 0.0,
                                             "last_postfire_time": 0.0,
                                             "tau_facilitation": 0.001,
                                             "tau_depression": 0.001,
                                             "tau_ltp": 0.015,
                                             "tau_ltd": 0.03,
                                             "A_ltp_minimum": 0.003,
                                             "A_ltd_minimum": 0.003,
                                             "max_conductance": 0.004,
                                             "reversal_potential": 0.0,
                                             "tau_postsynaptic_conductance": 0.025,
                                             "psg_waveform_duration": 0.05,
                                             "delay": [1,5],
                                           })
  all_to_all = sim.connect("all_to_all", all_cells, "all_2", 0.1, flat_parameters)
  all_to_all_2 = sim.connect("all_to_all_2", 
                             [group_1, group_2], 
                             "all_2", 
                             0.1, 
                             flat_parameters)
  one_to_two = sim.connect("one_to_two", 
                           group_1, 
                           "group_2", 
                           0.1, 
                           "synapse")

  all_connections = sim.addConnectionAlias("all_connections", [all_to_all, one_to_two])

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addInput("rectangular_current", { "amplitude": 18.0 }, group_1, 0.1, 0.0, 0.1)
  sim.addInput("rectangular_current", { "amplitude": 18.0 }, "all", 0.1, 0.0, 0.2)
  sim.addInput("rectangular_current", { "amplitude": 18.0 }, "group_2", 0.1, 0.1, 0.3)

  voltage_report = sim.addReport("group_1", "neuron", "neuron_voltage", 1.0)
  voltage_report.toAsciiFile("/tmp/foo.txt")

  fire_report = sim.addReport(all_cells, "neuron", "neuron_fire", 1.0)
  fire_report.toStdOut()
  current_report = sim.addReport(all_cells, "neuron", "synaptic_current", 1.0)
  current_report.toStdOut()

  
  sim.step(500)
  return

if __name__ == "__main__":
  Run(sys.argv)

