#!/usr/bin/python

import sys

import ncs

def Run(argv):
  potassium_channel = {
    "type": "voltage_gated",
    "conductance": 36.0,
    "particles": [
      { #m
        "alpha": {
          "a": 0.0,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "f": 0.0,
          "h": 0.0,
        },
        "beta": {
          "a": 0.0,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "f": 0.0,
          "h": 0.0,
        },
        "power": 4.0,
      }
    ],
  }

  sim = ncs.Simulation()
  neuron_parameters = sim.addModelParameters("ncs_neuron",
                                             "ncs",
                                             {
                                              "channels": [
                                                potassium_channel
                                              ],
                                             }
                                            )
  group_1 = sim.addCellGroup("group_1", 100, "ncs_neuron", None) # last param is geometry

  all_cells = sim.addCellAlias("all_cells", [group_1])
  sim.addCellAlias("all", all_cells)
  sim.addCellAlias("all_2", "all_cells")

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addInput("rectangular_current", { "amplitude": 18.0 }, group_1, 0.1, 0.0, 0.1)

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


