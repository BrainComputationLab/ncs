#!/usr/bin/python

import sys

import ncs

def Run(argv):
  potassium_channel = {
    "type": "voltage_gated",
    "conductance": 36.0,
    "particles": [
      { #m
        "x_initial": 0.0,
        "alpha": {
          "a": 0.1,
          "b": 0.01,
          "c": -1.0,
          "d": -10.0,
          "f": -10.0,
          "h": 1.0,
        },
        "beta": {
          "a": 0.125,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "f": 80.0,
          "h": 1.0,
        },
        "power": 4.0,
      }
    ],
  }
  excitatory_soma = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 5.0,
    "calcium_spike_increment": 100.0,
    "tau_calcium": 0.07,
    "leak_reversal_potential": 0.0,
    "leak_conductance": 0.0,
    "tau_membrane": 0.020,
    "r_membrane": 200.0,
    "channels": [
      potassium_channel
    ]
  }

  sim = ncs.Simulation()
  neuron_parameters = sim.addModelParameters("ncs_neuron",
                                             "ncs",
                                             excitatory_soma
                                            )
  group_1 = sim.addCellGroup("group_1", 1, "ncs_neuron", None) # last param is geometry

  all_cells = sim.addCellAlias("all_cells", [group_1])
  sim.addCellAlias("all", all_cells)
  sim.addCellAlias("all_2", "all_cells")

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addInput("rectangular_current", { "amplitude": 1.0 }, group_1, 1.0, 0.0, 0.1)

  voltage_report = sim.addReport("group_1", "neuron", "neuron_voltage", 1.0)
  voltage_report.toStdOut()

  sim.step(10)
  return

if __name__ == "__main__":
  Run(sys.argv)


