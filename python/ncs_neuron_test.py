#!/usr/bin/python

import math
import sys

import ncs

def Run(argv):
  voltage_channel = {
    "type": "voltage_gated_ion",
    "m_initial": 0.0,
    "reversal_potential": -80,
    "v_half": -44,
    "deactivation_slope": 40,
    "activation_slope": 20,
    "equilibrium_slope": 8.8,
    "r": 1.0 / 0.303,
    "conductance": 5 * 0.00015
  }
  calcium_channel = {
    "type": "calcium_dependent",
    "m_initial": 0.0,
    "reversal_potential": -80,
    "m_power": 2,
    "conductance": 6.0 * 0.0009,
    "forward_scale": 0.000125,
    "forward_exponent": 2,
    "backwards_rate": 2.5,
    "tau_scale": 0.01,
  }
  ncs_cell = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 5.0,
    "calcium_spike_increment": 100.0,
    "tau_calcium": 0.07,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": [
      -38, 30, -43, -60, -60
    ],
    "capacitance": 1.0,
    "channels": [
      voltage_channel,
      calcium_channel,
    ]
  }

  sim = ncs.Simulation()
  neuron_parameters = sim.addModelParameters("ncs_neuron",
                                             "ncs",
                                             ncs_cell
                                            )
  group_1 = sim.addCellGroup("group_1", 100, "ncs_neuron", None) # last param is geometry

  all_cells = sim.addCellAlias("all_cells", [group_1])
  sim.addCellAlias("all", all_cells)
  sim.addCellAlias("all_2", "all_cells")

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addInput("rectangular_current", { "amplitude": 0.1 }, group_1, 1.0, 0.0, 1.0)

#voltage_report = sim.addReport("group_1", "neuron", "neuron_voltage", 1.0)
#voltage_report.toAsciiFile("/tmp/voltages.txt")
#voltage_report.toStdOut()

  sim.step(10)
  del sim
  return

if __name__ == "__main__":
  Run(sys.argv)



