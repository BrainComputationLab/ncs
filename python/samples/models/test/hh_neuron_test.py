#!/usr/bin/python

import math
import os,sys
ncs_lib_path = ('../../../../python/')
sys.path.append(ncs_lib_path)
import ncs

def Run(argv):
  ak = 10.0 / (100.0 * (math.exp(1.0) - 1))
  bk = 0.125
  tk = 1.0 / (ak + bk)
  xk0 = ak * tk;
  potassium_channel = {
    "type": "voltage_gated",
    "conductance": 36.0,
    "reversal_potential": -72.0,
    "particles": [
      { #n
        "x_initial": xk0,
        "alpha": {
          "a": 0.5,
          "b": 0.01,
          "c": 1.0,
          "d": 50.0,
          "f": -10.0,
          "h": -1.0,
        },
        "beta": {
          "a": 0.125,
          "b": 0.0,
          "c": 0.0,
          "d": 60.0,
          "f": 80.0,
          "h": 1.0,
        },
        "power": 4.0,
      }
    ],
  }
  am = 25.0 / (10.0 * (math.exp(2.5) - 1.0))
  bm = 4.0
  tm = 1.0 / (am + bm)
  xm0 = am * tm
  ah = 0.07;
  bh = 1.0 / (math.exp(3.0) + 1)
  th = 1.0 / (ah + bh)
  xh0 = ah * th
  sodium_activation = {
    "x_initial": xm0,
    "alpha": {
      "a": 3.5,
      "b": 0.1,
      "c": 1.0,
      "d": 35.0,
      "f": -10.0,
      "h": -1.0,
    },
    "beta": {
      "a": 4.0,
      "b": 0.0,
      "c": 0.0,
      "d": 60.0,
      "f": 18.0,
      "h": 1.0,
    },
    "power": 3.0,
  }
  sodium_deactivation = { 
    "x_initial": xh0,
    "alpha": {
      "a": 0.07,
      "b": 0.0,
      "c": 0.0,
      "d": 60.0,
      "f": 20.0,
      "h": 1.0,
    },
    "beta": {
      "a": 1.0,
      "b": 0.0,
      "c": 1.0,
      "d": 30.0,
      "f": -10.0,
      "h": 1.0,
    },
    "power": 1.0,
  }
  sodium_channel = {
    "type": "voltage_gated",
    "conductance": 120.0,
    "reversal_potential": 55.0,
    "particles": [
      sodium_activation,
      sodium_deactivation
    ],
  }
  leak_channel = {
    "type": "voltage_gated",
    "conductance": 0.3,
    "reversal_potential": 10.6 - 60.0,
    "particles": [
      { #m
        "x_initial": 1.0,
        "alpha": {
          "a": 1.0,
          "b": 0.0,
          "c": 1.0,
          "d": 0.0,
          "f": 0.0,
          "h": 0.0,
        },
        "beta": {
          "a": 1.0,
          "b": 0.0,
          "c": 1.0,
          "d": 0.0,
          "f": 0.0,
          "h": 0.0,
        },
        "power": 1.0,
      }
    ],
  }
  hh_cell = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "capacitance": 1.0,
    "channels": [
      potassium_channel,
      sodium_channel,
      leak_channel
    ]
  }

  sim = ncs.Simulation()
  neuron_parameters = sim.addNeuron("hh_neuron",
                                             "hh",
                                             hh_cell
                                            )
  group_1 = sim.addNeuronGroup("group_1", 1, "hh_neuron", None) # last param is geometry

  all_cells = sim.addNeuronAlias("all_cells", [group_1])
  sim.addNeuronAlias("all", all_cells)
  sim.addNeuronAlias("all_2", "all_cells")

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addStimulus("rectangular_current", { "amplitude": 10.0 }, group_1, 1.0, 0.0, 1.0)

  voltage_report = sim.addReport("group_1", "neuron", "neuron_voltage", 1.0,0.0,0.5)
#voltage_report.toAsciiFile("/tmp/voltages.txt")
  voltage_report.toStdOut()

  sim.run(duration=0.500)
  return

if __name__ == "__main__":
  Run(sys.argv)


