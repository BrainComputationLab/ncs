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
                                                  "v": ncs.Normal(-60.0, 5.0)
                                                 }
                                                )
  group_1 = sim.addCellGroup("group_1", 100, "label_excitatory", None) # last param is geometry
  group_2 = sim.addCellGroup("group_2", 100, excitatory_parameters)

  all_cells = sim.addCellAlias("all_cells", [group_1, "group_2"])
  sim.addCellAlias("all", all_cells)
  sim.addCellAlias("all_2", "all_cells")

  flat_parameters = sim.addModelParameters("flat_synapse", 
                                           "flat", 
                                           { "delay": [2,5],
                                             "current": ncs.Normal(18.0,2.0)
                                           })
  all_to_all = sim.connect(all_cells, "all_2", 1.0, flat_parameters)
  all_to_all_2 = sim.connect([group_1, group_2], "all_2", 1.0, flat_parameters)
  one_to_two = sim.connect(group_1, "group_2", 1.0, "flat_synapse")

  all_connections = sim.addConnectionAlias("all_connections", [all_to_all, one_to_two])

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.addInput("rectangular_current", { "amplitude": 18.0 }, group_1, 0.0, 1.0)
  sim.addInput("rectangular_current", { "amplitude": 18.0 }, "all", 0.0, 0.5)
  sim.addInput("rectangular_pulse", { "frequency": 60.0,
                                      "amplitude": 18.0 }, "group_2", 0.1, 1.1)

  data_source = sim.addReport("group_1", "neuron", "neuron_voltage", 1.0)
  data_source.to_file("/tmp/foo.txt")
  sim.step(100)
  return

if __name__ == "__main__":
  Run(sys.argv)
