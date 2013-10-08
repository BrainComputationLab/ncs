#!/usr/bin/python
import sys


import json_model 
import pyncs

def Run(argv):
  if len(argv) < 2:
    print "Usage: %s <model_file>" % argv[0]
  model = json_model.JSONModel(argv[1])
  if not model.valid:
    print "Failed to load model"
    return
  model_specification = model.model_specification
  simulation = pyncs.Simulation(model_specification)
  if not simulation.init(pyncs.string_list(argv)):
    print "Failed to initialize simulator."
    return
  print "Injecting pre-specified inputs."
  for name, group in model.input_groups.items():
    simulation.addInput(group)
  print "Injection complete."
  for i in range(0,10):
    simulation.step()
  del simulation

if __name__ == "__main__":
  Run(sys.argv)
  
