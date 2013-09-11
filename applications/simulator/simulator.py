#!/usr/bin/python
import ctypes
import sys
flag = sys.getdlopenflags()
sys.setdlopenflags(flag | ctypes.RTLD_GLOBAL)
#ctypes.cdll.LoadLibrary("/home/roger/git/ncs/build/base/libncs_sim.so")
#ctypes.cdll.LoadLibrary("/home/roger/git/ncs/build/base/libncs_spec.so")


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
  simulation.run(pyncs.string_list(argv))

if __name__ == "__main__":
  Run(sys.argv)
  
