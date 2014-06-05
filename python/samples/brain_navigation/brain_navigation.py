#!/usr/bin/python

import ncs
import sys

import anatomy_pcrain

def Run(argv):
  sim = ncs.Simulation()
  anatomy_pcrain.add(sim)

  if not sim.init(argv):
    print "Failed to initialize simulation."
    return

  sim.step(50)
  return

if __name__ == "__main__":
  Run(sys.argv)

