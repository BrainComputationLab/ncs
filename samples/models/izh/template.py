#!/usr/bin/python


#include this two statements at the top
import sys
import ncs

def run(argv):
	#ncs.simulation() is required
	sim = ncs.Simulation()

	#start writing model - biology information
	
	#Model Parameters
	#Model parameters can be added by addModelParameters function.
	#Parameters for addModelParameters function:
	#	1. A model name (string)
	#	2. A model type (string)
	#		izhikevich, ncs, regular_current, flat (synapse), etc.
	# 	3. A map for parameter names to their values(Generators)
	#		Generators can be exact, Normal, uniform
	#		exact example : "a": 0.02
	#		uniform example: "a": ncs.Uniform(min, max)
        #		normal example: "a": ncs.Normal(mean, standard deviation)
	# Example of addModelParameters with regular_spiking izhikevich neuron 	
	regular_spiking_parameters = sim.addModelParameters("regular_spiking","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.2,
								 "c": -65.0,
								 "d": 8.0,
								 "u": -12.0,
								 "v": -60.0,
								 "threshold": 30,
								})
	#CellGroup
	#Cell group can be created by addCellGroup function.
	#Parameters for addCellGroup function:
	#	1. name of the group
	#	2. number of cells
	#	3. model parameters
	#	4. geometry generator (optional)
	group_1=sim.addCellGroup("group_1",1,regular_spiking_parameters,None)

	#after all the model information, initialize simulation
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	#after initializing simulation, add stimulus/current and report information

	#current can be added by addInput function. 
	#parameters for addInput function:
	#	1. A model type 
	#		(rectangular_current, linear, etc.)
	#	2. A map from parameter names (strings) to their values (Generators)
	#		Parameter names are amplitude, width, frequency, etc.
	#		 	(all of those are not required. Only amplitude is requird)
	#		Generators can be exact, normal, or uniform
	#               	exact example : "a": 0.02
        #               	uniform example: "a": ncs.Uniform(min, max)
        #               	normal example: "a": ncs.Normal(mean, standard deviation)
	#	3. A set of target cell groups
	#	4. probability of a cell receiving input
	#	5. start time for stimulus (seconds)
	#		For example, if you wanted to start stimulus at 1 ms, write 0.01
	#	6. end time for stimulus (seconds)
	sim.addInput("rectangular_current",{"amplitude":10,"width": 1, "frequency": 1},group_1,1,0.01,1.0)
	
	#Report can be added by add report function
	#Parameters for addReport function:
	#	1. A set of neurons or a set of synapses to report on
	#	2. A target type: "neuron" or "synapses"
	#	3. type of report: synaptic_current, neuron_voltage, etc.
	#	4. Probability (the percentage of elements to report on)

	#Report can be displayed to terminal OR file (not both)

	#example of a report to terminal
	#this model doesn't have synapse information so 0s will be displayed on terminal
	current_report=sim.addReport("group_1","neuron","synaptic_current",1.0)
	current_report.toStdOut()

	#example of a report to file
	voltage_report=sim.addReport("group_1","neuron","neuron_voltage",1.0)
	# ./ places report in the same directory as the code file
	voltage_report.toAsciiFile("./template_regular_spiking.txt")	

	#number of time steps - each time step is 1 ms
	sim.step(1000)

	#return statement
	return

#following is required
if __name__ == "__main__":
	run(sys.argv)
