#!/usr/bin/python


import os, sys
ncs_lib_path = ('../../../') #Path to ncs.py
sys.path.append(ncs_lib_path)
import ncs

def run(argv):
	#ncs.simulation() is required
	sim = ncs.Simulation()

	#start writing model - biology information

        #addNeuron function
        #Parameters for addNeuron function:
        #       1. A neuron name (string)
        #       2. A neuron type (string)
        #               izhikevich, ncs, or hh
        #       3. A map for parameter names to their values(Generators)
        #               Generators can be exact, Normal, uniform
        #               exact example : "a": 0.02
        #               uniform example: "a": ncs.Uniform(min, max)
        #               normal example: "a": ncs.Normal(mean, standard deviation)
        # Example of addNeuron with regular_spiking izhikevich neuron   
	regular_spiking_parameters = sim.addNeuron("regular_spiking","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.2,
								 "c": -65.0,
								 "d": 8.0,
								 "u": -12.0,
								 "v": -60.0,
								 "threshold": 30,
								})

	#addSynapse function
	#Parameters for addSynapse function
	#	1. A synapse name (string)
	#	2. A synapse type (string)
	#		ncs or flat
	#	3. A map for parameter names to their values (Generators)	
	ncs_synapse_parameters = sim.addSynapse("flat_synapse","ncs",{
										
									"utilization": ncs.Normal(0.5,0.05),
									"redistribution": 1.0,
									"last_prefire_time": 0.0,
									"last_postfire_time": 0.0,
									"tau_facilitation": 0.001,
									"tau_depression": 0.001,
									"tau_ltp": 0.015,
									"tau_ltd": 0.03,
									"A_ltp_minimum": 0.003,
									"A_ltd_minimum": 0.003,
									"max_conductance": 0.3,
									"reversal_potential":0.0,
									"tau_postsynaptic_conductance": 0.02,
									"psg_waveform_duration": 0.05,
									"delay": 1,
								})	

	#addNeuronGroup function
        #Parameters for addNeuronGroup function:
        #       1. A name of the group (string)
        #       2. Number of cells (integer)
        #       3. Neuron parameters
        #       4. Geometry generator (optional)
	group_1=sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
	group_2=sim.addNeuronGroup("group_2",1,regular_spiking_parameters,None)
	
	#addSynapseGroup function
	#Parameters for addSynapseGroup function:
	#	1. A name of the connection
	#	2. Presynaptic NeuronAlias or NeuronGroup
	#	3. Postsynaptic NeuronAlias or NeuronGroup
	#	4. Probability of connection
	#	5. Parameters for synapse
	connection1=sim.addSynapseGroup("connection1","group_1","group_2",1,"flat_synapse");

	#initialize simulation	
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	#addStimulus function
        #parameters for addStimulus function:
        #       1. A stimulus type (string) 
        #               rectangular_current, rectangular_voltage, linear_current, linear_voltage, 
        #               sine_current, or sine_voltage
        #       2. A map from parameter names (strings) to their values (Generators)
        #               Parameter names are amplitude, starting_amplitude, ending_amplitude,
        #               delay, current, amplitude_scale, time_scale, phase, amplitude_shift,
        #               etc. based on the stimulus type
        #       3. A set of target neuron groups
        #       4. probability of a neuron receiving input
        #       5. start time for stimulus (seconds)
        #               For example, if you wanted to start stimulus at 1 ms, write 0.01
        #       6. end time for stimulus (seconds)
	sim.addStimulus("rectangular_current",{"amplitude":10},group_1,1,0.01,1.0)

	#addReport function
        #Parameters for addReport function:
        #       1. A set of neuron group or a set of synapse group to report on
        #       2. A target type: "neuron" or "synapses"
        #       3. type of report: synaptic_current, neuron_voltage, neuron_fire, 
        #          input current, etc.
        #       4. Probability (the percentage of elements to report on)
	voltage_report_1=sim.addReport([group_1,group_2],"neuron","neuron_voltage",1.0,0.0,1.0)
	#An example of a report to file
	voltage_report_1.toAsciiFile("reg_voltage_report.txt");	

	#duration (in seconds) - each time step is 1 ms
	sim.run(duration=1.0)

	return

if __name__ == "__main__":
	run(sys.argv)
