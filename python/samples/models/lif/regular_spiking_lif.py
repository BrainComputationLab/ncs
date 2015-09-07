#! /usr/bin/python

import math
import os, sys
ncs_lib_path = ('../../../')
sys.path.append(ncs_lib_path)
import ncs

def run(argv):
	
	calcium_channel = {
		#type of channel
		"type": "calcium_dependent",
		#starting value, Unit: none
		"m_initial": 0.0,
		#Unit: mV
		"reversal_potential": -80,
		"m_power": 2,
		#conductance = unitary_g * strength, unit = pS/(cm^2)
		"conductance": 6.0*0.025,
		"forward_scale": 0.000125,
		"forward_exponent": 2,
		"backwards_rate": 2.5,
		"tau_scale": 0.01,

	}


	ncs_cell = {
		"threshold": -47.0,
		"resting_potential": -65.0,
		#initial calcium concentration (required. default 0)
		"calcium": 5.0,
		#increment calcium concentration by this value each time the cell spikes
		#required. default 0
		"calcium_spike_increment": 100.0,
		"tau_calcium": 0.08,
		#decay time for voltage, Unit:seconds
		"tau_membrane": 0.020,
		#resistance, Unit ohms
		"r_membrane": 180,
		"leak_reversal_potential": 0.0,
		"leak_conductance": 0.0,
		#spike template
		"spike_shape": [-38, 30, -38, -43],
		#channels
		"channels":[
			calcium_channel,
		],
		"capacitance": 1.0,
		
	}


	sim=ncs.Simulation()
	
	neuron_parameters = sim.addNeuron("ncs_neuron","ncs",ncs_cell)
	group_1 = sim.addNeuronGroup("group_1",1,"ncs_neuron",None)	

	#initialize
	if not sim.init(argv):
		print "Failed to initialize simulation"
		return
	
	#stimulus 
	#sim.addInput("linear_current",{"starting_amplitude":1.98, "ending_amplitude":1.98,"width":0.3,"time_increment":0.01,"dyn_range": ncs.Uniform(25,60)},group_1,1,0.02,1.0)
	sim.addStimulus("linear_current",{"starting_amplitude": 1.98,"ending_amplitude":1.98},group_1,1,0.02,1.0)

	voltage_report = sim.addReport("group_1","neuron", "neuron_voltage", 1.0,0.0,1.0)
	voltage_report.toAsciiFile("./reg_voltage.txt")
	current_report = sim.addReport("group_1","neuron", "input_current", 1.0,0.0,1.0)
	current_report.toAsciiFile("./reg_current.txt")	
	

	sim.run(duration=1.0)
	del sim
	return

if __name__ == "__main__":
	run(sys.argv)

