#!/usr/bin/python

import os, sys
ncs_lib_path = ('../../../')
sys.path.append(ncs_lib_path)
import ncs

def run(argv):
	sim = ncs.Simulation()
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
	
	flat_synapse_parameters = sim.addSynapse("synapse","ncs",{
										
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
									"max_conductance": 0.4,
									"reversal_potential":0.0,
									"tau_postsynaptic_conductance": 0.025,
									"psg_waveform_duration": 0.05,
									"delay": 1,
								})	

	group_1=sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
	group_2=sim.addNeuronGroup("group_2",1,regular_spiking_parameters,None)
	
	connection1=sim.addSynapseGroup("connection1","group_1","group_2",1,"synapse");

	
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	sim.addStimulus("rectangular_current",{"amplitude":10,"width": 1, "frequency": 1},group_1,1,0.01,1.0)
	voltage_report1=sim.addReport("group_1","neuron","neuron_voltage",1.0,0.0,0.01)
	voltage_report1.toAsciiFile("./group1_synapse_model.txt")	
	voltage_report2=sim.addReport("group_2","neuron","neuron_voltage",1.0,0.0,0.01)
	voltage_report2.toAsciiFile("./group2_synapse_model.txt");	
	sim.run(duration=0.01)

	return

if __name__ == "__main__":
	run(sys.argv)
