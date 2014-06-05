#! /usr/bin/python

#izh and synapes

#import math
import os, sys
ncs_lib_path = ('../../../')
sys.path.append(ncs_lib_path)
import ncs

def run(argv):

	izh_parameters = {
		"a": .2,
		"b": .2,
		"c": -65,
		"d": ncs.Uniform(7.0, 9.0),
		"u": ncs.Uniform(-15.0, -11.0),
		"v": ncs.Normal(-60.0, 5.0),
		"threshold" : 30,
	}

	ex_synapse_parameters = {
		"utilization": ncs.Normal(0.05, 0.5),
		"redistribution": 1, 
                "last_prefire_time": 0,
                "last_postfire_time": 0,
                "tau_facilitation": .001,
                "tau_depression": .001,
                "tau_ltp": .015,
                "tau_ltd": .03,
                "A_ltp_minimum": .003,
                "A_ltd_minimum": .003,
                "tau_postsynaptic_conductance": .025,
                "psg_waveform_duration": .05,
		# playing parameters
                "reversal_potential": 0,
		"max_conductance": .004,
                "delay": ncs.Uniform(1,5)
	}

	
	sim=ncs.Simulation()

	izh_model = sim.addNeuron("model_izh", "izhikevich", izh_parameters)

	group1 = sim.addNeuronGroup("group1", 1, izh_model)
	group2 = sim.addNeuronGroup("group2", 1, izh_model)

	all_cells = sim.addNeuronAlias("allCells", [group1, group2])

	ex_syn_model = sim.addSynapse("ex_synapse", "ncs", ex_synapse_parameters)

	one_to_two = sim.addSynapseGroup("g1_to_g2", group1, group2, 1, "ex_synapse")

	if not sim.init(argv):
		print "Failed to initialize simulation."
		return


	sim.addStimulus("rectangular_current", { "amplitude": 25 }, group1, 1, 0.0, 0.3)
	# sim.addInput("rectangular_current", { "amplitude": 18.0 }, all_cells, 0.1, 0.0, 0.2)

	voltage_report1 = sim.addReport(group1, "neuron", "neuron_voltage", 1.0,0.0,0.3)
	voltage_report1.toAsciiFile("./juli_si_voltage_report_group1.txt")
	
	voltage_report2 = sim.addReport(group2, "neuron", "neuron_voltage", 1.0,0.0,0.3)
	voltage_report2.toAsciiFile("./juli_si_voltage_report_group2.txt")

	fire_report = sim.addReport(all_cells, "neuron", "neuron_fire", 1.0,0.0,0.3)
	fire_report.toAsciiFile("./juli_si_fire_report.txt")

	syn_current_report = sim.addReport(all_cells, "neuron", "synaptic_current", 1,0.0,0.3)
	syn_current_report.toAsciiFile("./juli_si_syn_current_report.txt")

	input_current_report=sim.addReport(all_cells, "neuron", "input_current", 1, 0.0,0.3)
	input_current_report.toStdOut();

	sim.run(duration=1.0)
	del sim
	return



if __name__=="__main__":
	run(sys.argv)	
