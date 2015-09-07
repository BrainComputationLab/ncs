#!/usr/bin/python

import os, sys
#specify the location of ncs.py in ncs_lib_path
ncs_lib_path = ('../')
sys.path.append(ncs_lib_path)
import ncs

import json

def run(argv):
	sim = ncs.Simulation()
	regular_spiking_parameters = sim.parseNeuron("regular_spiking","izhikevich",
								{
								 "a": 0.02,
								 "b": 0.2,
								 "c": -65.0,
								 "d": 8.0,
								 "u": -12.0,
								 "v": ncs.Normal(0.5, 0.05),
								 "threshold": 30
								})

	group_1 = sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
	
	if not sim.parseInit(argv):
		print "failed to initialize simulation."
		return

	input_parameters = {
				"amplitude":10
			   }


	sim.parseStimulus("rectangular_current", input_parameters, group_1, 1, 0.01, 1.0)
	report = sim.parseReport("group_1", "neuron", "neuron_voltage", 1, 0.0, 1.0)
	report.parseToAsciiFile("./PARSE_TEST.txt")	

	sim.parseRun(duration=1.0)


	sim_spec = {}
	sim_spec['inputs'] = sim.parse_stim_spec
	sim_spec['outputs'] = sim.parse_report_spec
	sim_spec['run'] = sim.parse_run_spec
	sim_json = {'model': sim.parse_model_spec, 'simulation': sim_spec}
	json_file = open('test_json.json', 'w')
	json_file.write(json.dumps(sim_json, sort_keys=True, indent=2) + '\n\n\n')

	return

if __name__ == "__main__":
	run(sys.argv)
