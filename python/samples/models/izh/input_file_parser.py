#!/usr/bin/python

# this only works with regular_spiking_izh.py

import os, sys
ncs_lib_path = ('../../../') #Path to ncs.py
sys.path.append(ncs_lib_path)
import ncs
from collections import namedtuple

def run(argv):
	#ncs.simulation() is required
	sim = ncs.Simulation()

	# open input file
	filename = 'test_input.txt'
	input = open(filename)

	# store entire file as a string, then split into word array
	text = input.read()

	input.close()

	# remove all white space from the text
	text = text.replace('\n', ' ');
	data = text.split();

	neuron_names = []
	neuron_types = []
	neuron_params = []
	#neuron = namedtuple("neuron", "names types params")
	neuron_group = namedtuple("neuron_group", "name num_cells params generator")
	synapse_names = []
	synapse_types = []
	synapse_params = []
	synapse_group = namedtuple("synapse_group", "connection group_1 group_2 probability params")
	stimulus = namedtuple("stimulus", "type params groups probability start_time end_time")
	report = namedtuple("report", "groups target type probability start_time end_time")

	# get neurons and synapses

	index = neuron_index = synapse_index = 0
	while (data[index + 1] == 'izhikevich' or data[index + 1] == 'ncs' or data[index + 1] == 'hh'):
		# HOW TO DIFFERENTIATE BETWEEN A NEURON AND A SYNAPSE?
		# assumes a synapse parameter will have 'synapse' in the name
		if (data[index].find("synapse") == -1):
			neuron_names.append(data[index])
			neuron_types.append(data[index + 1])

			i = index + 2
			neuron_params.append({})
			# temporary hard coded loop until exact file structure is known
			for x in range (0,7):
				neuron_params[neuron_index][data[i]] = float(data[i + 1])
				i+=2
			neuron_index+=1	

			print 'NEURON'
			print (neuron_names)
			print (neuron_types)
			print (neuron_params)	
			print '\n'

			index = i
		#else:
		#	synapse_names.append(data[index])
		#	synapse_types.append(data[index + 1])

		#	i = index + 2
		#	synapse_params.append([])
		#	while (data[i].find(":") != -1):
		#		if data[i].endswith(':'):
		#			synapse_params[synapse_index].append(data[i] + data[i + 1])
		#			i+=2
		#		else:
		#			synapse_params[synapse_index].append(data[i])
		#			i+=1
		#	synapse_index+=1	

		#	print 'SYNAPSE'
		#	print (synapse_names)
		#	print (synapse_types)
		#	print (synapse_params)	
		#	print '\n'

		#	index = i

	# get neuron groups
	neuron_groups = []
	while (data[index].find("group") != -1):
		# need method of determining which neuron parameters to use
		group = neuron_group(name = data[index], num_cells = int(data[index + 1]), params = data[index + 2], generator = data[index + 3]) 
		index+=4
		#print group
		neuron_groups.append(group);

	# get synapse groups
	print '\n'
	if synapse_names:
		synapse_groups = []
		# assumes a synapse parameter will have 'synapse' in the name
		while (data[index + 4].find("synapse") != -1) :
			#for name in synapse_names:
			#	if(data[index+4] == name):
			group = synapse_group(connection = data[index], group_1 = data[index + 1], group_2 = data[index + 2], probability = data[index + 3], params = data[index + 4])
			index+=5 
			#print group
			synapse_groups.append(group)


	# apply parameters to the simulator
	sim_neurons = []
	sim_neuron_groups = []

	for i in range (0, len(neuron_names)):
		sim_neurons.append(sim.addNeuron(neuron_names[i], neuron_types[i], neuron_params[i]))
		#print sim_neurons

	# add synapses	

	for i in range (0, len(neuron_groups)):
		# need method of determining which neuron parameters to use
		sim_neuron_groups.append(sim.addNeuronGroup(neuron_groups[i].name, neuron_groups[i].num_cells, sim_neurons[0], None))	

	# add synapse groups	

	# attempt to initialize simulation 
	if not sim.init(argv):
		print "failed to initialize simulation."
		return

	# get stimuli
	# this currently only works on a single stimulus
	stimuli = []
	param_entry = {data[index+1] : float(data[index+2])}
	group = stimulus(type = data[index], params = param_entry, groups = data[index+3], probability = float(data[index+4]), start_time = float(data[index+5]), end_time = float(data[index+6]))
	#print '\n'
	#print group
	stimuli.append(group)
	index+=7

	# add stimuli
	sim_stimuli = []
	sim_stimuli.append(sim.addStimulus(stimuli[0].type, stimuli[0].params, sim_neuron_groups[0], stimuli[0].probability, stimuli[0].start_time,stimuli[0].end_time))

	# get reports
	# this currently only works on a single report
	reports = []
	target_groups = []
	target_groups.append(data[index])
	#target_groups.append(data[index + 1])

	group = report(groups = target_groups, target = data[index + 1], type = data[index+2], probability = float(data[index+3]), start_time = float(data[index+4]), end_time = float(data[index+5]))
	#print group
	reports.append(group)
	index+=6

	# add reports
	sim_reports = []
	sim_reports.append(sim.addReport(reports[0].groups, reports[0].target, reports[0].type, reports[0].probability, reports[0].start_time, reports[0].end_time))
	temp = filename.split(".")
	report_name = temp[0] + "_report.txt"
	sim_reports[0].toAsciiFile(report_name);

	# get simulation run time
	runtime = float(data[index])

	# run the simulation
	sim.run(duration=runtime)

	return

if __name__ == "__main__":
	run(sys.argv)
