# -*- coding: utf-8 -*-

import optparse, os
from twisted.python import log
import json
import sys, ncs

from model import ModelService

DEBUG = True

class Simulation:

	sim = ncs.Simulation()
	modelService = ModelService()

	def build_sim(self, params):

		json_model = params['model']
		json_sim_input_and_output = params['simulation']

		# temporarily write the JSON to a file for comparision
		file = open("json_recvd.txt", "w")
		file.write(json.dumps(params, sort_keys=True, indent=2) + '\n\n\n')
		file.write('MODEL:\n')
		file.write(json.dumps(json_model, sort_keys=True, indent=2) + '\n\n\n')
		file.write('SIMULATION:\n')
		file.write(json.dumps(json_sim_input_and_output, sort_keys=True, indent=2) + '\n')
		file.close()

		# this function takes dictionaries (converted json objects) and handles assigning neurons, synapses, and groups
		neuron_groups = []
		synapse_groups = []

		self.modelService.process_model(self.sim, json_model, neuron_groups, synapse_groups)

		if DEBUG:
			print "ATTEMPTING TO INIT SIM..."

		if not self.sim.init(sys.argv):
		    print "Failed to initialize simulation." # THIS SHOULD BE AN ERROR CALLBACK
		    log.msg("Failed to initialize simulation.")
		    return 

		if DEBUG: 
			print "ATTEMPTING TO ADD STIMS AND REPORTS"

		self.modelService.add_stims_and_reports(self.sim, json_sim_input_and_output, json_model, neuron_groups, synapse_groups)

	def run_sim(self, params):
		#self.sim.run(duration=1.0)
		pass