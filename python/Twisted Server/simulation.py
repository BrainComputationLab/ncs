# -*- coding: utf-8 -*-

import optparse, os, subprocess
from twisted.python import log
import json
import sys, ncs

from model import ModelService

DEBUG = True

class Simulation:

	sim = ncs.Simulation()
	modelService = ModelService()
	script = None
	script_str = ''

	def build_sim(self, params, username):

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

		# add initial contents to script string
		self.script_str += '#!/usr/bin/python\n'
		self.script_str += 'import os, sys\n'
		self.script_str += 'ncs_lib_path = ("../../../")\n'
		self.script_str += 'sys.path.append(ncs_lib_path)\n'
		self.script_str += 'import ncs' + '\n\n'
		self.script_str += 'if __name__ == "__main__":\n'
		self.script_str += '\tsim = ncs.Simulation()\n'

		#print self.script_str

		# this function takes dictionaries (converted json objects) and handles assigning neurons, synapses, and groups
		neuron_groups = []
		synapse_groups = []
		self.script_str = self.modelService.process_model(self.sim, json_model, neuron_groups, synapse_groups, self.script_str)
		#print self.script_str
		if DEBUG:
			print "ATTEMPTING TO INIT SIM..."

		self.script_str += '\tif not sim.init(sys.argv):\n\t\tprint "failed to initialize simulation."\n\t\treturn\n'
		#print self.script_str
		if not self.sim.init(sys.argv):
		    print "Failed to initialize simulation." # THIS SHOULD BE AN ERROR CALLBACK
		    log.msg("Failed to initialize simulation.")
		    return 

		if DEBUG: 
			print "ATTEMPTING TO ADD STIMS AND REPORTS"

		self.script_str = self.modelService.add_stims_and_reports(self.sim, json_sim_input_and_output, json_model, neuron_groups, synapse_groups, username, self.script_str)
		#print self.script_str
	def run_sim(self, params, ign):
		self.script = open("script.py", "w")
		self.script.write(self.script_str)
		self.script.close()
		#subprocess.call("./script.py", shell=True)
		#os.remove("script.py")