# -*- coding: utf-8 -*-

import optparse, os, subprocess, stat, sys
from subprocess import Popen, PIPE
from twisted.python import log
from twisted.internet import reactor
import json, ncs, uuid

from sim_subprocess import SubProcessProtocol
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

		if DEBUG:
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

		# takes the model and simulation dictionaries and builds a simulation script
		neuron_groups = []
		synapse_groups = []
		self.script_str = self.modelService.process_model(self.sim, json_model, neuron_groups, synapse_groups, self.script_str)

		self.script_str += '\tif not sim.init(sys.argv):\n\t\tprint "failed to initialize simulation."\n\t\tsys.exit(1)\n'

		self.script_str = self.modelService.add_stims_and_reports(self.sim, json_sim_input_and_output, json_model, neuron_groups, synapse_groups, username, self.script_str)

	def run_sim(self, params, ign, script_file):
		self.script = open(script_file, "w")
		self.script.write(self.script_str)
		self.script.close()

		st = os.stat(script_file)
		os.chmod(script_file, st.st_mode | stat.S_IEXEC)

		# set the subprocess environment to the location of pyncs
		pp = SubProcessProtocol()
		pyncs_env = os.environ.copy()
		pyncs_env["PATH"] = '../../../../' + pyncs_env["PATH"]

		# run the simulation script as a subprocess
		reactor.spawnProcess(pp, "./" + script_file, env=pyncs_env)