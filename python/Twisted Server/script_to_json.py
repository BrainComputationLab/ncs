# -*- coding: utf-8 -*-

import os, stat, sys
import optparse, subprocess
from subprocess import Popen, PIPE
import ncs
from twisted.python import log
from twisted.internet import reactor
from twisted.internet.defer import Deferred
import json, uuid, re, types

from sim_subprocess import SubProcessProtocol

DEBUG = True

class Parser:

	script_name = None
	sim_params = None
	report_paths = {}
	reports_list = []

	def __init__(self, script_str):
		self.script_name = str(uuid.uuid4()) + '.py'
		script_file = open(self.script_name, 'w')
		script_file.write(script_str)
		script_file.close()

	def modify_script_file(self):
            input = open(self.script_name)
            line_list = input.readlines()
            input.close()

            # get sim variable name and report path
            sim = ''
            for line in line_list:
            	if 'ncs.Simulation()' in line:
            		space_free = re.sub(r'\s', '', line)
            		sim = space_free.split('=ncs.Simulation()')[0]

            	elif 'toAsciiFile' in line:
            		space_free = re.sub(r'\s', '', line)
            		report_obj = space_free.split('.toAsciiFile')[0]
            		self.report_paths[report_obj] = re.findall(r'"(.*?)"', line)[0]

            # replace build function calls with parse functions
            insert_index = None
            num_tabs = 0
            new_line_list = []
            for index, line in enumerate(line_list):
                if sim + '.addNeuron' in line:
                    new_line_list.append(line.replace('addNeuron', 'parseNeuron'))
                elif sim + '.addSynapse' in line:
                    new_line_list.append(line.replace('addSynapse', 'parseSynapse'))
                elif sim + '.init' in line:
                    new_line_list.append(line.replace('init', 'parseInit'))
                elif sim + '.addStimulus' in line:
                    new_line_list.append(line.replace('addStimulus', 'parseStimulus'))
                elif sim + '.addReport' in line:
        	    space_free = re.sub(r'\s', '', line)
    		    self.reports_list.append(space_free.split('=' + sim +'.addReport')[0])
                    new_line_list.append(line.replace('addReport', 'parseReport'))
                elif 'toAsciiFile' in line:
                    new_line_list.append(line.replace('toAsciiFile', 'parseToAsciiFile'))
            	elif sim + '.run' in line:
                    new_line_list.append(line.replace('run', 'parseRun'))
		    insert_index = index + 1
		    num_tabs = len(line) - len(line.lstrip('\t'))
                else:
                    new_line_list.append(line)

            # append appropriate number of tabs
            tabs = ''
            for i in range(num_tabs):
            	tabs += '\t'

            # insert lines that will output the sim structure to a file
            new_line_list[insert_index:insert_index] = [tabs + 'import json\n']
            insert_index += 1
            new_line_list[insert_index:insert_index] = [tabs + 'sim_spec = {}\n']
            insert_index += 1
            new_line_list[insert_index:insert_index] = [tabs + 'sim_spec["inputs"] = sim.parse_stim_spec\n']
            insert_index += 1
            new_line_list[insert_index:insert_index] = [tabs + 'sim_spec["outputs"] = sim.parse_report_spec\n']
            insert_index += 1
            new_line_list[insert_index:insert_index] = [tabs + 'sim_spec["run"] = sim.parse_run_spec\n']
            insert_index += 1
    	    new_line_list[insert_index:insert_index] = [tabs + 'sim_json = {"model": sim.parse_model_spec, "simulation": sim_spec}\n']
            insert_index += 1
    	    new_line_list[insert_index:insert_index] = [tabs + 'json_file = open("' + self.script_name.split('.py')[0] + '.json", "w")\n']
            insert_index += 1
    	    new_line_list[insert_index:insert_index] = [tabs + 'json_file.write(json.dumps(sim_json, sort_keys=True, indent=2) + "\\n\\n\\n")\n']

	    # create modified file
	    script_file = open(self.script_name, "w")
	    for line in new_line_list:
	        script_file.write(line)
	    script_file.close()
	    
	    d = Deferred()
	    d.addCallback(self.create_json_file)
	    d.callback('ign')
	    return d

	def create_json_file(self, ign):
		st = os.stat(self.script_name)
		os.chmod(self.script_name, st.st_mode | stat.S_IEXEC)

		# set the subprocess environment to the location of pyncs
		pp = SubProcessProtocol()
		pyncs_env = os.environ.copy()
		pyncs_env["PATH"] = '../../../../' + pyncs_env["PATH"]

		# run the simulation script as a subprocess
		reactor.spawnProcess(pp, "./" + self.script_name, env=pyncs_env)
		
	def build_ncb_json(self):
            json_file = open(self.script_name.split('.py')[0] + '.json')
	    try:
	        script_data = json.load(json_file)
	    except ValueError:
                log.msg('Could not load JSON file.')
	        script_data = {} 

            if script_data:
	    	self.sim_params = {
		          "model": {
		            "author": "", 
		            "cellAliases": [], 
		            "cellGroups": {
		              "cellGroups": [], 
		              "classification": "cellGroup", 
		              "description": "Description", 
		              "name": "Home"
		            }, 
		            "classification": "model", 
		            "description": "Description", 
		            "name": "Current Model", 
		            "synapses": []
		          }, 
		          "simulation": {
		            "duration": None, 
		            "fsv": None, 
		            "includeDistance": "No", 
		            "inputs": [], 
		            "interactive": "No", 
		            "name": "sim", 
		            "outputs": [], 
		            "seed": None
		          }
		        }
	        model = self.sim_params['model']
	        sim = self.sim_params['simulation']
	        neurons = model['cellGroups']['cellGroups']
	        synapses = model['synapses']
	        stimuli = sim['inputs']
	        reports = sim['outputs']

	        # add neurons
	        script_neurons = script_data['model']['neuron_groups']
	        for group_name, neuron in script_neurons.iteritems():
	        	name = neuron['parameters']['name']
	        	del neuron['parameters']['name']
	        	neurons.append({"$$hashKey": "052", 
          						"classification": "cells", 
          						"description": "Description", 
          						"geometry": neuron['geometry'], 
          						"name": name, 
          						"num": neuron['num'], 
          						"parameters": neuron['parameters']
          						})

	        # add synapses
	        # DOES NCB WANT MORE INFORMATION THEN JUST THE NAMES OF THE PRE AND POST?
	        script_synapses = script_data['model']['synapse_groups']
	        for synape_name, synapse in script_synapses.iteritems():
	        	synapses.append({"$$hashKey": "05V", 
								 "classification": "synapseGroup", 
       							 "description": "Description", 
								 "parameters":synapse['parameters'],
								 "post": synapse['postsynaptic'][0]['parameters']['name'], 
								 "postPath": [{
								  	"$$hashKey": "05R", 
								  	"index": 0, 
								  	"name": "Home"
									}], 
								"pre": synapse['presynaptic'][0]['parameters']['name'],
								"prePath": [{
								  	"$$hashKey": "05N", 
								  	"index": 0, 
								  	"name": "Home"
									}], 
								"prob": synapse['probability']
	          					})

	        # add stimulus
	        script_stimuli = script_data['simulation']['inputs']
	        for index, stimulus in enumerate(script_stimuli):
	        	stimuli.append({"$$hashKey": "05L",  
      							"className": "simulationInput", 
      							"endTime": stimulus['end_time'], 
      							"inputTarget": stimulus['group_names'], 
      							"name": "Input" + str(index + 1), 
      							"probability": stimulus['probability'], 
      							"startTime": stimulus['start_time'], 
      							"stimulusType": stimulus['parameters']['type'], 
      							"parameters": stimulus['parameters']
    							})

	        # add reports
	        script_reports = script_data['simulation']['outputs']
	        for index, report in enumerate(script_reports):
	        	if self.reports_list[index] in self.report_paths:
		        	reports.append({"$$hashKey": "05O", 
							        "className": "simulationOutput", 
							        "endTime": report['end_time'], 
							      	"fileName": self.report_paths[self.reports_list[index]], 
							      	"name": "Output" + str(index + 1), 
							      	"numberFormat": "ascii", 
							      	"outputType": "Save As File", 
							      	"probability": report['probability'], 
							      	"reportTarget": report['target_names'], 
							      	"reportType": report['report_type'], 
							      	"startTime": report['start_time']
	    							})
		        else:
		        	reports.append({"$$hashKey": "05O", 
							        "className": "simulationOutput", 
							        "endTime": report['end_time'], 
							      	"fileName": "", 
							      	"name": "Output" + str(index + 1), 
							      	"numberFormat": "ascii", 
							      	"outputType": "", 
							      	"probability": report['probability'], 
							      	"reportTarget": report['target_names'], 
							      	"reportType": report['report_type'], 
							      	"startTime": report['start_time']
	    							})	

	        sim['duration'] = script_data['simulation']['run']['duration']

	        # convert keys to NCB format
	        for neuron in neurons:
	        	if neuron['parameters']:
	        		self.convert_keys_to_ncb_input(neuron['parameters'])
	        for synapse in synapses:
	        	if synapse['parameters']:
	        		self.convert_keys_to_ncb_input(synapse['parameters'])
	        for stimulus in stimuli:
	        	if stimulus['parameters']:
	        		self.convert_keys_to_ncb_input(stimulus['parameters'])

	    d = Deferred()
	    d.callback('ign')
	    return d

	def delete_files(self, ign):
		try:
			os.remove(self.script_name)
			if DEBUG:
				print 'Removed script file.'
		except OSError:
			log.msg("Could not delete script file.")

		json_file_name = self.script_name.split('.py')[0] + '.json'
		try:
			os.remove(json_file_name)
			if DEBUG:
				print 'Removed json file.'
		except OSError:
			log.msg("Could not delete json file.")

		d = Deferred()
	        d.callback(self)
	        return d

	def convert_keys_to_ncb_input(self, dict):
		for key, value in dict.iteritems():
				if '_' in key:
					dict[self.underscore_to_camelcase(key)] = dict.pop(key)
					
				if type(value) is types.DictType:
					self.convert_keys_to_ncb_input(value)


	def underscore_to_camelcase(self, str):
	    components = str.split('_')
	    return components[0] + "".join(x.title() for x in components[1:])
