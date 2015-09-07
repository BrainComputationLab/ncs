# -*- coding: utf-8 -*-

import os, stat, sys
from twisted.python import log
from twisted.internet import reactor
import json, uuid, re

from sim_subprocess import SubProcessProtocol

class Parser:

	script_name = None
	sim_params = None

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
            report_paths = []
            for line in line_list:
            	if 'ncs.Simulation()' in line:
            		space_free = re.sub(r'\s', '', line)
            		sim = space_free.split('=ncs.Simulation()')[0]

            	elif 'toAsciiFile' in line:
            		report_paths.append(re.findall(r'"(.*?)"', line)[0])

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
            new_line_list[:0] = ['import json\n']

	    # create modified file
	    script_file = open(self.script_name, "w")
	    for line in new_line_list:
	        script_file.write(line)
	    script_file.close()

	def create_json_file(self):

	    st = os.stat(self.script_name)
	    os.chmod(self.script_name, st.st_mode | stat.S_IEXEC)

	    # set the subprocess environment to the location of pyncs
	    pp = SubProcessProtocol()
	    pyncs_env = os.environ.copy()
	    pyncs_env["PATH"] = '../../../../' + pyncs_env["PATH"]

	    # run the simulation script as a subprocess
	    reactor.spawnProcess(pp, "./" + self.script_name, env=pyncs_env)
		
	def build_ncb_json(self, file):
			# REMOVE FILE INPUT PARAM AFTER TESTING
            #json_file = open(self.script_name.split('.py')[0] + '.json')
            json_file = open(file)
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
	        # FIGURE OUT HOW TO GET THE REPORT PATH WITH THE REPORT IN NCS.PY
	        script_reports = script_data['simulation']['outputs']
	        for index, report in enumerate(script_reports):
	        	reports.append({"$$hashKey": "05O", 
						        "className": "simulationOutput", 
						        "endTime": report['end_time'], 
						      	"fileName": "report.txt", 
						      	"name": "Output" + str(index + 1), 
						      	"numberFormat": "ascii", 
						      	"outputType": "Save As File", 
						      	"probability": report['probability'], 
						      	"reportTarget": report['target_names'], 
						      	"reportType": report['report_type'], 
						      	"startTime": report['start_time']
    							})

	        sim['duration'] = script_data['simulation']['run']['duration']

	        # TESTING
	        converted_json = open('converted_json.json', 'w')
	        converted_json.write(json.dumps(self.sim_params, sort_keys=True, indent=2) + "\n\n\n")


	def delete_files(self):
		try:
			os.remove(self.script_name)
			print 'Removed script file.'
		except OSError:
			log.msg("Could not delete script file.")

		json_file_name = self.script_name.split('.py')[0] + 'json'
		try:
			os.remove(json_file_name)
			print 'Removed json file.'
		except OSError:
			log.msg("Could not delete json file.")

# testing testing
if __name__ == "__main__":

	input = open('../samples/models/izh/regular_spiking_synapse.py')
	text = input.read()
	input.close()

	parser = Parser(text)

	#parser.modify_script_file()

	parser.build_ncb_json('ae75ca9d-19d9-4354-9878-f5f1fca14fe9.json')
