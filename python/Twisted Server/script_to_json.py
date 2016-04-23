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

            # get sim variable name
            sim = ''
            for line in line_list:
            	if 'ncs.Simulation()' in line:
            		space_free = re.sub(r'\s', '', line)
            		sim = space_free.split('=ncs.Simulation()')[0]

            # replace build function calls with parse functions
            insert_index = None
            num_tabs = 0
	    num_spaces = 0
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
		    num_spaces = len(line) - len(line.lstrip())
                else:
                    new_line_list.append(line)

            # append appropriate number of tabs
            tabs = ''
	    if num_tabs != 0:
		for i in range(num_tabs):
			tabs += '\t'
	    else:
		for i in range(num_spaces):
			tabs += ' '

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

		if DEBUG:
			file = open("pyncs_params.txt", "w")
			file.write(json.dumps(script_data, sort_keys=True, indent=2) + '\n\n\n')
			file.close()

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

	        	# assign neuron parameters
	        	if neuron['parameters']['type'] == 'izhikevich':
					spec = {
					    "a": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "b": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "c": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "d": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "u": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "v": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					    "threshold": {
					        "maxValue": 0, 
					        "mean": 0, 
					        "minValue": 0, 
					        "stddev": 0, 
					        "type": "exact", 
					        "value": 0
				      	},
					}

					for key, value in neuron['parameters'].iteritems():
						spec[key] = value

					spec['type'] = 'Izhikevich'

			elif neuron['parameters']['type'] == 'ncs':

	                	spec = {
			          "calcium": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "calcium_spike_increment": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "capacitance": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "leak_conductance": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value":30
			          }, 
			          "leak_reversal_potential": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "r_membrane": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "resting_potential": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          },  
			          "tau_calcium": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "tau_membrane": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "threshold": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }
	                	}

	                	spike_shape_vals = []
				for key, value in neuron['parameters'].iteritems():
					if key == 'spike_shape':
						for val in value:
							spike_shape_vals.append(val['value'])

					spec[key] = value

				if not spike_shape_vals:
					spike_shape_vals.append(0)

				spec["spike_shape"] = spike_shape_vals

	                	channels = []
	                	if 'channels' in neuron['parameters']:
					del spec['channels']
	                		for ch in neuron['parameters']['channels']:

						if ch['type'] == 'calcium_dependent':
	                				channel = {
							      "backwards_rate": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "className": "calciumDependantChannel", 
							      "conductance": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "description": "Description", 
							      "forward_exponent": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "forward_scale": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "m_initial": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "m_power": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "name": "Calcium Dependant Channel", 
							      "reversal_potential": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "tau_scale": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }
							    }

							for ch_key, ch_val in ch.iteritems():
					            		channel[ch_key] = ch_val

							del channel['type']

					            	channels.append(channel)

						elif ch['type'] == 'voltage_gated_ion':
	                				channel = {
							      "activation_slope": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "className": "voltageGatedIonChannel", 
							      "conductance": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "deactivation_slope": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "description": "Description", 
							      "equilibrium_slope": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "m_initial": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "name": "Voltage Gated Ion Channel", 
							      "r": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "reversal_potential": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }, 
							      "v_half": {
							        "maxValue": 0, 
							        "mean": 0, 
							        "minValue": 0, 
							        "stddev": 0, 
							        "type": "exact", 
							        "value": 0
							      }
							    }

							for ch_key, ch_val in ch.iteritems():
					            		channel[ch_key] = ch_val

							del channel['type']

							channels.append(channel)

				spec['channel'] = channels 

				spec['type'] = 'NCS'

			elif neuron['parameters']['type'] == 'hh':

	        		spec = {
			          "capacitance": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "resting_potential": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }, 
			          "threshold": {
			            "maxValue": 0, 
			            "mean": 0, 
			            "minValue": 0, 
			            "stddev": 0, 
			            "type": "exact", 
			            "value": 0
			          }
	        		}

				for key, value in neuron['parameters'].iteritems():
					spec[key] = value

				spec['type'] = 'HodgkinHuxley'

				spec['channel'] = []

				if 'channels' in neuron['parameters']:
					del spec['channels']
					for ch in neuron['parameters']['channels']:
						channel = {
						      "className": "voltageGatedChannel",  
						      "description": "Description", 
						      "name": "Voltage Gated Channel"
						    }

						for key, value in ch.iteritems():
							channel[key] = value

						particles = []
						if 'particles' in ch:
							for p in ch['particles']:
								particle = {
									"alpha": {
									  "a": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "b": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "c": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "d": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "f": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "h": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }
									}, 
									"beta": {
									  "a": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "b": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "c": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "d": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "f": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }, 
									  "h": {
									    "maxValue": 0, 
									    "mean": 0, 
									    "minValue": 0, 
									    "stddev": 0, 
									    "type": "exact", 
									    "value": 0
									  }
									}, 
									"className": "voltageGatedParticle", 
									"m_power": {
									  "maxValue": 0, 
									  "mean": 0, 
									  "minValue": 0, 
									  "stddev": 0, 
									  "type": "exact", 
									  "value": 0
									}, 
									"power": {
									  "maxValue": 0, 
									  "mean": 0, 
									  "minValue": 0, 
									  "stddev": 0, 
									  "type": "exact", 
									  "value": 0
									}, 
									"x_initial": {
									  "maxValue": 0, 
									  "mean": 0, 
									  "minValue": 0, 
									  "stddev": 0, 
									  "type": "exact", 
									  "value": 0
									}
								      }

								for p_key, p_value in p.iteritems():
									if 'type' in p_value:
										del p_value['type']
						      			particle[p_key] = p_value
						      			
								del particle['type']
						      		particles.append(particle)

						channel['particles'] = particles
						del channel['type']
						spec['channel'].append(channel)

			neurons.append({"classification": "cells", 
          						"description": "Description", 
          						"geometry": neuron['geometry'], 
          						"name": name, 
          						"num": neuron['num'], 
          						"parameters": spec
          						})

	        # add synapses
	        count = 0
	        script_synapses = script_data['model']['synapse_groups']
		for synape_name, synapse in script_synapses.iteritems():

	        	count += 1

	        	# get synapse parameters
	        	if synapse['parameters']['type'] == 'flat':

				spec = {
				        "current": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "delay": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "name": "flatSynapse"
				      }

				for key, value in synapse['parameters'].iteritems():
					spec[key] = value

				spec['name'] = "flatSynapse" + str(count)
				spec['type'] = 'Flat'

	        	elif synapse['parameters']['type'] == 'ncs':

				spec = {
				        "A_ltd_minimum": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "A_ltp_minimum": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "delay": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "last_postfire_time": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "last_prefire_time": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "max_conductance": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        },  
				        "psg_waveform_duration": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "redistribution": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "reversal_potential": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "tau_depression": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "tau_facilitation": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "tau_ltd": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "tau_ltp": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "tau_postsynaptic_conductance": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }, 
				        "utilization": {
				          "maxValue": 0, 
				          "mean": 0, 
				          "minValue": 0, 
				          "stddev": 0, 
				          "type": "exact", 
				          "value": 0
				        }
				      }

				for key, value in synapse['parameters'].iteritems():
					spec[key] = value

				spec['name'] = "ncsSynapse" + str(count)
	        		spec['type'] = 'NCS'

	        		spec["a_ltd_minimum"] = spec.pop("A_ltd_minimum")
	        		spec["a_ltp_minimum"] = spec.pop("A_ltp_minimum")
	        		spec["tau_post_synaptic_conductance"] = spec.pop("tau_postsynaptic_conductance")

	        	synapses.append({"classification": "synapseGroup", 
       							 "description": "Description", 
								 "parameters": spec,
								 "post": synapse['postsynaptic'][0]['parameters']['name'], 
								 "postPath": [{
								  	"index": 0, 
								  	"name": "Home"
									}], 
								"pre": synapse['presynaptic'][0]['parameters']['name'],
								"prePath": [{
								  	"index": 0, 
								  	"name": "Home"
									}], 
								"prob": synapse['probability']
	          					})

	        # add stimulus
	        script_stimuli = script_data['simulation']['inputs']
	        for index, stimulus in enumerate(script_stimuli):

	        	# convert stimulus types
	        	stimulus["stimulusType"] = stimulus['parameters']['type']
	        	del stimulus['parameters']['type']
	        	if stimulus['stimulusType'] == 'rectangular_current':
	        		stimulus['stimulusType'] = 'Rectangular Current'
	        	elif stimulus['stimulusType'] == 'rectangular_voltage':
	        		stimulus['stimulusType'] = 'Rectangular Voltage'
	        	elif stimulus['stimulusType'] == 'linear_current':
	        		stimulus['stimulusType'] = 'Linear Current'
	        	elif stimulus['stimulusType'] == 'linear_voltage':
	        		stimulus['stimulusType'] = 'Linear Voltage'
	        	elif stimulus['stimulusType'] == 'sine_current':
	        		stimulus['stimulusType'] = 'Sine Current'
	        	elif stimulus['stimulusType'] == 'sine_voltage':
	        		stimulus['stimulusType'] = 'Sine Voltage'

			spec = {
			      "amplitude": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "amplitude_scale": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "amplitude_shift": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "current": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "delay": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      },  
			      "end_amplitude": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "frequency": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      },  
			      "phase": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "probability": 0.5, 
			      "startTime": 0, 
			      "start_amplitude": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "time_scale": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }, 
			      "width": {
			        "maxValue": 0, 
			        "mean": 0, 
			        "minValue": 0, 
			        "stddev": 0, 
			        "type": "exact", 
			        "value": 0
			      }
	        	}

	        	for key, val in stimulus['parameters'].iteritems():
	        		spec[key] = val

	        	stimuli.append({"className": "simulationInput", 
      							"endTime": str(stimulus['end_time']), 
      							"inputTargets": stimulus['group_names'], 
      							"name": "Input" + str(index + 1), 
      							"probability": stimulus['probability'], 
      							"startTime": stimulus['start_time'], 
      							"stimulusType": stimulus['stimulusType'], 
      							"parameters": spec
    							})

	        # add reports
	        script_reports = script_data['simulation']['outputs']
	        for index, report in enumerate(script_reports):

	        	# convert report types
	        	if report['report_type'] == 'neuron_voltage':
	        		report['report_type'] = 'Neuron Voltage'
	        	elif report['report_type'] == 'synaptic_current':
	        		report['report_type'] = 'Synaptic Current'
	        	elif report['report_type'] == 'neuron_fire':
	        		report['report_type'] = 'Neuron Fire'
	        	elif report['report_type'] == 'input_current':
	        		report['report_type'] = 'Input Current'

	        	if report['target_type'] == 'synapses':
	        		target_report_type = 3
	        	else:
	        		target_report_type = 1

		        reports.append({"className": "simulationOutput", 
						        "endTime": str(report['end_time']), 
						      	"name": "Output" + str(index + 1),  
						      	"possibleReportType": target_report_type, 
						      	"probability": report['probability'], 
						      	"reportTargets": report['target_names'], 
						      	"reportType": report['report_type'], 
						      	"startTime": report['start_time']
    							})	

	        sim['duration'] = script_data['simulation']['run']['duration']

	        # convert keys to NCB format
	        for neuron in neurons:
	        	if neuron['parameters']:
	        		self.convert_keys_to_ncb_input(neuron['parameters'])
	        		if 'channel' in neuron['parameters']:
		        		for channel in neuron['parameters']['channel']:
		        			self.convert_keys_to_ncb_input(channel)
						if 'particles' in channel:
		        				for particle in channel['particles']:
		        					self.convert_keys_to_ncb_input(particle)
	        for synapse in synapses:
	        	if synapse['parameters']:
	        		self.convert_keys_to_ncb_input(synapse['parameters'])
	        '''for stimulus in stimuli:
	        	if stimulus['parameters']:
	        		self.convert_keys_to_ncb_input(stimulus['parameters'])'''

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
