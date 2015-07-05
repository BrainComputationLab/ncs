#from __future__ import absolute_import, unicode_literals
import os, sys, re, ast
from string import whitespace
#specify the location of ncs.py in ncs_lib_path
ncs_lib_path = ('../')
sys.path.append(ncs_lib_path)
import ncs
class ModelService(object):

    @classmethod
    def sim_test(cls):
        sim = ncs.Simulation()
        regular_spiking_parameters = sim.addNeuron("regular_spiking","izhikevich",
                                {
                                 "a": 0.02,
                                 "b": 0.2,
                                 "c": -65.0,
                                 "d": 8.0,
                                 "u": -12.0,
                                 "v": -60.0,
                                 "threshold": 30
                                })
        group_1=sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
    
        if not sim.init(sys.argv):
            print "failed to initialize simulation."
            return

        input_parameters = {
                "amplitude":10
               }

        sim.addStimulus("rectangular_current", input_parameters, group_1, 1, 0.01, 1.0)
        voltage_report=sim.addReport("group_1", "neuron", "neuron_voltage", 1, 0.0, 1.0)
        voltage_report.toAsciiFile("./regular_spiking_izh.txt") 

        sim.run(duration=1.0) 

        return
    @classmethod
    def parenthetic_contents(cls, string):
        """Generate parenthesized contents in string as pairs (level, contents)."""
        stack = []
        for i, c in enumerate(string):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                start = stack.pop()
                yield [len(stack), string[start + 1: i]]

    @classmethod
    def script_to_JSON(cls, script_file):

        # TODO: Handle Aliases (story of my life...)

        sim_params = {
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
        model = sim_params['model']
        sim = sim_params['simulation']

        neurons = model['cellGroups']['cellGroups']
        synapses = model['synapses']
        stimuli = sim['inputs']
        reports = sim['outputs']

        # read lines into a list
        input = open(script_file)
        line_list = input.readlines()
        input.close()

        # find all comments
        comments = []
        for line in line_list:
          if '#' in line:
            comments.append('#' + line.split('#')[1])

        lines = []
        var_val_dict = {}
        nested_list = []
        iterator = iter(line_list)
        for line in iterator:
            if '{' in line:
                string = re.sub(r'\s', '', line)
                count = 1

                while count != 0:
                    try:
                        line = iterator.next()
                        if '{' in line:
                          count += 1

                        if '}' in line:
                          count -= 1

                        # don't add comments
                        if '#' in line:
                          line = line.split('#')[0]

                        string += re.sub(r'\s', '', line)
                    except StopIteration:
                        break
                lines.append(string)
                var_val = string.split('=')
                if len(var_val) > 1:

                    # replace any keys in this string with ones values already stored
                    for key, val in var_val_dict.iteritems():
                      if key in var_val[1]:
                        var_val[1] = var_val[1].replace(key, val)
                    var_val_dict[var_val[0]] = var_val[1]

            else:
                lines.append(line)

        # store entire file as string
        input = open(script_file)
        text = input.read()
        input.close()

        # remove comments
        for comment in comments:
            text = text.replace(comment, '')

        # remove all white space from the text
        text = re.sub(r'\s', '', text)

        # replace all variables used with their assigned values
        for key, value in var_val_dict.iteritems():
            text = text.replace(key, value)

        #print text
        file = open("text.txt", "w")
        file.write("%s\n" % text)
        file.close()

        # MODEL
        # search for neurons
        neuron_params = text.split('addNeuron(')
        for i in range(1, len(neuron_params)):
            print 'Adding neuron...'

            # this creates a list of the nesting level and the contents within parentheses
            vals = list(cls.parenthetic_contents('(' + neuron_params[i]))

            for val in vals:
                if val[0] == 0:
                    # This splits the string by commas, but excludes those in parentheses or brackets (normal/uniform parameters)
                    params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
                    params = params.findall(val[1])

                    if params[2][0] == '{':
                        print 'Found neuron spec...'
                        #neurons.append(cls.populate_neuron_dict(params))
                        break
                    else:
                        print 'Error when parsing neuron.'
                        return
                    break

        # search for synapses
        synape_params = text.split('addSynapse(')
        for i in range(1, len(synape_params)):

          print 'Adding synapse...'

          # this creates a list of the nesting level and the contents within parentheses
          vals = list(cls.parenthetic_contents('(' + synape_params[i]))

          for val in vals:
              if val[0] == 0:
                  # This splits the string by commas, but excludes those in parentheses or brackets (normal/uniform parameters)
                  params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
                  params = params.findall(val[1])

                  for param in params:
                    print param

                  if params[2][0] == '{':
                      print 'Found synapse spec...'
                      neurons.append(cls.populate_synapse_dict(params))
                      break
                  else:
                      print 'Error when parsing synapse.'
                      return
                  break

        # search for neuron groups
        # We are assuming the script file will follow one of these forms:
        #   neuron_parameters = sim.addNeuron("ncs_neuron","ncs",ncs_cell)
        #   group_1 = sim.addNeuronGroup("group_1",1,"ncs_neuron",None) OR
        #   group_1 = sim.addNeuronGroup("group_1",1,neuron_parameters,None)
        # CHECK IF ITS IN QUOTES TO DETERMINE? SHOULD ALSO PROBABLY LOOK AT PYNCS TO SEE ALL POSSIBILITIES...
        neuron_group_params = text.split('addNeuronGroup')
        for i in range(1, len(neuron_group_params)):
            print 'Adding neuron group...'
            # WAIT UNTIL WE DECIDE HOW GROUPS WILL BE STRUCTURED TO HANDLE THIS
            # IF THE GROUPS ARE JUST THE NUMBER IN THE NEURONS, CHECK FOR THIS ABOVE
            # (CHECK IF PARAMETER MATCHES NAME) AND SET THE NUMBER IN THE NEURON JSON

        # search for synapse groups
        synapse_group_params = text.split('addSynapseGroup')
        for i in range(1, len(synapse_group_params)):
            pass

        # SIMULATOR
        # search for stimulus
        stim_params = text.split('addStimulus')
        for i in range(1, len(stim_params)):
          print 'Adding stimulus...'

          vals = list(cls.parenthetic_contents(stim_params[i]))

          for val in vals:
            if val[0] == 0:

              # This splits the string by commas, but excludes those in parentheses or brackets (normal/uniform parameters)
              params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
              params = params.findall(val[1])

              if params[1][0] == '{':
                print 'Found stim spec...'

                stimuli.append(cls.populate_stimulus_dict(params))

                # we only want the parameters closest to this call
                break
              else:
                print 'Error when parsing stimulus.'
                return
              break

        # search for reports
        report_params = text.split('addReport')
        for i in range(1, len(report_params)):
          print 'Adding report...'

          # addReport([group_1,group_2],"neuron","neuron_voltage",1.0,0.0,1.0)
          # addReport("group_1","neuron", "neuron_voltage", 1.0,0.0,1.0)
          report_type = report_params[i].split('toAsciiFile')
          vals = list(cls.parenthetic_contents(report_params[i]))

          for val in vals:
            if val[0] == 0:

              # This splits the string by commas, but excludes those in parentheses or brackets (normal/uniform parameters)
              params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
              params = params.findall(val[1])

              # check if writing report data to a file
              if len(report_type) > 1:
                report_type[1] = report_type[1][report_type[1].find("(")+1:report_type[1].find(")")]
                reports.append(cls.populate_report_dict(params, report_type[1]))
              else:
                reports.append(cls.populate_report_dict(params, ''))

              # we only want the parameters closest to this call
              break

        return sim_params

    @classmethod
    def populate_neuron_dict(cls, cell_spec):
        neuron = None

        # this is done to eliminate any math operators (*/+-) in parameter assignment
        cls.remove_operators(cell_spec)

        if 'izhikevich' in cell_spec[1]:
            neuron = {
                "$$hashKey": "08Z", 
                "classification": "cells", 
                "description": "Description", 
                "geometry": "Sphere", 
                "name": cell_spec[0].replace('"',''),
                "num": 1, 
                "parameters": {
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
                  "threshold": {
                    "maxValue": 0, 
                    "mean": 0, 
                    "minValue": 0, 
                    "stddev": 0, 
                    "type": "exact", 
                    "value": 0
                    }, 
                  "type": "Izhikevich", 
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
                    }
                }
            }
            # assign values
            # cell_spec[2 : end] looks like [{"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "u": -12.0, "v": -60.0, "threshold": 30, }]
            for i in range(2, len(cell_spec)):
                entry = cell_spec[i].split(':')
                if len(entry) > 1:
                  entry[1] = entry[1].replace('}', '')
                  if '"a"' in entry[0] or "'a'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['a'], entry[1])
                  elif '"b"' in entry[0] or "'b'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['b'], entry[1])
                  elif '"c"' in entry[0] or "'c'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['c'], entry[1])
                  elif '"d"' in entry[0] or "'d'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['d'], entry[1])
                  elif '"u"' in entry[0] or "'u'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['u'], entry[1])
                  elif '"v"' in entry[0] or "'v'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['v'], entry[1])
                  elif '"threshold"' in entry[0] or "'threshold'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['threshold'], entry[1])

        elif 'ncs' in cell_spec[1]:
            neuron = {
                "$$hashKey": "08R", 
                "classification": "cells", 
                "description": "Description", 
                "geometry": "Box", 
                "name": cell_spec[0].replace('"',''), 
                "num": 1, 
                "parameters": {
                    "calcium": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    }, 
                    "calciumSpikeIncrement": {
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
                    "channel": [], 
                    "leakConductance": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    },
                    "leakReversalPotential": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    }, 
                    "rMembrane": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    }, 
                    "restingPotential": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    }, 
                    "spikeShape": [], 
                    "tauCalcium": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0
                    }, 
                    "tauMembrane": {
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
                "type": "NCS"
                }
            }
            # assign values
            # cell_spec looks like { "threshold": -50.0, "resting_potential": -60.0, "calcium": 5.0, "calcium_spike_increment": 100.0, "tau_calcium": 0.03,
            #                        "tau_membrane": 0.020, "r_membrane": 200, "leak_reversal_potential": 0.0, "leak_conductance": 0.0, "spike_shape": [-33, 30, -42],
            #                        "channels":[ calcium_channel, voltage_channel_1, voltage_channel_2, ], 
            #                        "capacitance": 1.0, }

            # find channels
            param_str = ','.join(cell_spec)
            channel_params = param_str.split('"channels":')[1]
            channel_params = channel_params[channel_params.find("[")+1:channel_params.find("]")]
            param_list = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
            param_list = param_list.findall(channel_params)
            channels = []
            iterator = iter(param_list)
            for line in iterator:
              if '{' in line:
                  string = re.sub(r'\s', '', line)
                  string += ','
                  while '}' not in line:
                      try:
                          line = iterator.next()
                          string += re.sub(r'\s', '', line)
                          string += ','
                      except StopIteration:
                          break
                  string = string[:-1]
                  channels.append(string)

            for channel_str in channels:
              channel_spec_params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
              channel_spec_params = channel_spec_params.findall(channel_str)
              cls.populate_channel_dict(neuron['parameters']['channel'], channel_spec_params)

            # find spike shape values
            spike_shape_params = param_str.split('"spike_shape":')[1]
            spike_shape_params = spike_shape_params[spike_shape_params.find("[")+1:spike_shape_params.find("]")]
            spike_shape_vals = spike_shape_params.split(',')
            for i in spike_shape_vals:
              neuron['parameters']['spikeShape'].append(i)

            for i in range(2, len(cell_spec)):
                entry = cell_spec[i].split(':')
                if len(entry) > 1:
                  entry[1] = entry[1].replace('}', '')
                  if '"threshold"' in entry[0] or "'threshold'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['threshold'], entry[1])
                  elif '"resting_potential"' in entry[0] or "'resting_potential'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['restingPotential'], entry[1])
                  elif '"calcium"' in entry[0] or "'calcium'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['calcium'], entry[1])
                  elif '"calcium_spike_increment"' in entry[0] or "'calcium_spike_increment'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['calciumSpikeIncrement'], entry[1])
                  elif '"tau_calcium"' in entry[0] or "'tau_calcium'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['tauCalcium'], entry[1])
                  elif '"tau_membrane"' in entry[0] or "'tau_membrane'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['tauMembrane'], entry[1])
                  elif '"r_membrane"' in entry[0] or "'r_membrane'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['rMembrane'], entry[1])
                  elif '"leak_reversal_potential"' in entry[0] or "'leak_reversal_potential'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['leakReversalPotential'], entry[1])
                  elif '"leak_conductance"' in entry[0] or "'leak_conductance'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['leakConductance'], entry[1])
                  elif '"capacitance"' in entry[0] or "'capacitance'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['capacitance'], entry[1])

        elif 'hh' in cell_spec[1]:
            neuron = {
              "$$hashKey": "08S", 
              "classification": "cells", 
              "description": "Description", 
              "geometry": "Box", 
              "name": cell_spec[0].replace('"',''), 
              "num": 1, 
              "parameters": {
                "capacitance": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "channel": [], 
                "restingPotential": {
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
                "type": "HodgkinHuxley"
              }
            }

            # find channels
            param_str = ','.join(cell_spec)
            channel_params = param_str.split('"channels":')[1]
            char_list = list(channel_params)
            iterator = iter(char_list)
            string = ''
            for char in iterator:
              if '[' in char:
                count = 1
                while count != 0:
                    try:
                        char = iterator.next()
                        if '[' in char:
                          count += 1

                        if ']' in char:
                          count -= 1

                        string += char
                    except StopIteration:
                        break
            channel_params = string[:-1]

            char_list = list(channel_params)
            channels = []
            iterator = iter(char_list)
            for char in iterator:
              if '{' in char:
                  string = char
                  count = 1
                  while count != 0:
                      try:
                          char = iterator.next()

                          if '{' in char:
                            count += 1

                          if '}' in char:
                            count -= 1

                          string += char
                      except StopIteration:
                          break
                  channels.append(string)

            for channel_str in channels:

              # this stores the type, conductance, reversal potential, and particles into a list
              channel_spec_params = re.compile(r'(?:[^,(\[]|\([^)]*\)|\[[^)]*\])+')
              channel_spec_params = channel_spec_params.findall(channel_str)
              cls.populate_channel_dict(neuron['parameters']['channel'], channel_spec_params)

            # assign values
            # cell_spec looks like { "threshold": -50.0, "resting_potential": -60.0, "capacitance": 1.0,
            #                        "channels": [ potassium_channel, sodium_channel, leak_channel ] }
            for i in range(2, len(cell_spec)):
                entry = cell_spec[i].split(':')
                if len(entry) > 1:
                  entry[1] = entry[1].replace('}', '')
                  if '"threshold"' in entry[0] or "'threshold'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['threshold'], entry[1])
                  elif '"resting_potential"' in entry[0] or "'resting_potential'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['restingPotential'], entry[1])
                  elif '"capacitance"' in entry[0] or "'capacitance'" in entry[0]:
                      cls.assign_spec_param_dict(neuron['parameters']['capacitance'], entry[1])

        return neuron

    @classmethod
    def populate_channel_dict(cls, neuron_channels, channel_params):
      channel = None
      channel_type = [s for s in channel_params if 'type' in s][0]
      channel_type = channel_type.split(':')[1]

      # this is done to eliminate any math operators (*/+-) in parameter assignment
      cls.remove_operators(channel_params)

      if 'voltage_gated_ion' in channel_type:
        channel = {
                "$$hashKey": "09B", 
                "activationSlope": {
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
                "deactivationSlope": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "description": "Description", 
                "equilibriumSlope": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "mInitial": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "mPower": {
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
                "reversalPotential": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "vHalf": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }
              }

        # assign values
        # dictionary containing
        # {"type":"voltage_gated_ion","m_initial":0.0,"m_power":1,"reversal_potential":ncs.Normal(102.0,4.0),"v_half":2,
        #  "deactivation_slope":5,"activation_slope":12,"equilibrium_slope":2.5,"r":1/0.1,"conductance":5*0.0001}

        for param in channel_params:
                entry = param.split(':')
                if len(entry) == 2:
                  entry[1] = entry[1].replace('}', '')

                  if '"m_initial"' in entry[0] or "'m_initial'" in entry[0]:
                    cls.assign_spec_param_dict(channel['mInitial'], entry[1])
                  elif '"m_power"' in entry[0] or "'m_power'" in entry[0]:
                    cls.assign_spec_param_dict(channel['mPower'], entry[1])
                  elif '"reversal_potential"' in entry[0] or "'reversal_potential'" in entry[0]:
                    cls.assign_spec_param_dict(channel['reversalPotential'], entry[1])
                  elif '"v_half"' in entry[0] or "'v_half'" in entry[0]:
                    cls.assign_spec_param_dict(channel['vHalf'], entry[1])
                  elif '"deactivation_slope"' in entry[0] or "'deactivation_slope'" in entry[0]:
                    cls.assign_spec_param_dict(channel['deactivationSlope'], entry[1])
                  elif '"activation_slope"' in entry[0] or "'activation_slope'" in entry[0]:
                    cls.assign_spec_param_dict(channel['activationSlope'], entry[1])
                  elif '"equilibrium_slope"' in entry[0] or "'equilibrium_slope'" in entry[0]:
                    cls.assign_spec_param_dict(channel['equilibriumSlope'], entry[1])
                  elif '"r"' in entry[0] or "'r'" in entry[0]:
                    cls.assign_spec_param_dict(channel['r'], entry[1])
                  elif '"conductance"' in entry[0] or "'conductance'" in entry[0]:
                    cls.assign_spec_param_dict(channel['conductance'], entry[1])

      elif 'calcium_dependent' in channel_type:
        channel = {
                "$$hashKey": "0EJ", 
                "backwardsRate": {
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
                "forwardExponent": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "forwardScale": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "mInitial": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "mPower": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                },
                "name": "Calcium Dependant Channel", 
                "reversalPotential": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "tauScale": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }
              }

        # { "type": "calcium_dependent", "m_initial": 0.0, "reversal_potential": -82, "m_power": 2, "conductance": 6.0*0.0108125, "forward_scale": 0.000125,
        #   "forward_exponent": 2, "backwards_rate": 2.5, "tau_scale": 0.01,}

        for param in channel_params:
                entry = param.split(':')
                if len(entry) == 2:
                  entry[1] = entry[1].replace('}', '')

                  if '"m_initial"' in entry[0] or "'m_initial'" in entry[0]:
                    cls.assign_spec_param_dict(channel['mInitial'], entry[1])
                  elif '"m_power"' in entry[0] or "'m_power'" in entry[0]:
                    cls.assign_spec_param_dict(channel['mPower'], entry[1])
                  elif '"reversal_potential"' in entry[0] or "'reversal_potential'" in entry[0]:
                    cls.assign_spec_param_dict(channel['reversalPotential'], entry[1])
                  elif '"forward_scale"' in entry[0] or "'forward_scale'" in entry[0]:
                    cls.assign_spec_param_dict(channel['forwardScale'], entry[1])
                  elif '"forward_exponent"' in entry[0] or "'forward_exponent'" in entry[0]:
                    cls.assign_spec_param_dict(channel['forwardExponent'], entry[1])
                  elif '"backwards_rate"' in entry[0] or "'backwards_rate'" in entry[0]:
                    cls.assign_spec_param_dict(channel['backwardsRate'], entry[1])
                  elif '"tau_scale"' in entry[0] or "'tau_scale'" in entry[0]:
                    cls.assign_spec_param_dict(channel['tauScale'], entry[1])
                  elif '"conductance"' in entry[0] or "'conductance'" in entry[0]:
                    cls.assign_spec_param_dict(channel['conductance'], entry[1])

      elif 'voltage_gated' in channel_type:
        channel = {
                "$$hashKey": "0JS", 
                "className": "voltageGatedChannel", 
                "conductance": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }, 
                "description": "Description", 
                "name": "Voltage Gated Channel", 
                "particles": {
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
                  "power": {
                    "maxValue": 0, 
                    "mean": 0, 
                    "minValue": 0, 
                    "stddev": 0, 
                    "type": "exact", 
                    "value": 0
                  }, 
                  "xInitial": {
                    "maxValue": 0, 
                    "mean": 0, 
                    "minValue": 0, 
                    "stddev": 0, 
                    "type": "exact", 
                    "value": 0
                  }
                }, 
                "reversalPotential": {
                  "maxValue": 0, 
                  "mean": 0, 
                  "minValue": 0, 
                  "stddev": 0, 
                  "type": "exact", 
                  "value": 0
                }
              }

      # handle particles
      for param in channel_params:
        entry = param.split(':')
        if '"particles"' in entry[0] or "'particles'" in entry[0]:
          particles = ':'.join(entry[1:])
          particles = particles[particles.find("[")+1:particles.find("]")]

          # remove outer parentheses
          char_list = list(particles)
          iterator = iter(char_list)
          for char in iterator:
              if '{' in char:
                  particles = ''
                  count = 1
                  while count != 0:
                      try:
                          char = iterator.next()

                          if '{' in char:
                            count += 1

                          if '}' in char:
                            count -= 1

                          particles += char
                      except StopIteration:
                          break
                  particles = particles[:-1]

          # split elements by commas not within parentheses, brackets, or curley braces
          particle_params = re.compile(r'(?:[^,({\[]|\([^)]*\)|\[[^)]*\]|\{[^)]*\})+')
          particle_params = particle_params.findall(particles)

          for param in particle_params:

            # handle alpha and beta
            alpha_beta = []
            if 'alpha' in param or 'beta' in param:
              char_list = list(param)
              iterator = iter(char_list)
              string = ''
              for char in iterator:
                  string += char
                  if '{' in char:
                      count = 1
                      while count != 0:
                          try:
                              char = iterator.next()

                              if '{' in char:
                                count += 1

                              if '}' in char:
                                count -= 1

                              string += char
                          except StopIteration:
                              break
                      alpha_beta.append(string)
                      string = ''

              for i in alpha_beta:
                entry = i.split('alpha')
                if len(entry) > 1:
                  entry[1] = entry[1][entry[1].find("{")+1:entry[1].find("}")]
                  alpha_params = re.compile(r'(?:[^,(]|\([^)]*\))+')
                  alpha_params = alpha_params.findall(entry[1])
                  for element in alpha_params:
                    entry = element.split(':')
                    if '"a"' in entry[0] or "'a'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['a'], entry[1])
                    elif '"b"' in entry[0] or "'b'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['b'], entry[1])
                    elif '"c"' in entry[0] or "'c'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['c'], entry[1])
                    elif '"d"' in entry[0] or "'d'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['d'], entry[1])
                    elif '"f"' in entry[0] or "'f'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['f'], entry[1])
                    elif '"h"' in entry[0] or "'h'" in entry[0]:
                      cls.assign_spec_param_dict(channel['particles']['alpha']['h'], entry[1])
                else:
                  entry = i.split('beta')
                  if len(entry) > 1:
                    entry[1] = entry[1][entry[1].find("{")+1:entry[1].find("}")]
                    beta_params = re.compile(r'(?:[^,(]|\([^)]*\))+')
                    beta_params = beta_params.findall(entry[1])
                    for element in beta_params:
                      entry = element.split(':')
                      if '"a"' in entry[0] or "'a'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['a'], entry[1])
                      elif '"b"' in entry[0] or "'b'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['b'], entry[1])
                      elif '"c"' in entry[0] or "'c'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['c'], entry[1])
                      elif '"d"' in entry[0] or "'d'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['d'], entry[1])
                      elif '"f"' in entry[0] or "'f'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['f'], entry[1])
                      elif '"h"' in entry[0] or "'h'" in entry[0]:
                        cls.assign_spec_param_dict(channel['particles']['beta']['h'], entry[1])

            else:
              entry = param.split(':')

              if '"power"' in entry[0] or "'power'" in entry[0]:
                cls.assign_spec_param_dict(channel['particles']['power'], entry[1])
              elif '"x_initial"' in entry[0] or "'x_initial'" in entry[0]:
                # TODO: FIGURE OUT HOW TO HANDLE WHEN VALUE IS A VARIABLE SET BY AN EQUATION
                #cls.assign_spec_param_dict(channel['particles']['xInitial'], entry[1])
                pass

        else:
          if len(entry) == 2:
            entry[1] = entry[1].replace('}', '')

            if '"reversal_potential"' in entry[0] or "'reversal_potential'" in entry[0]:
              cls.assign_spec_param_dict(channel['reversalPotential'], entry[1])
            elif '"conductance"' in entry[0] or "'conductance'" in entry[0]:
              cls.assign_spec_param_dict(channel['conductance'], entry[1])

      # add channel specification to neuron object
      neuron_channels.append(channel)

    @classmethod
    def populate_synapse_dict(cls, synapse_spec):
      
      synapse = None

      if 'ncs' in synapse_spec[1]:

        synapse = {
          "$$hashKey": "09W", 
          "classification": "synapseGroup", 
          "description": "Description", 
          "parameters": {
            "aLtdMinimum": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "aLtpMinimum": {
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
            "lastPostfireTime": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "lastPrefireTime": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "maxConductance": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 30
            }, 
            "name": "ncsSynapse", 
            "psgWaveformDuration": {
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
            "reversalPotential": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "tauDepression": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "tauFacilitation": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "tauLtd": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "tauLtp": {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": 0
            }, 
            "tauPostSynapticConductance": {
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
          }, 
          "post": "Cell 1", 
          "postPath": [
            {
              "$$hashKey": "09S", 
              "index": 0, 
              "name": "Home"
            }
          ], 
          "pre": "Cell 3", 
          "prePath": [
            {
              "$$hashKey": "09O", 
              "index": 0, 
              "name": "Home"
            }
          ], 
          "prob": 0.5
        }

        # TODO: how to assign postpath and prepath
        for i in range(2, len(synapse_spec)):
          entry = synapse_spec[i].split(':')
          if len(entry) > 1:
            entry[1] = entry[1].replace('}', '')
            if '"A_ltd_minimum"' in entry[0] or "'A_ltd_minimum'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['aLtdMinimum'], entry[1])
            elif '"A_ltp_minimum"' in entry[0] or "'A_ltp_minimum'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['aLtpMinimum'], entry[1])
            elif '"delay"' in entry[0] or "'delay'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['delay'], entry[1])
            elif '"last_postfire_time"' in entry[0] or "'last_postfire_time'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['lastPostfireTime'], entry[1])
            elif '"last_prefire_time"' in entry[0] or "'last_prefire_time'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['lastPrefireTime'], entry[1])
            elif '"max_conductance"' in entry[0] or "'max_conductance'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['maxConductance'], entry[1])
            elif '"psg_waveform_duration"' in entry[0] or "'psg_waveform_duration'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['psgWaveformDuration'], entry[1])
            elif '"redistribution"' in entry[0] or "'redistribution'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['redistribution'], entry[1])
            elif '"reversal_potential"' in entry[0] or "'reversal_potential'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['reversalPotential'], entry[1])
            elif '"tau_depression"' in entry[0] or "'tau_depression'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['tauDepression'], entry[1])
            elif '"tau_facilitation"' in entry[0] or "'tau_facilitation'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['tauFacilitation'], entry[1])
            elif '"tau_ltd"' in entry[0] or "'tau_ltd'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['tauLtd'], entry[1])
            elif '"tau_ltp"' in entry[0] or "'tau_ltp'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['tauLtp'], entry[1])
            elif '"tau_postsynaptic_conductance"' in entry[0] or "'tau_postsynaptic_conductance'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['tauPostSynapticConductance'], entry[1])
            elif '"utilization"' in entry[0] or "'utilization'" in entry[0]:
              cls.assign_spec_param_dict(synapse['parameters']['utilization'], entry[1])


          # THIS IS DEPENDENT ON ADDING THE SYNAPSE GROUP
          # sim.addSynapseGroup("1_to_2", group_1, group_2, 1.0, flat_parameters)
          ''' 
          elif '"v"' in entry[0] or "'v'" in entry[0]:
              cls.assign_spec_param_dict(synapse['prob'], entry[1])
          elif '"c"' in entry[0] or "'c'" in entry[0]:
              cls.assign_spec_param_dict(synapse['post'], entry[1])
          elif '"d"' in entry[0] or "'d'" in entry[0]:
              cls.assign_spec_param_dict(synapse['pre'], entry[1])'''

      elif 'flat' in synapse_spec[1]:
      
        synapse = {
          "$$hashKey": "0A2", 
          "classification": "synapseGroup", 
          "description": "Description", 
          "parameters": {
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
          }, 
          "post": "Cell 3", 
          "postPath": [
            {
              "$$hashKey": "0A0", 
              "index": 0, 
              "name": "Home"
            }
          ], 
          "pre": "Cell 1", 
          "prePath": [
            {
              "$$hashKey": "09Y", 
              "index": 0, 
              "name": "Home"
            }
          ], 
          "prob": 0.5
        }

        # TODO: how to assign postpath and prepath
        for i in range(2, len(synapse_spec)):
          entry = synapse_spec[i].split(':')
          if len(entry) > 1:
            entry[1] = entry[1].replace('}', '')
            if '"current"' in entry[0] or "'current'" in entry[0]:
                cls.assign_spec_param_dict(synapse['parameters']['current'], entry[1])
            elif '"delay"' in entry[0] or "'delay'" in entry[0]:
                cls.assign_spec_param_dict(synapse['parameters']['delay'], entry[1])


          # THIS IS DEPENDENT ON ADDING THE SYNAPSE GROUP
          # sim.addSynapseGroup("1_to_2", group_1, group_2, 1.0, flat_parameters)
          ''' 
          elif '"v"' in entry[0] or "'v'" in entry[0]:
              cls.assign_spec_param_dict(synapse['prob'], entry[1])
          elif '"c"' in entry[0] or "'c'" in entry[0]:
              cls.assign_spec_param_dict(synapse['post'], entry[1])
          elif '"d"' in entry[0] or "'d'" in entry[0]:
              cls.assign_spec_param_dict(synapse['pre'], entry[1])'''

      return synapse


    @classmethod
    def populate_stimulus_dict(cls, stim_spec):

      # WE WILL MOST LIKELY NEED TO CHECK THE STIM TYPE HERE AND THE JSON WILL 
      # BE DIFFERENT DEPENDING ON THE TYPE

      stimulus = {
        "$$hashKey": "0PO", 
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
        "className": "simulationInput", 
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
        "dyn_range": {
          "maxValue": 0, 
          "mean": 0, 
          "minValue": 0, 
          "stddev": 0, 
          "type": "exact", 
          "value": 0
        },
        "endTime": "0", 
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
        "inputTargets": [], 
        "name": "Input1", 
        "phase": {
          "maxValue": 0, 
          "mean": 0, 
          "minValue": 0, 
          "stddev": 0, 
          "type": "exact", 
          "value": 0
        }, 
        "probability": "0", 
        "startTime": 0, 
        "start_amplitude": {
          "maxValue": 0, 
          "mean": 0, 
          "minValue": 0, 
          "stddev": 0, 
          "type": "exact", 
          "value": 0
        }, 
        "stimulusType": "Rectangular Current", 
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

      # assign stimulus type
      stimulus['stimulusType'] = cls.underscore_to_camelcase(stim_spec[0]).title().replace('"','').replace("'", '')

      # parameters
      for i in range (1, len(stim_spec) - 4):
        entry = stim_spec[i].split(':')
        entry[1] = entry[1].replace('}', '')

        if '"amplitude"' in entry[0] or "'amplitude'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['amplitude'], entry[1])
        elif '"amplitude_scale"' in entry[0] or "'amplitude_scale'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['amplitude_scale'], entry[1])
        elif '"amplitude_shift"' in entry[0] or "'amplitude_shift'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['amplitude_shift'], entry[1])
        elif '"current"' in entry[0] or "'current'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['current'], entry[1])
        elif '"delay"' in entry[0] or "'delay'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['delay'], entry[1])
        elif '"dyn_range"' in entry[0] or "'dyn_range'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['dyn_range'], entry[1])
        elif '"ending_amplitude"' in entry[0] or "'ending_amplitude'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['end_amplitude'], entry[1])
        elif '"frequency"' in entry[0] or "'frequency'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['frequency'], entry[1])
        elif '"phase"' in entry[0] or "'phase'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['phase'], entry[1])
        elif '"starting_amplitude"' in entry[0] or "'starting_amplitude'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['start_amplitude'], entry[1])
        elif '"time_scale"' in entry[0] or "'time_scale'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['time_scale'], entry[1])
        elif '"width"' in entry[0] or "'width'" in entry[0]:
          cls.assign_spec_param_dict(stimulus['width'], entry[1])

      # target group (this is an array)
      # IS THE ONLY WAY OF PASSING A GROUP WITH AN ALIAS? (only one parameter can be passed)
      # FIX THIS ONCE WE KNOW HOW THE GROUPS ARE GOING TO BE STRUCTURED
      # can be single group, alias, or array [group1, group1]
      stimulus['inputTargets'].append(stim_spec[len(stim_spec) - 4].replace('"','').replace("'", ''))

      # probability
      stimulus['probability'] = stim_spec[len(stim_spec) - 3]

      # start time
      stimulus['startTime'] = stim_spec[len(stim_spec) - 2]

      # end time
      stimulus['endTime'] = stim_spec[len(stim_spec) - 1]

      return stimulus

    @classmethod
    def underscore_to_camelcase(cls, value):
      def camelcase(): 
          yield str.lower
          while True:
              yield str.capitalize

      c = camelcase()
      return " ".join(c.next()(x) if x else '_' for x in value.split("_"))

    @classmethod
    def populate_report_dict(cls, report_spec, file_name):

      report =       {
        "$$hashKey": "0Q2", 
        "className": "simulationOutput", 
        "fileName": "",
        "endTime": 0, 
        "name": "Output1", 
        "numberFormat": "ascii", 
        "possibleOutputTargets": [], 
        "possibleReportType": 1, 
        "probability": "0", 
        "reportTargets": [], 
        "reportType": "Neuron Voltage", 
        "saveAsFile": False, 
        "startTime": 0
      }
        #       1. A set of neuron group or a set of synapse group to report on
        #       2. A target type: "neuron" or "synapses"
        #       3. type of report: synaptic_current, neuron_voltage, neuron_fire, 
        #          input current, etc.
        #       4. Probability (the percentage of elements to report on)

      # TODO: FIGURE OUT HOW TO HANDLE VARIOUS METHODS OF SPECIFYING TARGETS
      # AS WELL AS HOW TO POPULATE THE ALL POSSIBLE GROUPS IN THE JSON
      report['reportTargets'].append(report_spec[0].replace('"','').replace("'", ''))
      report['reportType'] = cls.underscore_to_camelcase(report_spec[2]).title().replace('"','').replace("'", '')
      report['probability'] = report_spec[3]
      report['startTime'] = report_spec[4]
      report['endTime'] = report_spec[5]

      if file_name:
        report['fileName'] = file_name.replace('"','').replace("'", '').replace('(', '').replace(')', '')
        report['saveAsFile'] = True

      return report

    @classmethod
    def process_model(cls, sim, entity_dicts, neuron_groups, synapse_groups):
        # create a list of errors to output if necessary
        errors = []
        # schemaloader
        '''schema_loader = SchemaLoader()
        # see if the schema works
        try:
            validate(entity_dicts, schema_loader.get_schema('transfer_schema'))
        # if it doesn't pass validation, return a bad request error
        except ValidationError:
            errors.append("Improper json format")'''

        # models on DB can be found by <author, model name> key? hash key is a part of cell group (single in total groups)
        # WHEN THE MODEL IS STORED IN THE DB, NEED TO STORE DESCRIPTIONS, ETC.

        # get the lists of entities
        neurons = entity_dicts['cellGroups']['cellGroups']
        synapses = entity_dicts['synapses']

        # add neurons to sim
        groups = []
        for group in neurons:
            # TODO: Validate neuron spec
            pass

            # get neuron parameters
            neuron_params = group['parameters']

            # Get neuron type from the model
            neuron_type = ModelService.convert_neuron_type(neuron_params['type'])

            # TODO: Move the 3 cells types into functions so it doesn't look so disgusting
            # dictionary for neuron parameters which vary depending on cell type
            if neuron_type == 'izhikevich':
                spec = {
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0,
                    "d": 0.0,
                    "u": 0.0,
                    "v": 0.0,
                    "threshold": 0,
                }
                ModelService.convert_params_to_specification_format(neuron_params, spec)
                ModelService.convert_unicode_ascii(spec)

            elif neuron_type == 'ncs':
                spec = {
                    "calcium": 0.0,
                    "calciumSpikeIncrement": 0.0,
                    "capacitance": 0.0,
                    "leakConductance": 0.0,
                    "leakReversalPotential": 0.0,
                    "rMembrane": 0.0,
                    "restingPotential": 0.0,
                    "tauCalcium": 0.0,
                    "tauMembrane": 0.0,
                    "threshold": 0,
                }
                ModelService.convert_params_to_specification_format(neuron_params, spec)
                ModelService.convert_unicode_ascii(spec)

                # convert dict keys to match what simulator takes
                ModelService.convert_keys_to_sim_input(spec)

                # loop through channels
                channels = []
                for channel in neuron_params['channel']:
                    # determine channel type so parameters can be populated
                    if channel['name'] == "Calcium Dependant Channel":
                        channel_type = 'calcium_dependent'

                        channel_spec = {
                            "backwardsRate": 0.0,
                            "conductance": 0.0,
                            "forwardExponent": 0.0,
                            "forwardScale": 0.0,
                            "mInitial": 0.0,
                            "mPower": 0.0,
                            "reversalPotential": 0.0,
                            "tauScale": 0.0
                        }

                    elif channel['name'] == "Voltage Gated Ion Channel":
                        channel_type = 'voltage_gated_ion'

                        channel_spec = {
                            "activationSlope": 0.0,
                            "conductance": 0.0,
                            "deactivationSlope": 0.0,
                            "equilibriumSlope": 0.0,
                            "mInitial": 0.0,
                            "mPower": 0.0,
                            "r": 0.0,
                            "reversalPotential": 0.0,
                            "vHalf": 0.0
                        }

                    # convert channels to spec format
                    ModelService.convert_params_to_specification_format(channel, channel_spec)
                    ModelService.convert_unicode_ascii(channel_spec)
                    ModelService.convert_keys_to_sim_input(channel_spec)

                    # this must be added after the conversion to spec format
                    channel_spec['type'] = channel_type
                    if channel['name'] == "Voltage Gated Channel":
                        channel_spec['particles'] = particle_spec

                    channels.append(channel_spec)

                    '''spike_shape_vals = []
                    for value in neuron_params['spikeShape']:
                        spike_shape_vals.append(value)'''
                    # DEFAULTED UNTIL NCB CAN SEND AN ARRAY
                    spike_shape_vals = [-33, 30, -42]

                    # add arrays to the spec map
                    spec['channels'] = channels
                    spec['spike_shape'] = spike_shape_vals
 
            else: #Hodgkin Huxley
                spec = {
                    "capacitance": 0.0,
                    "restingPotential": 0.0,
                    "threshold": 0
                }

                ModelService.convert_params_to_specification_format(neuron_params, spec)
                ModelService.convert_unicode_ascii(spec)
                ModelService.convert_keys_to_sim_input(spec)

                # loop through channels
                # This is the only cell that accepts the Voltage Gated channels
                channels = []
                for channel in neuron_params['channel']:
                    # TODO: else throw an error
                    if channel['name'] == "Voltage Gated Channel":
                        channel_type = 'voltage_gated'

                        particles = channel['particles']
                        particle_spec = {
                            "power": 0.0,
                            "xInitial": 0.0,
                        }
                        # convert particles to spec format
                        ModelService.convert_params_to_specification_format(particles, particle_spec)
                        ModelService.convert_unicode_ascii(particle_spec)
                        ModelService.convert_keys_to_sim_input(particle_spec)

                        alpha_spec = {
                            "a": 0.0,
                            "b": 0.0,
                            "c": 0.0,
                            "d": 0.0,
                            "f": 0.0,
                            "h": 0.0
                        }
                        beta_spec = {
                            "a": 0.0,
                            "b": 0.0,
                            "c": 0.0,
                            "d": 0.0,
                            "f": 0.0,
                            "h": 0.0
                        }
                        # convert alpha and beta to spec format
                        ModelService.convert_params_to_specification_format(particles['alpha'], alpha_spec)
                        ModelService.convert_unicode_ascii(alpha_spec)
                        ModelService.convert_keys_to_sim_input(alpha_spec)
                        ModelService.convert_params_to_specification_format(particles['beta'], beta_spec)
                        ModelService.convert_unicode_ascii(beta_spec)
                        ModelService.convert_keys_to_sim_input(beta_spec)

                        particle_spec['alpha'] = alpha_spec
                        particle_spec['beta'] = beta_spec

                        channel_spec = {
                            "conductance": 0.0,
                            "reversalPotential": 0.0
                        }

                    # convert channels to spec format
                    ModelService.convert_params_to_specification_format(channel, channel_spec)
                    ModelService.convert_unicode_ascii(channel_spec)
                    ModelService.convert_keys_to_sim_input(channel_spec)

                    # this must be added after the conversion to spec format
                    channel_spec['type'] = channel_type
                    if channel['name'] == "Voltage Gated Channel":
                        channel_spec['particles'] = particle_spec

                    # add arrays to the spec map
                    channels.append(channel_spec)
                    spec['channels'] = channels

            # store this neuron type
            print('ADDING NEURON: ')
            print group['name']
            print neuron_type
            print spec
            print '\n'
            groups.append(sim.addNeuron(group['name'].encode('ascii', 'ignore'), neuron_type.encode('ascii', 'ignore'), spec))

        # add synapses
        connections = []
        for synapse in synapses:
            # TODO: Validate synapse spec
            pass

            # get synapse parameters
            synapse_params = synapse['parameters']

            # TODO: add this to model class in NCB
            # Get synapse type from the model (NCS or flat)
            #synapse_type = synapse_params['type']
            synapse_type = 'ncs'

            # dictionary for synapse parameters
            spec = {          
                "utilization": 0.0,
                "redistribution": 0.0,
                "lastPrefireTime": 0.0,
                "lastPostfireTime": 0.0,
                "tauFacilitation": 0.0,
                "tauDepression": 0.0,
                "tauLtp": 0.0,
                "tauLtd": 0.0,
                "aLtpMinimum": 0.0,
                "aLtdMinimum": 0.0,
                "maxConductance": 0.0,
                "reversalPotential":0.0,
                "tauPostSynapticConductance": 0.0,
                "psgWaveformDuration": 0.0,
                "delay": 0,
            }

            ModelService.convert_params_to_specification_format(synapse_params, spec)
            ModelService.convert_unicode_ascii(spec)

            # store this synapse
            connections.append(sim.addSynapse(synapse_params['name'].encode('ascii', 'ignore'), synapse_type.encode('ascii', 'ignore'), spec))

        # create neuron groups
        for i in range(len(groups)):
            print "ADDING TO GROUP..."
            print neurons[i]['name']
            print neurons[i]['num']

            # final parameter is optional geometry generator? neurons[i]['geometry']
            group_name = 'group_' + str(i)
            neuron_groups.append(sim.addNeuronGroup(group_name,int(neurons[i]['num']),groups[i],None))

        # do the same for the synapse groups
        for i in range(len(connections)):
            synapse_groups.append(sim.addSynapseGroup(synapses[i]['parameters']['name'], synapses[i]['pre'], synapses[i]['post'], synapses[i]['prob'], connections[i]))


    # This function takes in the neuron and synapse groups created in the process_model function
    @classmethod
    def add_stims_and_reports(cls, sim, entity_dicts, model_entity_dicts, neuron_groups, synapse_groups):
        errors = []
        neurons = model_entity_dicts['cellGroups']['cellGroups']
        synapses = model_entity_dicts['synapses']
        stimuli = entity_dicts['inputs']
        for stimulus in stimuli:
            # TODO: Validate stimulus spec
            pass

            # Get stimulus type from the model
            # STIM TYPE NEEDS TO BE IN THE FORMAT "rectangular_current" not "Rectangular Current"?
            # rectangular_current, rectangular_voltage, linear_current, linear_voltage, sine_current, or sine_voltage
            stimulus_type = stimulus['stimulusType'].encode('ascii', 'ignore')
            stimulus_type = "rectangular_current"
            prob = float(stimulus['probability'])
            time_start = float(stimulus['startTime'])
            time_end = float(stimulus['endTime'])
            targets = stimulus['inputTargets']

            # these parameters will mostly change depending on the stimulus type
            spec = {
                "amplitude": 10.0, # THIS IS 10 BECAUSE IT IS NOT SET BY NCB
                "amplitude_scale": 0.0,
                "amplitude_shift": 0.0,
                "current": 0.0,
                "delay": 0.0,
                "end_amplitude": 0.0,
                "frequency": 0.0,
                "phase": 0.0,
                "start_amplitude": 0.0,
                "time_scale": 0.0,
                "width": 0.0
            }

            ModelService.convert_params_to_specification_format(stimulus, spec)
            ModelService.convert_unicode_ascii(spec)

            # Determine target group
            # TODO: Handle aliases/groups of groups-From samples it looks like you can specify one neuron group or use an alias to do multiple groups
            # This determines the target by its name only. If this is not unique can possibly use hash key
            for target in targets:
                for i in range(len(neurons)):
                    if neurons[i]['name'] == target:
                        sim.addStimulus(stimulus_type, spec, neuron_groups[i], prob, time_start, time_end)
            # DEFAULT THIS UNTIL IT CAN BE SET BY NCB
            sim.addStimulus(stimulus_type, spec, neuron_groups[0], prob, time_start, time_end)

        # reports
        reports = entity_dicts['outputs']
        for report in reports:
            # TODO: Validate report spec
            pass

            # get report type
            report_type = ModelService.convert_report_type(report['reportType'].encode('ascii', 'ignore'))

            # determine target group
            # TODO: This can be multiple groups. How to handle this?
            '''report_target = None
            target_type = None
            for i in range(len(neurons)):
                if neurons[i]['name'] == report['reportTarget']:
                    report_target = neuron_groups[i]
                    target_type = 'neuron'

            for i in range(len(synapses)):
                if synapses[i]['name'] == report['reportTarget']:
                    report_target = synapse_groups[i]
                    target_type = 'synapses'''
            # DEFAULTED BECAUSE THIS IS STILL SAYING NO GROUPS AVAILABLE
            report_target = neuron_groups[0]
            target_type = 'neuron'

            probability = float(report['probability'])
            time_start = float(report['startTime'])
            time_end = float(report['endTime'])
            # DEFAULT THESE UNTIL THEY CAN BE SET BY NCB
            probability = 1.0
            time_start = 0.0
            time_end = 1.0

            print 'ADDING REPORT...'
            rpt = sim.addReport(report_target, target_type, report_type, probability, time_start, time_end)
            print 'ADDING OUTPUT FILE...'
            # TODO: determine if other formats other than ASCII are available
            if report['saveAsFile'] == True:
                rpt.toAsciiFile("./" + report['fileName'].encode('ascii', 'ignore'))

        # For running the simulation
        # fsv, includeDistance, interactive, and seed are not used...  
        print "ATTEMPTING TO RUN"
        # duration (in seconds) - each time step is 1 ms         
        sim.run(duration=float(entity_dicts['duration'])) 
        #sim.run(duration=1.0)         

        return errors

    ''' This method is not used with the current JSON format received from NCB'''
    @classmethod
    def traverse_groups(cls, group, entity_dicts, location_string):
        spec = group['specification']
        # if the neuron_groups key doesn't exist in entity_dicts
        if 'neuron_groups' not in entity_dicts:
            # make it an empty list
            entity_dicts['neuron_groups'] = []
        # find the neuron that corresponds to the id in the neuron_group
        for neuron_group in spec['neuron_groups']:
            new_neuron_group = {}
            # set the parameters
            new_neuron_group['label'] = neuron_group['label']
            new_neuron_group['count'] = neuron_group['count']
            new_neuron_group['geometry'] = neuron_group['geometry']
            # TODO This will eventually need to be absolute geometry
            new_neuron_group['location'] = neuron_group['location']
            # this is to identify stuff for connections and reports
            new_neuron_group['location_string'] = location_string + ':' + \
                neuron_group['label']
            neuron_id = neuron_group['neuron']
            for neuron in entity_dicts['neurons']:
                if neuron['_id'] == neuron_id:
                    new_neuron_group['neuron'] = neuron
            entity_dicts['neuron_groups'].append(new_neuron_group)
        # call this method for each subgroup
        for subgroup in spec['subgroups']:
            for group in entity_dicts['groups']:
                if subgroup['group'] == group['_id']:
                    ModelHelper.traverse_groups(group,
                                                entity_dicts,
                                                location_string + ':' +
                                                group['entity_name'])

    @classmethod
    def convert_neuron_type(cls, neuron_type):
        if neuron_type == 'Izhikevich':
            return 'izhikevich'
        elif neuron_type == 'NCS':
            return 'ncs'
        elif neuron_type == 'HodgkinHuxley':
            return 'hh'
        else:
            return neuron_type

    @classmethod
    def convert_unicode_ascii(cls, spec):
        for param, value in spec.iteritems():
            v = value
            del spec[param]
            spec[param.encode('ascii','ignore')] = value

    ''' This method is not used with the current JSON format received from NCB'''
    @classmethod
    def process_normal_uniform_parameters(cls, spec):
        for param, value in spec.iteritems():
            if type(value) is dict:
                if value['type'] == 'normal':
                    spec[param] = ncs.Normal(value['mean'], value['stdev'])
                if value['type'] == 'uniform':
                    spec[param] = ncs.Uniform(value['min'], value['max'])

    @classmethod
    def convert_params_to_specification_format(cls, params, spec):
        # exact example : "a": 0.02
        # uniform example: "a": ncs.Uniform(min, max)
        # normal example: "a": ncs.Normal(mean, standard deviation)

        for param, value in params.iteritems():
            if type(value) is dict and 'type' in params:
                if value['type'] == 'exact':
                    spec[param] = float(value['value'])
                elif value['type'] == 'normal':
                    spec[param] = ncs.Normal(float(value['mean']), float(value['stddev']))
                elif value['type'] == 'uniform':
                    spec[param] = ncs.Uniform(float(value['minValue']), float(value['maxValue']))

    @classmethod
    def assign_spec_param_dict(cls, param, param_entry):
        # normal example: ncs.Normal(102.0,4.0)
        # uniform example: ncs.Uniform(0,4.0)
        # exact example: 4.0

        if 'ncs.Normal' in param_entry:
            vals = param_entry[param_entry.find("(")+1:param_entry.find(")")]
            vals = vals.split(',')
            param['mean'] = float(vals[0])
            param['stddev'] = float(vals[1])
            param['type'] = 'normal'

        elif 'ncs.Uniform' in param_entry:
            vals = param_entry[param_entry.find("(")+1:param_entry.find(")")]
            vals = vals.split(',')
            param['minValue'] = float(vals[0])
            param['maxValue'] = float(vals[1])
            param['type'] = 'uniform'

        else:
            param['value'] = float(param_entry)
            param['type'] = 'exact'

    @classmethod
    def convert_keys_to_sim_input(cls, spec):
        for key, value in spec.iteritems():
            spec[cls.convert_camel_case_to_underscore(key)] = spec.pop(key)

    @classmethod
    def convert_camel_case_to_underscore(cls, string):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @classmethod
    def remove_operators(cls, param_spec):
      for i in range(len(param_spec)):
        if ':' in param_spec[i]:
          param = param_spec[i].split(':')
          if '*' in param[1]:
            vals = param[1].split('*')
            if len(vals) == 2:
              if '}' in vals[1]:
                vals[1] = vals[1].replace('}','')
                product = float(vals[0]) * float(vals[1])
                param_spec[i] = param[0] + ':' + str(product) + '}'
              else:
                product = float(vals[0]) * float(vals[1])
                param_spec[i] = param[0] + ':' + str(product)
          elif '/' in param[1]:
            vals = param[1].split('/')
            if len(vals) == 2:
              if '}' in vals[1]:
                vals[1] = vals[1].replace('}','')
                quotient = float(vals[0]) / float(vals[1])
                param_spec[i] = param[0] + ':' + str(quotient) + '}'
              else:
                quotient = float(vals[0]) * float(vals[1])
                param_spec[i] = param[0] + ':' + str(quotient)
          elif '+' in param[1]:
            vals = param[1].split('+')
            if len(vals) == 2:
              if '}' in vals[1]:
                vals[1] = vals[1].replace('}','')
                sum = float(vals[0]) + float(vals[1])
                param_spec[i] = param[0] + ':' + str(sum) + '}'
              else:
                sum = float(vals[0]) * float(vals[1])
                param_spec[i] = param[0] + ':' + str(sum)

          elif '-' in param[1]:
            vals = param[1].split('-')
            if len(vals) == 2:
              if vals[0]:
                if '}' in vals[1]:
                  vals[1] = vals[1].replace('}','')
                  difference = float(vals[0]) - float(vals[1])
                  param_spec[i] = param[0] + ':' + str(difference) + '}'
                else:
                  difference = float(vals[0]) - float(vals[1])
                  param_spec[i] = param[0] + ':' + str(difference)

    @classmethod
    def convert_report_type(cls, report_type):
        if report_type == 'Synaptic Current':
            return 'synaptic_current'
        elif report_type == 'Neuron Voltage':
            return 'neuron_voltage'
        elif report_type == 'Neuron Fire':
            return 'neuron_fire'
        elif report_type == 'Input Current':
            return 'input_current'
        else:
            return report_type             

    def create_ncs_normal(cls, params):
        return ncs.Normal(params['mean'], params['std_dev'])

    def create_ncs_uniform(cls, params):
        return ncs.Uniform(params['lower_bound'], params['upper_bound'])

    def create_ncs_exact(cls, params):
        return params['value']
