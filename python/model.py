import os, sys, json
from string import whitespace

#specify the location of ncs.py in ncs_lib_path
ncs_lib_path = ('../')
sys.path.append(ncs_lib_path)
import ncs

class ModelService(object):

    # TODO: ADD FUNCTIONS FOR BUILD SIM AND EXPORT TO FILE INSTEAD OF HAVING IT BUILD THE SIM AS 
    # SOON AS IT HAS THE PARAMETERS

    @classmethod
    def process_model(self, sim, entity_dicts, neuron_groups, synapse_groups, script):

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
        for index, group in enumerate(neurons):
            # TODO: Validate neuron spec
            pass

            # get neuron parameters
            neuron_params = group['parameters']

            # Get neuron type from the model
            neuron_type = self.convert_neuron_type(neuron_params['type'])

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
                self.convert_params_to_specification_format(neuron_params, spec)
                self.convert_unicode_ascii(spec)

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
                self.convert_params_to_specification_format(neuron_params, spec)
                self.convert_unicode_ascii(spec)

                # convert dict keys to match what simulator takes
                self.convert_keys_to_sim_input(spec)

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
                    self.convert_params_to_specification_format(channel, channel_spec)
                    self.convert_unicode_ascii(channel_spec)
                    self.convert_keys_to_sim_input(channel_spec)

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

                self.convert_params_to_specification_format(neuron_params, spec)
                self.convert_unicode_ascii(spec)
                self.convert_keys_to_sim_input(spec)

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
                        self.convert_params_to_specification_format(particles, particle_spec)
                        self.convert_unicode_ascii(particle_spec)
                        self.convert_keys_to_sim_input(particle_spec)

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
                        self.convert_params_to_specification_format(particles['alpha'], alpha_spec)
                        self.convert_unicode_ascii(alpha_spec)
                        self.convert_keys_to_sim_input(alpha_spec)
                        self.convert_params_to_specification_format(particles['beta'], beta_spec)
                        self.convert_unicode_ascii(beta_spec)
                        self.convert_keys_to_sim_input(beta_spec)

                        particle_spec['alpha'] = alpha_spec
                        particle_spec['beta'] = beta_spec

                        channel_spec = {
                            "conductance": 0.0,
                            "reversalPotential": 0.0
                        }

                    # convert channels to spec format
                    self.convert_params_to_specification_format(channel, channel_spec)
                    self.convert_unicode_ascii(channel_spec)
                    self.convert_keys_to_sim_input(channel_spec)
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
            #index
            #spec_json = json.dumps(spec)
            script += '\tparameters_' + str(index) + ' = sim.addNeuron("' + str(group['name'].encode('ascii', 'ignore')) + '", "' + str(neuron_type.encode('ascii', 'ignore')) + '", ' + str(spec) + ')\n'
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

            self.convert_params_to_specification_format(synapse_params, spec)
            self.convert_unicode_ascii(spec)

            # store this synapse
            script += '\tconnections_' + str(index) + ' = sim.addSynapse("' + str(synapse_params['name'].encode('ascii', 'ignore')) + '", "' + str(synapse_type.encode('ascii', 'ignore')) + '", ' + str(spec) + ')\n'
            connections.append(sim.addSynapse(synapse_params['name'].encode('ascii', 'ignore'), synapse_type.encode('ascii', 'ignore'), spec))

        # create neuron groups
        for i in range(len(groups)):
            print "ADDING TO GROUP..."
            print neurons[i]['name']
            print neurons[i]['num']

            # final parameter is optional geometry generator? neurons[i]['geometry']
            group_name = 'group_' + str(i)
            script += '\t' + group_name + ' = sim.addNeuronGroup("' + group_name + '", ' + str(int(neurons[i]['num'])) + ', parameters_' + str(i) + ', None)\n'
            neuron_groups.append(sim.addNeuronGroup(group_name,int(neurons[i]['num']),groups[i],None))

        # do the same for the synapse groups
        for i in range(len(connections)):
            script += '\t' + synapses[i]['parameters']['name'] + ' = sim.addSynapseGroup("' + synapses[i]['parameters']['name'] + '", ' + str(synapses[i]['pre']) + ', ' + str(synapses[i]['post']) + str(synapses[i]['prob']) + 'connections_' + str(i) + ')\n'
            synapse_groups.append(sim.addSynapseGroup(synapses[i]['parameters']['name'], synapses[i]['pre'], synapses[i]['post'], synapses[i]['prob'], connections[i]))

        return script


    # This function takes in the neuron and synapse groups created in the process_model function
    @classmethod
    def add_stims_and_reports(self, sim, entity_dicts, model_entity_dicts, neuron_groups, synapse_groups, username, script):
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

            self.convert_params_to_specification_format(stimulus, spec)
            self.convert_unicode_ascii(spec)

            # Determine target group
            # TODO: Handle aliases/groups of groups-From samples it looks like you can specify one neuron group or use an alias to do multiple groups
            # This determines the target by its name only. If this is not unique can possibly use hash key
            for target in targets:
                for i in range(len(neurons)):
                    if neurons[i]['name'] == target:
                        script += '\tsim.addStimulus("' + str(stimulus_type) + '", ' + str(spec) + ', ' + str(neuron_groups[i]) + ', ' + str(prob) + ', ' + str(time_start) + ', ' + str(time_end) + ')\n'
                        sim.addStimulus(stimulus_type, spec, neuron_groups[i], prob, time_start, time_end)
            # DEFAULT THIS UNTIL IT CAN BE SET BY NCB
            script += '\tsim.addStimulus("' + str(stimulus_type) + '", ' + str(spec) + ', ' + str(neuron_groups[0]) + ', ' + str(prob) + ', ' + str(time_start) + ', ' + str(time_end) + ')\n'
            sim.addStimulus(stimulus_type, spec, neuron_groups[0], prob, time_start, time_end)

        # reports
        reports = entity_dicts['outputs']
        for index, report in enumerate(reports):
            # TODO: Validate report spec
            pass

            # get report type
            report_type = self.convert_report_type(report['reportType'].encode('ascii', 'ignore'))

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
            script += '\treport_' + str(index) + '= sim.addReport("' + str(report_target) + '", "' + str(target_type) + '", "' + str(report_type) + '", ' + str(probability) + ', ' + str(time_start) + ', ' + str(time_end) + ')\n'
            rpt = sim.addReport(report_target, target_type, report_type, probability, time_start, time_end)
            print 'ADDING OUTPUT FILE...'
            # TODO: determine if other formats other than ASCII are available
            sim_identifier = username + '..' + report['name']
            if report['saveAsFile'] == True:
                script += '\treport_' + str(index) + '.toAsciiFileReportName("./' + str(report['fileName'].encode('ascii', 'ignore')) + '", "' + str(sim_identifier) + '")\n'
                rpt.toAsciiFile("./" + report['fileName'].encode('ascii', 'ignore'), sim_identifier)

        print "ATTEMPTING TO RUN"

        # duration (in seconds) - each time step is 1 ms       
        script += '\tsim.run(duration=' + str(float(entity_dicts['duration'])) + ')' + '\n'
        sim.run(duration=float(entity_dicts['duration']))         

        return script

    ''' ******************************Conversion Functions*********************************** '''

    @classmethod
    def convert_neuron_type(self, neuron_type):
        if neuron_type == 'Izhikevich':
            return 'izhikevich'
        elif neuron_type == 'NCS':
            return 'ncs'
        elif neuron_type == 'HodgkinHuxley':
            return 'hh'
        else:
            return neuron_type

    @classmethod
    def convert_unicode_ascii(self, spec):
        for param, value in spec.iteritems():
            v = value
            del spec[param]
            spec[param.encode('ascii','ignore')] = value

    @classmethod
    def convert_params_to_specification_format(self, params, spec):
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
    def convert_keys_to_sim_input(self, spec):
        for key, value in spec.iteritems():
            spec[self.convert_camel_case_to_underscore(key)] = spec.pop(key)

    @classmethod
    def convert_camel_case_to_underscore(self, string):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @classmethod
    def convert_report_type(self, report_type):
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

    ''' ******************This method should not be used***************************** '''
    @classmethod
    def sim_test(self):
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
