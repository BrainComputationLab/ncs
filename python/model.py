from __future__ import absolute_import, unicode_literals

class ModelService(object):

    @classmethod
    def process_model(cls, sim, entity_dicts, neuron_group_dict):
        # create a list of errors to output if neccessary
        errors = []
        # schemaloader
        '''schema_loader = SchemaLoader()
        # see if the schema works
        try:
            validate(entity_dicts, schema_loader.get_schema('transfer_schema'))
        # if it doesn't pass validation, return a bad request error
        except ValidationError:
            errors.append("Improper json format")'''
        # get the lists of entities
        top_group_id = entity_dicts['top_group']
        neurons = entity_dicts['neurons']
        synapses = entity_dicts['synapses']
        groups = entity_dicts['groups']
        # add neurons to sim
        for neuron in neurons:
            # TODO: Validate neuron spec
            pass
            # Get neuron type from the model
            neuron_type = neuron['specification']['neuron_type']
            neuron_type = ModelService.convert_neuron_type(neuron_type)
            spec = neuron['specification']
            ModelService.process_normal_uniform_parameters(spec)
            ModelService.convert_unicode_ascii(spec)
            del spec['neuron_type']

            # temporarily pass this into addNeuronGroup until it is handled
            # neuron_params = sim.addNeuron(neuron['_id'], neuron_type, spec)
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

        for synapse in synapses:
            # TODO: Validate synapse spec
            pass
            # Get synapse type from the model
            synapse_type = synapse['specification']['synapse_type']
            spec = synapse['specification']
            #sim.addSynapse(synapse['_id'], synapse_type, spec)
        for group in groups:
            if group['_id'] == top_group_id:
                top_group = group
        #ModelService.traverse_groups(top_group,
        #                            entity_dicts,
        #                            top_group['entity_name'])
        neuron_groups = entity_dicts['neuron_groups']
        #print neuron_groups
        for neuron_group in neuron_groups:
            neuron_type = neuron_group['neuron']['_id']
            '''    loc_string = neuron_group['location_string'].encode('ascii', 'ignore')
            alias = sim.addNeuronGroup(loc_string,
                                       neuron_group['count'],
                                       neuron_type.encode('ascii', 'ignore'),
                                       None
                                       #neuron_group['geometry']
                                       )
            neuron_group_dict[loc_string] = alias'''

            loc_string = neuron_group['location_string'].encode('ascii', 'ignore')
            '''alias = sim.addNeuronGroup(loc_string,
                                       neuron_group['count'],
                                       neuron_params,
                                       #neuron_type.encode('ascii', 'ignore'),
                                       None
                                       #neuron_group['geometry']
                                       )'''
        group_1=sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
            #neuron_group_dict[loc_string] = alias
        #return errors
        #return alias
        return group_1

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
    def add_stims_and_reports(cls, sim, entity_dicts, neuron_group_dict, group):
        errors = []
        stimuli = entity_dicts['stimuli']
        reports = entity_dicts['reports']
        for stimulus in stimuli:
            # TODO: Validate stimulus spec
            pass
            # Get stimulus type from the model
            stimulus_type = stimulus['specification']['stimulus_type'].encode('ascii', 'ignore')
            spec = stimulus['specification']
            prob = stimulus['specification']['probability']
            time_start = stimulus['specification']['time_start']
            time_end = stimulus['specification']['time_end']

            '''destinations = stimulus['specification']['destinations']
            neuron_group_list = []
            for loc_string in destinations:
                neuron_group_list.append(neuron_group_dict[loc_string])'''
            # TODO what to do about groups...

            parameters = {}
            for k, v in spec.iteritems():
                parameters[k.encode('ascii', 'ignore')] = v
            del parameters['stimulus_type']
            #del parameters['destinations']
            #sim.addStimulus(stimulus_type, parameters, neuron_group_list, prob, time_start, time_end)

            # temporarily pass group in from processModel
        input_parameters = {
            "amplitude":10
        }
            #sim.addStimulus(stimulus_type, input_parameters, group, prob, time_start, time_end)
        sim.addStimulus("rectangular_current", input_parameters, group, 1, 0.01, 1.0)
        for report in reports:
            # TODO: Validate report spec
            pass
            # Get report type from the model
            report_type = report['specification']['report_type']
            '''report_targets = stimulus['specification']['report_targets']
            probability = stimulus['specification']['probability']
            time_start = stimulus['specification']['time_start']
            time_end = stimulus['specification']['time_end']'''
            report_targets = report['specification']['report_targets']
            probability = report['specification']['probability']
            time_start = report['specification']['time_start']
            time_end = report['specification']['time_end']            
            # TODO what to do about targets...
            # need to loop through targets, for now just grab one
            #rpt = sim.addReport(report_targets[0], "neuron", report_type, probability, time_start, time_end)
        rpt=sim.addReport("group_1", "neuron", "neuron_voltage", 1, 0.0, 1.0)
        rpt.toAsciiFile("./regular_spiking_izh_json.txt") 
            #rpt.toAsciiFile("./" + report['specification']['method']['filename'])
        return errors


    @classmethod
    def convert_neuron_type(cls, neuron_type):
        if neuron_type == 'izh_neuron':
            return 'izhikevich'
        else:
            return neuron_type

    @classmethod
    def convert_unicode_ascii(cls, spec):
        for param, value in spec.iteritems():
            v = value
            del spec[param]
            spec[param.encode('ascii','ignore')] = value

    @classmethod
    def process_normal_uniform_parameters(cls, spec):
        for param, value in spec.iteritems():
            if type(value) is dict:
                if value['type'] == 'normal':
                    spec[param] = ncs.Normal(value['mean'], value['stdev'])
                if value['type'] == 'uniform':
                    spec[param] = ncs.Uniform(value['min'], value['max'])


    def create_ncs_normal(cls, params):
        return ncs.Normal(params['mean'], params['std_dev'])

    def create_ncs_uniform(cls, params):
        return ncs.Uniform(params['lower_bound'], params['upper_bound'])

    def create_ncs_exact(cls, params):
        return params['value']
