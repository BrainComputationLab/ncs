import pyncs

class Uniform:
  def __init__(self, lower_bound, upper_bound):
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def build(self):
    if isinstance(self.lower_bound, float):
      return pyncs.UniformDouble(self.lower_bound, self.upper_bound)
    elif isinstance(self.lower_bound, int):
      return pyncs.UniformInteger(self.lower_bound, self.upper_bound)
    else:
      return None

class Normal:
  def __init__(self, mean, std_dev):
    self.mean = mean
    self.std_dev = std_dev

  def build(self):
    if isinstance(self.mean, float):
      return pyncs.NormalDouble(self.mean, self.std_dev)
    elif isinstance(self.mean, int):
      return pyncs.NormalInteger(self.mean, self.std_dev)
    else:
      return None

class CellGroup:
  def __init__(self, name, count, parameters, geometry):
    self.name = name
    self.count = count
    self.parameters = parameters
    self.geometry = geometry
    self.neuron_group = None

class CellAlias:
  def __init__(self, name, subgroups):
    if not isinstance(subgroups, list):
      self.subgroups = [subgroups]
    else:
      self.subgroups = subgroups
    self.name = name
    self.resolved = False
    self.resolving = False
    self.neuron_alias = None

class ConnectionAlias:
  def __init__(self, name, subgroups):
    if not isinstance(subgroups, list):
      self.subgroups = [subgroups]
    else:
      self.subgroups = subgroups
    self.name = name
    self.resolving = False
    self.resolved = False
    self.synapse_alias = None

class Connection:
  def __init__(self, name, presynaptic, postsynaptic, probability, parameters):
    self.name = name
    self.presynaptic = CellAlias(None, presynaptic)
    self.postsynaptic = CellAlias(None, postsynaptic)
    self.probability = probability
    self.parameters = parameters
    self.synapse_group = None

class Report:
  def __init__(self, data_source):
    self.data_source = data_source
    self.sink = None

  def toAsciiFile(self, path):
    if self.sink:
      print "Data source is already in use."
      return self
    self.sink = pyncs.AsciiFileSink(self.data_source, path)
    return self

  def toAsciiFileReportName(self, path, report_name):
    if self.sink:
      print "Data source is already in use."
      return self
    self.sink = pyncs.AsciiFileSink(self.data_source, path, report_name)
    return self

  def toStdOut(self):
    if self.sink:
      print "Data source is already in use."
      return self
    self.sink = pyncs.AsciiStreamSink(self.data_source)
    return self    

  '''********** Functions added for Python to JSON conversion **********'''
  def parseToAsciiFile(self, path):
    pass

class EmptyReport:
  def toAsciiFile(self, path):
    return self
  def toStdOut(self):
    return self

class Simulation:
  def __init__(self):
    self.neuron_parameters = {}
    self.synapse_parameters = {}
    self.cell_groups = {}
    self.cell_aliases = {}
    self.connections = {}
    self.connection_aliases = {}
    self.simulaton_parameters = pyncs.SimulationParameters()
    self.simulaton_parameters.thisown = False

    # dictionaries added for Python to JSON conversion
    self.parse_model_spec = {}
    self.parse_stim_spec = []
    self.parse_report_spec = []
    self.parse_run_spec = {}

  def addNeuron(self, label, type_name, parameters):
    if self.getNeuronParameters(label):
      print "NeuronParameters %s already exists." % label
      return None
    model_parameters = self.buildModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build NeuronParameters %s" % label
      return None
    model_parameters.thisown = False
    self.neuron_parameters[label] = model_parameters
    return model_parameters

  def addSynapse(self, label, type_name, parameters):
    if self.getSynapseParameters(label):
      print "SynapseParameters %s already exists." % label
      return None
    model_parameters = self.buildModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build SynapseParameters %s" % label
      return None
    model_parameters.thisown = False
    self.synapse_parameters[label] = model_parameters
    return model_parameters

  def addNeuronGroup(self, label, count, parameters, geometry = None):
    if self.getCellGroup(label):
      print "NeuronGroup %s already exists." % label
      return None
    cell_group = CellGroup(label, count, parameters, geometry)
    self.cell_groups[label] = cell_group
    return self.addNeuronAlias(label, cell_group)

  def addNeuronAlias(self, label, subgroups):
    if label in self.cell_aliases:
      print "Neuron alias %s already exists" % label
      return None
    alias = CellAlias(label, subgroups)
    self.cell_aliases[label] = alias
    return alias
  
  def addSynapseGroup(self, label, presynaptic, postsynaptic, probability, parameters):
    if self.getConnection(label):
      print "Connection %s already exists." % label
      return None
    connection = Connection(label,
                            presynaptic,
                            postsynaptic,
                            probability,
                            parameters)
    self.connections[label] = connection
    return self.addSynapseAlias(label, connection)

  def addSynapseAlias(self, label, subgroups):
    if label in self.connection_aliases:
      print "Synapse alias %s already exists"% label
      return None
    alias = ConnectionAlias(label, subgroups)
    self.connection_aliases[label] = alias
    return alias

  def addStimulus(self, 
                  type_name, 
                  parameters, 
                  groups, 
                  probability, 
                  start_time, 
                  end_time):
    if not self.simulation.isMaster():
      return True
    group_list = None
    if isinstance(groups, list):
      group_list = list(groups)
    else:
      group_list = [groups]
    group_names = []
    for group in group_list:
      if isinstance(group, str):
        if group not in self.cell_aliases:
          print "Cell group or alias %s was never registered." % group
          return False
        group_names.append(group)
      elif isinstance(group, CellAlias):
        if not group.name:
          print "Anonymous CellAlias cannot be used for input."
          return False
        group_names.append(group.name)
      elif isinstance(group, CellGroup):
        if not group.name:
          print "Anonymous CellGroup cannot be used for input."
          return False
        group_names.append(group.name)
      else:
        print "Unknown input group type."
        return False

    if not group_names:
      print "No groups specified for Input."
      return False

    model_parameters = self.buildModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build model parameters for input"
      return False
    model_parameters.thisown = False
    input_group = pyncs.InputGroup(pyncs.string_list(group_names),
                                   model_parameters,
                                   probability,
                                   start_time,
                                   end_time)
    return self.simulation.addInput(input_group)

  def addReport(self, targets, target_type, attribute, probability, start_time, end_time):
    if not self.simulation.isMaster():
      return EmptyReport() 
    target = pyncs.Report.Unknown
    target_list = None
    if isinstance(targets, list):
      target_list = list(targets)
    else:
      target_list = [targets]
    target_names = None
    if target_type == "neuron":
      target = pyncs.Report.Neuron
      alias = CellAlias(None, targets)
      if not self.resolveCellAlias_(alias):
        print "A target was ill-defined."
        return None
      target_names = [ x.name for x in alias.subgroups ]
    elif target_type == "synapse":
      target = pyncs.Report.Synapse
      alias = ConnectionAlias(None, targets)
      if not self.resolveConnectionAlias_(alias):
        print "A target was ill-defined."
        return None 
      target_names = [ x.name for x in alias.subgroups ]
    else:
      print "Invalid target specified."
      return None 

    report = pyncs.Report(pyncs.string_list(target_names),
                          target,
                          attribute,
                          probability,
                          start_time,
                          end_time)
    data_source = self.simulation.addReport(report)
    if not data_source:
      print "Failed to add report."
      return None 
    return Report(data_source)

  def init(self, argv):
    self.model_specification = pyncs.ModelSpecification()
    self.model_specification.thisown = False
#self.model_specification.model_parameters = (
#     pyncs.string_to_model_parameters_map(self.model_parameters)
#   )
    neuron_group_map = {}
    for name, cell_group in self.cell_groups.items():
      model_parameters = self.getNeuronParameters(cell_group.parameters)
      if not model_parameters:
        print "ModelParameters %s not found" % cell_group.parameters
        return False
      neuron_group = pyncs.NeuronGroup(cell_group.count,
                                       model_parameters,
                                       cell_group.geometry)
      cell_group.neuron_group = neuron_group
      neuron_group.thisown = False
      neuron_group_map[name] = neuron_group
    self.model_specification.neuron_groups = (
      pyncs.string_to_neuron_group_map(neuron_group_map)
    )
    for name, alias in self.cell_aliases.items():
      if not alias.resolved:
        if not self.resolveCellAlias_(alias):
          print "Failed to resolve CellAlias %s" % name
          return False
      neuron_groups = [x.neuron_group for x in alias.subgroups]
      neuron_group_list = pyncs.neuron_group_list(neuron_groups)
      alias.neuron_alias = pyncs.NeuronAlias(neuron_group_list)
    neuron_alias_map = { n : a.neuron_alias 
                         for n, a in self.cell_aliases.items() }
    self.model_specification.neuron_aliases = (
      pyncs.string_to_neuron_alias_map(neuron_alias_map)
    )

    connection_map = {}
    for name, connection in self.connections.items():
      if not connection.presynaptic.resolved:
        if not self.resolveCellAlias_(connection.presynaptic):
          print "Invalid presynaptic group in connection %s" % name
          return False
      if not connection.postsynaptic.resolved:
        if not self.resolveCellAlias_(connection.postsynaptic):
          print "Invalid postsynaptic group in connection %s" % name
          return False
      model_parameters = self.getSynapseParameters(connection.parameters)
      if not model_parameters:
        print "ModelParameters %s not found" % connection.parameters
        return False

      group = pyncs.neuron_group_list
      presynaptic_neuron_groups = [x.neuron_group 
                                   for x in connection.presynaptic.subgroups]
      presynaptic = group([x.neuron_group 
                           for x in connection.presynaptic.subgroups])
      postsynaptic = group([x.neuron_group
                            for x in connection.postsynaptic.subgroups])
      synapse_group = pyncs.SynapseGroup(presynaptic,
                                         postsynaptic,
                                         model_parameters,
                                         connection.probability)
      connection.synapse_group = synapse_group
      synapse_group.thisown = False
      connection_map[name] = synapse_group
    self.model_specification.synapse_groups = (
      pyncs.string_to_synapse_group_map(connection_map)
    )

    for name, alias in self.connection_aliases.items():
      if not alias.resolved:
        if not self.resolveConnectionAlias_(alias):
          print "Failed to resolve ConnectionAlias %s" % name
          return False
      synapse_groups = [x.synapse_group for x in alias.subgroups]
      synapse_group_list = pyncs.synapse_group_list(synapse_groups)
      alias.synapse_alias = pyncs.SynapseAlias(synapse_group_list)
    synapse_alias_map = { n : a.synapse_alias
                          for n, a in self.connection_aliases.items() }
    self.model_specification.synapse_aliases = (
      pyncs.string_to_synapse_alias_map(synapse_alias_map)
    )
  
    self.simulation = pyncs.Simulation(self.model_specification,
                                       self.simulaton_parameters)
    return self.simulation.init(pyncs.string_list(argv))

  def step(self, steps = 1):
    if self.simulation.isMaster():
      for i in range(0, steps):
        self.simulation.step()

  def wait(self):
    self.simulation.wait()

  def run(self, steps = -1, duration = -1):
    if steps <= 0 and duration <= 0:
      print "Either steps or duration must be greater than zero for run()."
      return None
    if steps > 0 and duration > 0:
      print "Both steps and duration cannot be specified for run()."
      return None
    if steps > 0:
      self.step(steps)
    if duration > 0:
      # This appears to fix rounding problems with smaller durations
      milliseconds = duration * 1000
      timestep_in_milliseconds = self.simulaton_parameters.getTimeStep() * 1000
      num_steps = int(milliseconds) / int(timestep_in_milliseconds)
      if num_steps <= 0:
        print "Duration is too small to actually step the simulation in run()."
        return None
      self.step(num_steps)

  def getNeuronParameters(self, parameters):
    if isinstance(parameters, pyncs.ModelParameters):
      return parameters
    if parameters in self.neuron_parameters:
      return self.neuron_parameters[parameters]
    return None

  def getSynapseParameters(self, parameters):
    if isinstance(parameters, pyncs.ModelParameters):
      return parameters
    if parameters in self.synapse_parameters:
      return self.synapse_parameters[parameters]
    return None

  def getCellGroup(self, group):
    if isinstance(group, CellGroup):
      return group
    if group in self.cell_groups:
      return self.cell_groups[group]
    return None

  def getConnection(self, connection):
    if isinstance(connection, Connection):
      return connection
    if connection in self.connections:
      return self.connections[connection]
    return None

  def buildGenerator_(self, v):
    if isinstance(v, float):
      return pyncs.ExactDouble(v)
    elif isinstance(v, int):
      return pyncs.ExactInteger(v)
    elif isinstance(v, list):
      generators = [ self.buildGenerator_(x) for x in v ]
      if not all(generators):
        print "Failed to build a generator inside a list generator"
        return False
      for x in generators:
        x.thisown = False
      gen_list = pyncs.generator_list()
      for g in generators:
        gen_list.push_back(g)
      return pyncs.ExactList(gen_list)
    elif isinstance(v, Uniform):
      return v.build()
    elif isinstance(v, Normal):
      return v.build()
    elif isinstance(v, str):
      return pyncs.ExactString(v)
    elif isinstance(v, dict):
      subtype_name = ""
      if "type" in v:
        subtype_name = v["type"]
      sub_parameters = self.buildModelParameters_(subtype_name, v)
      if not sub_parameters:
        print "Failed to build subparameters inside generator."
        return None
      sub_parameters.thisown = False
      return pyncs.ExactParameters(sub_parameters)
    else:
      print "Unrecognized parameter", v
      return None

  def buildModelParameters_(self, type_name, parameters):
    parameter_map = {}
    for k, v in parameters.items():
      generator = self.buildGenerator_(v)
      if not generator:
        print "Failed to build generator for %s" % k
        return None
      generator.thisown = False
      parameter_map[k] = generator
    return pyncs.ModelParameters(type_name,
                                 pyncs.string_to_generator_map(parameter_map))

  def resolveCellAlias_(self, alias):
    visited_aliases = set()
    cell_groups = set()
    to_visit = list(alias.subgroups)
    alias.resolving = True
    while to_visit:
      v = to_visit.pop()
      if isinstance(v, CellAlias):
        if v.resolving:
          print "A CellAlias loop was detected."
          return False
        if not v.resolved:
          if not self.resolveCellAlias_(v):
            print "Failed to resolve CellAlias."
            return False
        for g in v.subgroups:
          cell_groups.add(g)
      elif isinstance(v, CellGroup):
        cell_groups.add(v)
      elif isinstance(v, str):
        if v not in self.cell_aliases:
          print "No CellAlias named %s found." % v
          return False
        alias_object = self.cell_aliases[v]
        to_visit.append(alias_object)
    alias.resolving = False
    alias.resolved = True
    alias.subgroups = cell_groups
    return True

  def resolveConnectionAlias_(self, alias):
    visited_aliases = set()
    connections = set()
    to_visit = list(alias.subgroups)
    alias.resolving = True
    while to_visit:
      v = to_visit.pop()
      if isinstance(v, ConnectionAlias):
        if v.resolving:
          print "A ConnectionAlias loop was detected."
          return False
        if not v.resolved:
          if not self.resolveConnectionAlias_(v):
            print "Failed to resolve ConnectionAlias."
            return False
        for g in v.subgroups:
          connections.add(g)
      elif isinstance(v, Connection):
        connections.add(v)
      elif isinstance(v, str):
        if v not in self.connections:
          print "No ConnectionAlias named %s found." % v
          return False
        alias_object = self.connection_aliases[v]
        to_visit.append(alias_object)
    alias.resolving = False
    alias.resolved = True
    alias.subgroups = connections 
    return True

  '''**********Functions added for Python to JSON conversion**********'''

  def parseNeuron(self, label, type_name, parameters):
    if self.getNeuronParameters(label):
      print "NeuronParameters %s already exists." % label
      return None
    model_parameters = self.parseModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build NeuronParameters %s" % label
      return None
    model_parameters['name'] = label
    self.neuron_parameters[label] = model_parameters
    return model_parameters

  def parseSynapse(self, label, type_name, parameters):
    if self.getSynapseParameters(label):
      print "SynapseParameters %s already exists." % label
      return None
    model_parameters = self.parseModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build SynapseParameters %s" % label
      return None
    model_parameters['name'] = label
    self.synapse_parameters[label] = model_parameters
    return model_parameters

  def parseNeuronGroup(self, label, count, parameters, geometry = None):
    if self.getCellGroup(label):
      print "NeuronGroup %s already exists." % label
      return None
    cell_group = CellGroup(label, count, parameters, geometry)
    self.cell_groups[label] = cell_group
    return self.addNeuronAlias(label, cell_group)

  def parseNeuronAlias(self, label, subgroups):
    if label in self.cell_aliases:
      print "Neuron alias %s already exists" % label
      return None
    alias = CellAlias(label, subgroups)
    self.cell_aliases[label] = alias
    return alias
  
  def parseSynapseGroup(self, label, presynaptic, postsynaptic, probability, parameters):
    if self.getConnection(label):
      print "Connection %s already exists." % label
      return None
    connection = Connection(label,
                            presynaptic,
                            postsynaptic,
                            probability,
                            parameters)
    self.connections[label] = connection
    return self.addSynapseAlias(label, connection)

  def parseSynapseAlias(self, label, subgroups):
    if label in self.connection_aliases:
      print "Synapse alias %s already exists"% label
      return None
    alias = ConnectionAlias(label, subgroups)
    self.connection_aliases[label] = alias
    return alias

  def parseStimulus(self, 
                  type_name, 
                  parameters, 
                  groups, 
                  probability, 
                  start_time, 
                  end_time):
    group_list = None
    if isinstance(groups, list):
      group_list = list(groups)
    else:
      group_list = [groups]
    group_names = []
    for group in group_list:
      if isinstance(group, str):
        if group not in self.cell_aliases:
          print "Cell group or alias %s was never registered." % group
          return False
        group_names.append(group)
      elif isinstance(group, CellAlias):
        if not group.name:
          print "Anonymous CellAlias cannot be used for input."
          return False
        group_names.append(group.name)
      elif isinstance(group, CellGroup):
        if not group.name:
          print "Anonymous CellGroup cannot be used for input."
          return False
        group_names.append(group.name)
      else:
        print "Unknown input group type."
        return False

    if not group_names:
      print "No groups specified for Input."
      return False

    model_parameters = self.parseModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build model parameters for input"
      return False

    input_group = { "group_names": group_names,
                    "model_parameters": model_parameters,
                    "probability": probability,
                    "start_time": start_time,
                    "end_time": end_time
                  }
    self.parse_stim_spec.append(input_group)
    return input_group

  def parseReport(self, targets, target_type, attribute, probability, start_time, end_time):
    target_list = None
    if isinstance(targets, list):
      target_list = list(targets)
    else:
      target_list = [targets]
    target_names = None
    if target_type == "neuron":
      alias = CellAlias(None, targets)
      if not self.resolveCellAlias_(alias):
        print "A target was ill-defined."
        return None
      target_names = [ x.name for x in alias.subgroups ]
    elif target_type == "synapse":
      alias = ConnectionAlias(None, targets)
      if not self.resolveConnectionAlias_(alias):
        print "A target was ill-defined."
        return None 
      target_names = [ x.name for x in alias.subgroups ]
    else:
      print "Invalid target specified."
      return None 

    report = { "target_names": target_names,
                "target_type": target_type,
                "report_type": attribute,
                "probability": probability,
                "start_time": start_time,
                "end_time": end_time
              } 
    self.parse_report_spec.append(report)
    return Report(None)

  def parseGenerator_(self, v):
    if isinstance(v, float):
      return {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": v
            }
    elif isinstance(v, int):
      return {
              "maxValue": 0, 
              "mean": 0, 
              "minValue": 0, 
              "stddev": 0, 
              "type": "exact", 
              "value": v
            }
    elif isinstance(v, list):
      generators = [ self.parseGenerator_(x) for x in v ]
      if not all(generators):
        print "Failed to build a generator inside a list generator"
        return False
      for x in generators:
        x.thisown = False
      gen_list = []
      for g in generators:
        gen_list.append(g)
      return gen_list

    elif isinstance(v, Uniform):
      return  {"maxValue": v.upper_bound, 
               "mean": 0,
               "minValue": v.lower_bound,
               "stddev": 0, 
               "type": "uniform",
               "value": 0
              }
    elif isinstance(v, Normal):
      return  {"maxValue": 0,
               "mean": v.mean, 
               "minValue": 0,
               "stddev": v.std_dev, 
               "type": "uniform",
               "value": 0
              }
    elif isinstance(v, str):
      return v
    elif isinstance(v, dict):
      subtype_name = ""
      if "type" in v:
        subtype_name = v["type"]
      sub_parameters = self.parseModelParameters_(subtype_name, v)
      if not sub_parameters:
        print "Failed to build subparameters inside generator."
        return None
      return sub_parameters
    else:
      print "Unrecognized parameter", v
      return None

  def parseModelParameters_(self, type_name, parameters):
    parameter_map = {}
    for k, v in parameters.items():
      generator = self.parseGenerator_(v)
      if not generator:
        print "Failed to build generator for %s" % k
        return None
      parameter_map[k] = generator
    parameter_map['type'] = type_name
    return parameter_map

  def parseInit(self, argv):
    neuron_group_map = {}
    for name, cell_group in self.cell_groups.items():
      model_parameters = cell_group.parameters
      if not model_parameters:
        print "ModelParameters %s not found" % cell_group.parameters
        return False
      neuron_group = {"count": cell_group.count,
                      "model_parameters": model_parameters,
                      "geometry": cell_group.geometry
                      }

      cell_group.neuron_group = neuron_group
      neuron_group_map[name] = neuron_group
    self.parse_model_spec['neuron_groups'] = neuron_group_map

    for name, alias in self.cell_aliases.items():
      if not alias.resolved:
        if not self.resolveCellAlias_(alias):
          print "Failed to resolve CellAlias %s" % name
          return False
      neuron_groups = [x.neuron_group for x in alias.subgroups]
      neuron_group_list = neuron_groups
      alias.neuron_alias = neuron_group_list
    neuron_alias_map = { n : a.neuron_alias 
                         for n, a in self.cell_aliases.items() }
    self.parse_model_spec['neuron_aliases'] = neuron_alias_map

    connection_map = {}
    for name, connection in self.connections.items():
      if not connection.presynaptic.resolved:
        if not self.resolveCellAlias_(connection.presynaptic):
          print "Invalid presynaptic group in connection %s" % name
          return False
      if not connection.postsynaptic.resolved:
        if not self.resolveCellAlias_(connection.postsynaptic):
          print "Invalid postsynaptic group in connection %s" % name
          return False
      model_parameters = connection.parameters
      if not model_parameters:
        print "ModelParameters %s not found" % connection.parameters
        return False

      presynaptic_neuron_groups = [x.neuron_group 
                                   for x in connection.presynaptic.subgroups]
      presynaptic = [x.neuron_group 
                           for x in connection.presynaptic.subgroups]
      postsynaptic = [x.neuron_group
                            for x in connection.postsynaptic.subgroups]

      synapse_group = { "presynaptic": presynaptic,
                        "postsynaptic": postsynaptic,
                        "model_parameters": model_parameters,
                        "probability": connection.probability
                      }
      connection.synapse_group = synapse_group
      connection_map[name] = synapse_group
    self.parse_model_spec['synapse_groups'] = connection_map

    for name, alias in self.connection_aliases.items():
      if not alias.resolved:
        if not self.resolveConnectionAlias_(alias):
          print "Failed to resolve ConnectionAlias %s" % name
          return False
      synapse_groups = [x.synapse_group for x in alias.subgroups]
      synapse_group_list = synapse_groups
      alias.synapse_alias = synapse_group_list
    synapse_alias_map = { n : a.synapse_alias
                          for n, a in self.connection_aliases.items() }
    self.parse_model_spec['synapse_aliases'] = synapse_alias_map

    return self.parse_model_spec


  def parseRun(self, steps = -1, duration = -1):
    if steps <= 0 and duration <= 0:
      print "Either steps or duration must be greater than zero for run()."
      return None
    if steps > 0 and duration > 0:
      print "Both steps and duration cannot be specified for run()."
      return None

    if steps > 0:
      self.parse_run_spec['steps'] = steps
    if duration > 0:
      self.parse_run_spec['duration'] = duration
    return self.parse_run_spec
