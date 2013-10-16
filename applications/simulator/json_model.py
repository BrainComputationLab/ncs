import json
import os
import six

import geometry_generator
import model_parameters
import pyncs

class JSONModel:
  def __init__(self, path):
    self.neuron_group_definitions = {}
    self.neuron_alias_definitions = {}
    self.synapse_group_definitions = {}
    self.synapse_alias_definitions = {}
    self.model_parameters_definitions = {}
    self.geometry_generator_definitions = {}
    self.input_group_definitions = {}
    self.report_definitions = {}
    self.Load(path, None)
    self.valid = self.BuildModel_()

  def Load(self, path, namespace):
    namespace_prefix = ""
    if namespace:
      namespace_prefix = namespace + ":"
    def ResolveHashes(v):
      if isinstance(v, six.integer_types) or isinstance(v, float):
        return v
      if isinstance(v, six.string_types):
        return v.replace("#", namespace_prefix)
      if isinstance(v, list):
        return [ResolveHashes(x) for x in v]
      if isinstance(v, dict):
        return {ResolveHashes(x): ResolveHashes(y) for x, y in v.items()}
      return v

    with open(path, "r") as model_file:
      definitions = ResolveHashes(json.loads(model_file.read()))
      definitions_by_type = {
        "model_parameters": {},
        "neuron_group": {},
        "neuron_alias": {},
        "synapse_group": {},
        "synapse_alias": {},
        "geometry_generator": {},
        "model_import": {},
        "input_group": {},
        "report": {},
      }

      for name, definition in definitions.items():
        definition_type = definition["type"]
        del definition["type"]
        definitions_by_type[definition_type][name] = definition

      base_path = os.path.dirname(path)
      for name, definition in definitions_by_type["model_import"].items():
        sub_namespace = name
        if namespace:
          sub_namespace = namespace + ":" + name
        new_path = os.path.join(base_path, definition["location"])
        self.Load(new_path, sub_namespace)

      transfer_locations = {
        "model_parameters": self.model_parameters_definitions,
        "neuron_group": self.neuron_group_definitions,
        "neuron_alias": self.neuron_alias_definitions,
        "synapse_group": self.synapse_group_definitions,
        "synapse_alias": self.synapse_alias_definitions,
        "geometry_generator": self.geometry_generator_definitions,
        "input_group": self.input_group_definitions,
        "report": self.report_definitions,
      }
      for source, destination in transfer_locations.items():
        for name, definition in definitions_by_type[source].items():
          destination[name] = definition

  def BuildModel_(self):
    if not self.BuildModelParameters_():
      print "Failed to build model_parameters"
      return False
    if not self.BuildGeometryGenerators_():
      print "Failed to build geometry_generators"
      return False
    if not self.BuildNeuronGroups_():
      print "Failed to build neuron_groups"
      return False 
    if not self.BuildNeuronAliases_():
      print "Failed to build neuron_aliases"
      return False
    if not self.BuildSynapseGroups_():
      print "Failed to build synapse_groups"
      return False
    if not self.BuildSynapseAliases_():
      print "Failed to build synapse_aliases"
      return False
    if not self.BuildSpecification_():
      print "Failed to build ModelSpecification"
      return False
    if not self.BuildInputGroups_():
      print "Failed to build input_groups"
      return False
    if not self.BuildReports_():
      print "Failed to build reports"
      return False
    return True

  def BuildModelParameters_(self):
    self.model_parameters = {}
    for name, spec in self.model_parameters_definitions.items():
      parameters = model_parameters.Build(spec)
      if not parameters:
        print "Invalid model_parameters for %s" % name
      self.model_parameters[str(name)] = parameters
    return True

  def BuildGeometryGenerators_(self):
    self.geometry_generators = {}
    for name, spec in self.geometry_generator_definitions.items():
      geo_gen = geometry_generator.Build(spec)
      if not geo_gen:
        print "Failed to build geometry_generator %s" % name
        return False
      self.geometry_generators[name] = geo_gen
    return True

  def BuildNeuronGroups_(self):
    self.neuron_groups = {}
    for name, spec in self.neuron_group_definitions.items():
      count = int(spec["count"])
      model_name = str(spec["specification"])
      geometry_generator_name = str(spec["geometry"])
      if model_name not in self.model_parameters:
        print "In neuron_group %s" % name
        print "  model_parameters %s not found" % model_name
        return False
      model_params = self.model_parameters[model_name]
      if geometry_generator_name not in self.geometry_generators:
        print "In neuron_group %s" % name
        print "  geometry_generator %s not found" % geometry_generator_name
        return False
      geo_gen = self.geometry_generators[geometry_generator_name]
      neuron_group = pyncs.NeuronGroup(count, model_params, geo_gen)
      self.neuron_groups[str(name)] = neuron_group
    return True

  def BuildNeuronAliases_(self):
    self.neuron_aliases = {}
    for name, spec in self.neuron_alias_definitions.items():
      visited_aliases = set()
      neuron_subgroups = set()
      to_visit = list()
      to_visit.append(name)
      while to_visit:
        subgroup_name = to_visit.pop()
        if subgroup_name in visited_aliases:
          print "In neuron_alias %s:" % name
          print "  Loop detected with subgroup %s" % subgroup_name
          return False
        if subgroup_name in self.neuron_groups:
          neuron_subgroups.add(self.neuron_groups[subgroup_name])
        elif subgroup_name in self.neuron_alias_definitions:
          visited_aliases.add(subgroup_name)
          subgroups = self.neuron_alias_definitions[subgroup_name]["subgroups"]
          to_visit = to_visit + subgroups
        else:
          print "In neuron_alias %s:" % name
          print "  subgroup %s is neither an alias or group" % subgroup_name
          return False
      neuron_alias = (
        pyncs.NeuronAlias(pyncs.neuron_group_list(list(neuron_subgroups)))
      )
      self.neuron_aliases[str(name)] = neuron_alias
    for name, group in self.neuron_groups.items():
      neuron_alias = (
        pyncs.NeuronAlias(pyncs.neuron_group_list([group]))
      )
      self.neuron_aliases[str(name)] = neuron_alias
    return True

  def BuildSynapseGroups_(self):
    self.synapse_groups = {}
    for name, spec in self.synapse_group_definitions.items():
      model_name = str(spec["specification"])
      presynaptic_name = str(spec["presynaptic"])
      probability = float(spec["probability"])
      presynaptic_groups = list()
      if presynaptic_name in self.neuron_groups:
        presynaptic_groups.append(self.neuron_groups[presynaptic_name])
      elif presynaptic_name in self.neuron_aliases:
        for subgroup in self.neuron_aliases[presynaptic_name].getSubgroups():
          presynaptic_groups.append(subgroup)
      else:
        print "For synapse_group %s:" % name
        print "  %s is not a neuron_group or neuron_alias" % presynaptic_name
        return False

      postsynaptic_name = str(spec["postsynaptic"])
      postsynaptic_groups = list()
      if postsynaptic_name in self.neuron_groups:
        postsynaptic_groups.append(self.neuron_groups[postsynaptic_name])
      elif postsynaptic_name in self.neuron_aliases:
        for subgroup in self.neuron_aliases[postsynaptic_name].getGroups():
          postsynaptic_groups.append(subgroup)
      else:
        print "For synapse_group %s:" % name
        print "  %s is not a neuron_group or neuron_alias" % postsynaptic_name
        return False

      if model_name not in self.model_parameters:
        print "For synapse_group %s" % name
        print "  %s is not a defined model_parameters" % model_name
        return False
      parameters = self.model_parameters[model_name]
      synapse_group = (
        pyncs.SynapseGroup(pyncs.neuron_group_list(presynaptic_groups),
                           pyncs.neuron_group_list(postsynaptic_groups),
                           parameters,
                           probability)
      )
      self.synapse_groups[str(name)] = synapse_group
    return True

  def BuildSynapseAliases_(self):
    self.synapse_aliases = {}
    for name, spec in self.synapse_alias_definitions.items():
      visited_aliases = set()
      synapse_subgroups = set()
      to_visit = list()
      to_visit.append(name)
      while to_visit:
        subgroup_name = to_visit.pop()
        if subgroup_name in visited_aliases:
          print "In synapse_alias %s:" % name
          print "  Loop detected with subgroup %s" % subgroup_name
          return False
        if subgroup_name in self.synapse_groups:
          synapse_subgroups.add(self.synapse_groups[subgroup_name])
        elif subgroup_name in self.synapse_alias_definitions:
          visited_aliases.add(subgroup_name)
          subgroups = self.synapse_alias_definitions[subgroup_name]["subgroups"]
          to_visit = to_visit + subgroups
        else:
          print "In synapse_alias %s:" % name
          print "  subgroup %s is neither an alias or group" % subgroup_name
          return False
      synapse_alias = (
        pyncs.SynapseAlias(pyncs.synapse_group_list(list(synapse_subgroups)))
      )
      self.synapse_aliases[str(name)] = synapse_alias
    return True

  def BuildInputGroups_(self):
    self.input_groups = {}
    for name, spec in self.input_group_definitions.items():
      probability = float(spec["probability"])
      neuron_alias_name = str(spec["neurons"])
      start_time = float(spec["start_time"])
      end_time = float(spec["end_time"])
      model_name = str(spec["specification"])
      if model_name not in self.model_parameters:
        print "In input_group %s" % name
        print "  model_parameters %s not found" % model_name
      model_params = self.model_parameters[model_name]
      input_group = pyncs.InputGroup(neuron_alias_name,
                                     model_params,
                                     probability,
                                     start_time,
                                     end_time)
      self.input_groups[str(name)] = input_group
    return True

  def BuildReports_(self):
    self.reports = {}
    for name, spec in self.report_definitions.items():
      target = pyncs.Report.Unknown 
      target_string = str(spec["target"])
      if target_string == "neuron":
        target = pyncs.Report.Neuron
      elif target_string == "synapse":
        target = pyncs.Report.Synapse
      else:
        print "invalid target specified in report %s" % name
        return False
      percentage = float(spec["percentage"])
      aliases = [ str(x) for x in spec["aliases"] ]
      report = pyncs.Report(pyncs.string_list(aliases),
                            target,
                            str(spec["attribute"]),
                            percentage)
      self.reports[str(name)] = report
    return True

  def BuildSpecification_(self):
    self.model_specification = pyncs.ModelSpecification()
    self.model_specification.model_parameters = (
      pyncs.string_to_model_parameters_map(self.model_parameters)
    )
    self.model_specification.neuron_groups = (
      pyncs.string_to_neuron_group_map(self.neuron_groups)
    )
    self.model_specification.neuron_aliases = (
      pyncs.string_to_neuron_alias_map(self.neuron_aliases)
    )
    self.model_specification.synapse_groups = (
      pyncs.string_to_synapse_group_map(self.synapse_groups)
    )
    self.model_specification.synapse_aliases = (
      pyncs.string_to_synapse_alias_map(self.synapse_aliases)
    )
    return True

