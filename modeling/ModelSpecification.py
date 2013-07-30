import json
import os
import six

class ModelSpecification:
  def __init__(self, path):
    self.neuron_groups = {}
    self.neuron_aliases = {}
    self.synapse_groups = {}
    self.synapse_aliases = {}
    self.model_parameters = {}
    self.geometry_generators = {}
    self.Load(path, None)

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
        "model_parameters": self.model_parameters,
        "neuron_group": self.neuron_groups,
        "neuron_alias": self.neuron_aliases,
        "synapse_group": self.synapse_groups,
        "synapse_alias": self.synapse_aliases,
        "geometry_generator": self.geometry_generators,
      }
      for source, destination in transfer_locations.items():
        for name, definition in definitions_by_type[source].items():
          destination[name] = definition
