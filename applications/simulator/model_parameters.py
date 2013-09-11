import generator
import pyncs

def Build(spec):
  model_type = spec["model_type"]
  parameters = pyncs.string_to_generator_map()
  parameters = {}
  for key, generator_spec in spec.items():
    if key == "model_type":
      continue
    value_generator = generator.Build(generator_spec)
    parameters[str(key)] = value_generator
    print value_generator.base().name()
  for value in parameters.values():
    print value.base().name()
  return pyncs.ModelParameters(str(model_type), pyncs.string_to_generator_map(parameters))

