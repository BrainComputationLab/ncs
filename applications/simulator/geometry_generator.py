import generator
import pyncs

def BuildBox(spec):
  x_def = spec["x"]
  y_def = spec["y"]
  z_def = spec["z"]
  x_gen = generator.Build(x_def)
  gens = dict()
  for dim in ["x", "y", "z"]:
    dim_spec = spec[dim]
    dim_gen = generator.Build(dim_spec)
    if not dim_gen:
      print "Failed to create generator for %s" % dim
      return None
    gens[dim] = dim_gen
  return pyncs.BoxGenerator(gens["x"], gens["y"], gens["z"])
  
def Build(spec):
  type_switch = {
    "box" : BuildBox
  }
  generator_type = spec["generator_type"]
  if generator_type not in type_switch:
    print "Unrecognized geometry_generator type %s" % generator_type
    return None
  return type_switch[generator_type](spec)
