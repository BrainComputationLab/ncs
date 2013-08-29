import six

import pyncs

def BuildExact(spec):
  value = spec["value"]
  if isinstance(value, six.string_types):
    return pyncs.ExactString(value)
  elif isinstance(value, six.integer_types):
    return pyncs.ExactInteger(value)
  elif isinstance(value, float):
    return pyncs.ExactDouble(value)
  else:
    print "Unrecognized exact value type"
    return None

def BuildNormal(spec):
  def isnumber(x):
    return isinstance(x, six.integer_types) or isinstance(x, float)
  mean = spec["mean"]
  std_dev = spec["std_dev"]
  if not isnumber(mean):
    print "mean must be a numerical type"
    return None
  if not isnumber(std_dev):
    print "std_dev must be a numerical type"
    return None
  return pyncs.NormalDouble(mean, std_dev)

def BuildUniform(spec):
  min_value = spec["min_value"]
  max_value = spec["max_value"]
  if isinstance(min_value, six.integer_types):
    if isinstance(max_value, six.integer_types):
      return pyncs.UniformInteger(min_value, max_value)
    else:
      print "Both min_value and max_value must be both integer or both float."
      return None
  elif isinstance(min_value, float):
    if isinstance(max_value, float):
      return pyncs.UniformDouble(min_value, max_value)
    else:
      print "Both min_value and max_value must be both integer or both float."
      return None
  else:
    print "Unrecognized data type for min_value for uniform generator."
    return None

def Build(spec):
  type_switch = {
    "exact" : BuildExact,
    "normal" : BuildNormal,
    "uniform": BuildUniform
  }
  generator_type = spec["type"]
  if generator_type not in type_switch:
    print "Unrecognized generator type %s" % generator_type
    return None
  return type_switch[generator_type](spec)
  
