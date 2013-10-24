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

class Simulation:
  def __init__(self):
    self.model_parameters = {}
    self.cell_groups = {}
    self.cell_aliases = {}
    return

  def addModelParameters(self, label, type_name, parameters):
    if self.getModelParameters(label):
      print "ModelParameters %s already exists." % label
      return None
    model_parameters = self.buildModelParameters_(type_name, parameters)
    if not model_parameters:
      print "Failed to build ModelParameters %s" % label
      return None
    self.model_parameters[label] = model_parameters
    return model_parameters

  def addCellGroup(self, label, count, parameters, geometry = None):
    model_parameters = self.getModelParameters(parameters)
    if not model_parameters:
      print "Failed to get parameters for cell group %s" % label
    if self.getCellGroup(label):
      print "CellGroup %s already exists." % label
      return None
    cell_group = pyncs.NeuronGroup(count, model_parameters, geometry)
    self.cell_groups[label] = cell_group
    if not self.addCellAlias(label, cell_group):
      print "A cell group or cell alias named %s already exists" % label
      return None
    return cell_group

  def addCellAlias(self, label, subgroups):
    if not isinstance(subgroups, list):
      subgroups = [subgroups]

          
    print "TODO aca"
    return

  def connect(self, presynaptic, postsynaptic, probability, parameters):
    print "TODO c"
    return

  def addConnectionAlias(self, label, subgroups):
    print "TODO addc"
    return

  def addInput(self, type_name, parameters, groups, start_time, end_time):
    print "TODO ai"
    return

  def addReport(self, targets, target_type, attribute, probability):
    print "TODO ar"
    return

  def init(self, argv):
    print "TODO i"
    return

  def step(self, steps = 1):
    print "TODO s"
    return

  def getModelParameters(self, parameters):
    if isinstance(parameters, pyncs.ModelParameters):
      return parameters
    if parameters in self.model_parameters:
      return self.model_parameters[parameters]
    return None

  def getCellGroup(self, group):
    if isinstance(group, pyncs.NeuronGroup):
      return group
    if group in self.cell_groups:
      return self.cell_groups[group]
    return None

  def buildModelParameters_(self, type_name, parameters):
    parameter_map = {}
    for k, v in parameters.items():
      generator = None
      if isinstance(v, float):
        generator = pyncs.ExactDouble(v)
      elif isinstance(v, int):
        generator = pyncs.ExactInteger(v)
      elif isinstance(v, list):
        if len(v) != 2:
          print "list as a uniform must be exactly two values"
          print "for parameter %s" % k
          return None
        uniform = Uniform(v[0], v[1])
        generator = uniform.build()
      elif isinstance(v, Uniform):
        generator = v.build()
      elif isinstance(v, Normal):
        generator = v.build()
      elif isinstance(v, str):
        generator = pyncs.ExactString(v)
      else:
        print "Unrecognized parameter", v
        return None
      if not generator:
        print "Failed to build generator for %s" % k
        return None
      parameter_map[k] = generator
    return pyncs.ModelParameters(type_name,
                                 pyncs.string_to_generator_map(parameter_map))


