# An example brain specification
# The parameters don't represent an actual biological brain model; this file
# simply illustrates how a model is specified. It should load into the
# simulation, but its behavior is arbitrary.

# Import ncs_spec components - for simplicity, dropping all objects into the
# global namespace

from ncs_spec import *

# Parameters specific to a particular model type are defined using dictionaries
# Note that this does not actually instantiate any cells; in fact, it can be
# reused to instantiate multiple groups of cells with similar properties.
izhikevich_cell_specification = {
  "a": ExactDouble(1.0),
  "b": NormalDouble(2.0, 0.3),
  "c": NormalDouble(4.0, 0.5),
  "d": ExactDouble(6.0),
  "u": ExactDouble(-1.0),
  "v": ExactDouble(-40.0)
}

# Position the cells based on a uniform spherical distribution
# Because the radius takes a generator, we can control how cells are
# positioned. For example, UniformDouble(0.0,1.0) would uniformly distribute
# cells by radius (not volumetrically even) while ExactDouble(1.0) would
# essentially form a spherical shell of cells.
spherical_distribution = (
  Sphere(UniformDouble(0.0, 1.0)) # how the radius should be determined
)

# Instantiate a group of actual cells
izhikevich_cell_group_1 = (
  NeuronSpecification("izhikevich", # type name
                      "izh_group_1", # a unique name so we can access it later
                      100, # the number of cells we want,
                      izhikevich_cell_specification, # parameters
                      spherical_distribution) # how to position the cells
)

# For fun, programatically generate two more groups using the same parameters
extra_groups = [NeuronSpecification("izikevich",
                                    "izh_group_%d" % i,
                                    100 * i,
                                    izhikevich_cell_specification,
                                    spherical_distribution)
                for i in [2,3]]
izhikevich_cell_group_2 = extra_groups[0]
izhikevich_cell_group_3 = extra_groups[1]

# For organizational purposes, we can cluster cell groups together
group12 = NeuronCluster("izh_group_12") # a unique name is required
group12.addGroup(izhikevich_cell_group_1)
group12.addGroup(izhikevich_cell_group_2)

# Groups can be made of other groups
group123 = NeuronCluster("izh_group_123")
group123.addGroup(group12)
group123.addGroup(izhikevich_cell_group_3)

# Synapses are specified in similar fashion
flat_synapse_specification = {
  "current": NormalDouble(1.0, 0.5),
  "delay": UniformInteger(5, 20), # delay is in timesteps
}

# Connect group 1 to 2
synapse_12 = SynapseSpecification("flat", # type name
                                  "synapse_12", # unique name
                                  izhikevich_cell_group_1, # presynaptic group
                                  izhikevich_cell_group_2, # postsynaptic group
                                  0.5, # connection probability
                                  flat_synapse_specification) # parameters

# Connect the super group to itself using the same parameters
synapse_123123 = SynapseSpecification("flat",
                                      "synapse_123123",
                                      group123,
                                      group123,
                                      0.1,
                                      flat_synapse_specifiation)

# We can also cluster synapse groups together for organization
all_synapses = SynapseCluster("all_synapses")
all_synapses.addGroup(synapse_12)
all_synapses.addGroup(synapse_123123)

# Suppose we had another model we wanted to import and in it was a cell cluster
# named "embedded_cells"
# Load the model
LoadModelFile("embedded_file", # namespace to load all objects into
              "embedded.py") # the filename

# We can pull the cell cluster out now that it's loaded
# The colon is a reserved character used to separate namespaces from object
# names
embedded_cells = GetCellGroup("embedded_file:embedded_cells")

# We can also chain namespaces - suppose embedded.py loaded its own file
double_embedded_synapses = (
  GetSynapseGroup("embedded_file:embedded_namespace:synapses")
)

# Finally, for more complex models (say, NCS LIF with channels)
# Channels can be separately specified
kahp_spec = {
   "type": ExactString("kahp"),
  # some parameters
}

ka_spec = {
  "type": ExactString("ka"),
  # some parameters
}

# The NCS plugin will first look for num_channels, then given the number of
# channels will search for channel[x]
ncs_specification = {
  # other parameters omitted
  "num_channels": ExactInteger(2),
  "channel[0]": ExactSpecification(kahp_spec),
  "channel[1]": ExactSpecification(ka_spec),
  "some_file": ExactString("path_to_file"),
}
