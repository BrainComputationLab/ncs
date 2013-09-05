#include <algorithm>
#include <mpi.h>
#include <queue>
#include <unistd.h>

#include <ncs/sim/Bit.h>
#include <ncs/sim/ClusterDescription.h>
#include <ncs/sim/File.h>
#include <ncs/sim/MPI.h>
#include <ncs/sim/Simulator.h>

namespace ncs {

namespace sim {

Simulator::Simulator(spec::ModelSpecification* model_specification)
  : model_specification_(model_specification),
    communicator_(nullptr),
    neurons_(nullptr) {
}

bool Simulator::initialize(int argc, char** argv) {
  std::clog << "Initializing MPI..." << std::endl;
  if (!MPI::initialize(argc, argv)) {
    std::cerr << "Failed to initialize MPI." << std::endl;
    return false;
  }

  std::clog << "Obtaining global communicator..." << std::endl;
  communicator_ = Communicator::global();
  if (!communicator_) {
    std::cerr << "Failed to create global communicator." << std::endl;
    return false;
  }

  std::clog << "Loading neuron simulator plugins..." << std::endl;
  if (!loadNeuronSimulatorPlugins_()) {
    std::cerr << "Failed to load neuron simulator plugins." << std::endl;
    return false;
  }

  std::clog << "Loading neuron instantiators..." << std::endl;
  if (!loadNeuronInstantiators_()) {
    std::cerr << "Failed to load neuron instantiators." << std::endl;
    return false;
  }

  std::clog << "Loading synapse simulator plugins..." << std::endl;
  if (!loadSynapseSimulatorPlugins_()) {
    std::cerr << "Failed to load synapse simulator plugins." << std::endl;
    return false;
  }

  std::clog << "Loading synapse instantiators..." << std::endl;
  if (!loadSynapseInstantiators_()) {
    std::cerr << "Failed to load synapse instantiators." << std::endl;
    return false;
  }

  std::clog << "Gathering cluster data..." << std::endl;
  if (!gatherClusterData_(DeviceType::CPU | DeviceType::CUDA)) {
    std::cerr << "Failed to gather cluster data." << std::endl;
    return false;
  }

  std::clog << "Gathering model statistics..." << std::endl;
  if (!gatherModelStatistics_()) {
    std::cerr << "Failed to gather model statistics." << std::endl;
    return false;
  }

  std::clog << "Allocating neurons..." << std::endl;
  if (!allocateNeurons_()) {
    std::cerr << "Failed to allocate neurons." << std::endl;
    return false;
  }

  std::clog << "Distributing neurons..." << std::endl;
  if (!distributeNeurons_()) {
    std::cerr << "Failed to distribute neurons." << std::endl;
    return false;
  }

  std::clog << "Assigning neuron IDs..." << std::endl;
  if (!assignNeuronIDs_()) {
    std::cerr << "Failed to assign neuron IDs." << std::endl;
    return false;
  }

  std::clog << "Initialization complete..." << std::endl;

  return true;
}

bool Simulator::gatherClusterData_(unsigned int enabled_device_types) {
  cluster_ =
    ClusterDescription::getThisCluster(communicator_,
                                       enabled_device_types);
  if (nullptr == cluster_) {
    std::cerr << "Failed to get cluster information." << std::endl;
    return false;
  }
  return true;
}

bool Simulator::gatherModelStatistics_() {
  model_statistics_ = new ModelStatistics(model_specification_);
  return nullptr != model_statistics_;
}

bool Simulator::loadNeuronSimulatorPlugins_() {
  auto plugin_path_ptr = std::getenv("NCS_PLUGIN_PATH");
  if (!plugin_path_ptr) {
    std::cerr << "NCS_PLUGIN_PATH was not set." << std::endl;
    return false;
  }
  std::string plugin_path(plugin_path_ptr);
  std::vector<std::string> paths = File::getContents(plugin_path + "/neuron");
  neuron_simulator_generators_ =
    PluginLoader<NeuronSimulator>::loadPaths(paths, "NeuronSimulator");
  return nullptr != neuron_simulator_generators_;
}

bool Simulator::loadNeuronInstantiators_() {
  for (auto key_value : model_specification_->neuron_groups) {
    spec::NeuronGroup* group = key_value.second;
    const std::string& type = group->getModelParameters()->getType();
    auto instantiator = neuron_simulator_generators_->getInstantiator(type);
    if (!instantiator) {
      std::cerr << "Failed to get instantiator for group " <<
        key_value.first << " of type " << type << std::endl;
      return false;
    }
    neuron_instantiators_by_group_[group] = 
      instantiator(group->getModelParameters());
  }
  return true;
}

bool Simulator::loadSynapseSimulatorPlugins_() {
  auto plugin_path_ptr = std::getenv("NCS_PLUGIN_PATH");
  if (!plugin_path_ptr) {
    std::cerr << "NCS_PLUGIN_PATH was not set." << std::endl;
    return false;
  }
  std::string plugin_path(plugin_path_ptr);
  std::vector<std::string> paths = File::getContents(plugin_path + "/synapse");
  synapse_simulator_generators_ =
    PluginLoader<SynapseSimulator>::loadPaths(paths, "SynapseSimulator");
  return nullptr != synapse_simulator_generators_;
}

bool Simulator::loadSynapseInstantiators_() {
  for (auto key_value : model_specification_->synapse_groups) {
    spec::SynapseGroup* group = key_value.second;
    const std::string& type = group->getModelParameters()->getType();
    auto instantiator = synapse_simulator_generators_->getInstantiator(type);
    if (!instantiator) {
      std::cerr << "Failed to get instantiator for group " <<
        key_value.first << " of type " << type << std::endl;
      return false;
    }
    synapse_instantiators_by_group_[group] = 
      instantiator(group->getModelParameters());
  }
  return true;
}

bool Simulator::allocateNeurons_() {
  if (neurons_) {
    std::cerr << "Neurons were already allocated." << std::endl;
    return false;
  }
  neurons_ = new Neuron[model_statistics_->getNumberOfNeurons()];
  unsigned int allocated_neurons = 0;
  for (auto key_value : model_specification_->neuron_groups) {
    spec::NeuronGroup* group = key_value.second;
    neurons_by_group_[group] = std::vector<Neuron*>();
    std::vector<Neuron*>& group_neurons = neurons_by_group_[group];
    unsigned int num_neurons = group->getNumberOfCells();
    void* instantiator_data = neuron_instantiators_by_group_[group];
    for (unsigned int i = 0; i < num_neurons; ++i) {
      Neuron* neuron = neurons_ + allocated_neurons;
      group_neurons.push_back(neuron);
      neuron->instantiator = instantiator_data;
      // TODO: fill in seed here
      ++allocated_neurons;
    }
  }
  return true;
}

bool Simulator::distributeNeurons_() {
  double total_compute_power = cluster_->estimateTotalPower();
  std::map<spec::NeuronGroup*, double> load_estimates =
    model_statistics_->estimateNeuronLoad();
  double total_compute_load = model_statistics_->estimateTotalLoad();

  struct DeviceLoad {
    DeviceDescription* device;
    double max_load;
    double current_load;
  };
  auto LowerDeviceLoad = [](DeviceLoad* left, DeviceLoad* right) {
    return (left->max_load - left->current_load) >
           (right->max_load - right->current_load);
  };
  std::priority_queue<DeviceLoad*,
                      std::vector<DeviceLoad*>,
                      decltype(LowerDeviceLoad)> device_loads(LowerDeviceLoad);
  for (auto machine : cluster_->getMachines()) {
    for (auto device : machine->getDevices()) {
      DeviceLoad* load = new DeviceLoad();
      load->device = device;
      load->current_load = 0.0;
      load->max_load =
        total_compute_load * device->getPower() / total_compute_power;
      device_loads.push(load);
    }
  }

  if (device_loads.empty()) {
    std::cerr << "No compute devices available." << std::endl;
    return false;
  }

  // Distribute neurons by distributing the heaviest ones first and then
  // using the smaller ones to fill in the cracks
  std::vector<spec::NeuronGroup*> neuron_groups_by_load;
  for (auto key_value : load_estimates) {
    neuron_groups_by_load.push_back(key_value.first);
  }
  std::sort(neuron_groups_by_load.begin(),
            neuron_groups_by_load.end(),
            [&load_estimates](spec::NeuronGroup* l, spec::NeuronGroup* r) {
              return load_estimates[l] > load_estimates[r];
            });

  for (auto group : neuron_groups_by_load) {
    const std::vector<Neuron*>& group_neurons = neurons_by_group_[group];
    double load_per_neuron = load_estimates[group];
    unsigned int neurons_allocated = 0;
    unsigned int num_neurons = group_neurons.size();
    while (neurons_allocated < num_neurons) {
      DeviceLoad* device_load = device_loads.top();
      device_loads.pop();

      double available_compute =
        device_load->max_load - device_load->current_load;
      unsigned int num_to_allocate = available_compute / load_per_neuron;
      num_to_allocate = std::min(num_to_allocate,
                                 num_neurons - neurons_allocated);
      num_to_allocate = std::max(1u, num_to_allocate);
      device_load->current_load += num_to_allocate * load_per_neuron;

      const std::string& model_type = group->getModelParameters()->getType();
      NeuronPluginDescription* plugin =
        device_load->device->getNeuronPlugin(model_type);
      for (unsigned int i = 0; i < num_to_allocate; ++i) {
        plugin->addNeuron(group_neurons[neurons_allocated++]);
      }
      device_loads.push(device_load);
    }
  }
  return true;
}

bool Simulator::assignNeuronIDs_() {
  unsigned int global_id = 0;
  for (unsigned int machine_index = 0;
       machine_index < cluster_->getMachines().size();
       ++machine_index) {
    unsigned int machine_id = 0;
    MachineDescription* machine = cluster_->getMachines()[machine_index];
    for (unsigned int device_index = 0;
         device_index < machine->getDevices().size();
         ++device_index) {
      unsigned int device_id = 0;
      DeviceDescription* device = machine->getDevices()[device_index];
      for (unsigned int plugin_index = 0;
           plugin_index < device->getNeuronPlugins().size();
           ++plugin_index) {
        unsigned int plugin_id = 0;
        NeuronPluginDescription* plugin =
          device->getNeuronPlugins()[plugin_index];
        for (auto neuron : plugin->getNeurons()) {
          neuron->location.machine = machine_index;
          neuron->location.device = device_index;
          neuron->location.plugin = plugin_index;
          neuron->id.global = global_id++;
          neuron->id.machine = machine_id++;
          neuron->id.device = device_id++;
          neuron->id.plugin = plugin_id++;
        }
        device_id = Bit::pad(device_id);
        machine_id = Bit::pad(machine_id);
        global_id = Bit::pad(global_id);
      }
    }
  }
  return true;
}

} // namespace sim

} // namespace ncs
