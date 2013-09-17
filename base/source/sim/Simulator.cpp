#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <ncs/sim/Bit.h>
#include <ncs/sim/ClusterDescription.h>
#include <ncs/sim/CUDADevice.h>
#include <ncs/sim/File.h>
#include <ncs/sim/MPI.h>
#include <ncs/sim/Simulator.h>

namespace ncs {

namespace sim {

Simulator::Simulator(spec::ModelSpecification* model_specification)
  : model_specification_(model_specification),
    communicator_(nullptr),
    neurons_(nullptr),
    vector_exchanger_(new MachineVectorExchanger()) {
}

bool Simulator::initialize(int argc, char** argv) {
  std::clog << "Initializing seeds..." << std::endl;
  if (!initializeSeeds_()) {
    std::cerr << "Failed to initialize seeds." << std::endl;
    return false;
  }

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

  std::clog << "Distributing synapses..." << std::endl;
  if (!distributeSynapses_()) {
    std::cerr << "Failed to distribute synapses." << std::endl;
    return false;
  }

  std::clog << "Initializing devices..." << std::endl;
  if (!initializeDevices_()) {
    std::cerr << "Failed to initialize devices." << std::endl;
    return false;
  }

  std::clog << "Initializing MachineVectorExchanger..." << std::endl;
  if (!initializeVectorExchanger_()) {
    std::cerr << "Failed to initialize MachineVectorExchanger." << std::endl;
    return false;
  }

  std::clog << "Initialization complete..." << std::endl;

  return true;
}

bool Simulator::initializeSeeds_() {
  // TODO(rvhoang): get seeds from input
  neuron_seed_ = 0;
  synapse_seed_ = 0;
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
    auto gen = group->getModelParameters()->getGenerator("a");
    auto instantiator = neuron_simulator_generators_->getInstantiator(type);
    if (!instantiator) {
      std::cerr << "Failed to get instantiator for group " <<
        key_value.first << " of type " << type << std::endl;
      return false;
    }
    void* instantiator_data = instantiator(group->getModelParameters());
    if (!instantiator_data) {
      std::cerr << "Failed to create instantiator for group " <<
        key_value.first << std::endl;
      return false;
    }
    neuron_instantiators_by_group_[group] = instantiator_data;
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
    void* instantiator_data = instantiator(group->getModelParameters());
    if (!instantiator_data) {
      std::cerr << "Failed to create instantiator for group " <<
        key_value.first << std::endl;
      return false;
    }
    synapse_instantiators_by_group_[group] = instantiator_data;
  }
  return true;
}

bool Simulator::allocateNeurons_() {
  if (neurons_) {
    std::cerr << "Neurons were already allocated." << std::endl;
    return false;
  }
  spec::RNG rng(getNeuronSeed_());
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
      neuron->seed = rng();
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
    return (left->max_load - left->current_load) <
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

  while (!device_loads.empty()) {
    delete device_loads.top();
    device_loads.pop();
  }
  return true;
}

bool Simulator::assignNeuronIDs_() {
  unsigned int global_id = 0;
  for (unsigned int machine_index = 0;
       machine_index < cluster_->getMachines().size();
       ++machine_index) {
    neuron_global_id_offsets_.push_back(global_id);
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
  global_neuron_vector_size_ = global_id;
  return true;
}

bool Simulator::distributeSynapses_() {
  unsigned int this_device_index = cluster_->getThisMachineIndex();
  auto isOnThisMachine = [=](Neuron* neuron) {
    return neuron->location.machine == this_device_index;
  };
  std::mt19937 generator(getSynapseSeed_());

  std::map<std::string, std::vector<Synapse*>> this_machine_synapses_by_type;
  for (auto key_value : model_specification_->synapse_groups) {
    const auto& type = key_value.second->getModelParameters()->getType();
    if (this_machine_synapses_by_type.count(type) == 0) {
      this_machine_synapses_by_type[type] = std::vector<Synapse*>();
    }
    std::vector<Synapse*>& synapses = this_machine_synapses_by_type[type];

    spec::SynapseGroup* group = key_value.second;
    void* instantiator = synapse_instantiators_by_group_[group];
    const auto& presynaptic_groups = group->getPresynapticGroups();
    unsigned int num_presynaptic_neurons = 0;
    std::vector<unsigned int> presynaptic_counts;
    for (auto neuron_group : presynaptic_groups) {
      num_presynaptic_neurons += neuron_group->getNumberOfCells();
      presynaptic_counts.push_back(neuron_group->getNumberOfCells());
    }
    const auto& postsynaptic_groups = group->getPostsynapticGroups();
    unsigned int num_postsynaptic_neurons = 0;
    std::vector<unsigned int> postsynaptic_counts;
    for (auto neuron_group : postsynaptic_groups) {
      num_postsynaptic_neurons += neuron_group->getNumberOfCells();
      postsynaptic_counts.push_back(neuron_group->getNumberOfCells());
    }
    double probability = group->getConnectionProbability();
    unsigned int num_connections =
      num_presynaptic_neurons *
      num_postsynaptic_neurons *
      probability;
    std::discrete_distribution<>
      presynaptic_distribution(presynaptic_counts.begin(),
                               presynaptic_counts.end());
    std::discrete_distribution<>
      postsynaptic_distribution(postsynaptic_counts.begin(),
                                postsynaptic_counts.end());
    for (unsigned int i = 0; i < num_connections; ++i) {
      spec::NeuronGroup* presynaptic_group =
        presynaptic_groups[presynaptic_distribution(generator)];
      spec::NeuronGroup* postsynaptic_group =
        postsynaptic_groups[postsynaptic_distribution(generator)];
      std::uniform_int_distribution<unsigned int> neuron_selector;
      const auto& presynaptic_neurons = neurons_by_group_[presynaptic_group];
      const auto& postsynaptic_neurons = neurons_by_group_[postsynaptic_group];
      Neuron* presynaptic_neuron =
        presynaptic_neurons[neuron_selector(generator) %
                            presynaptic_neurons.size()];
      Neuron* postsynaptic_neuron =
        postsynaptic_neurons[neuron_selector(generator) %
                             postsynaptic_neurons.size()];
      // Roll regardless
      unsigned int synapse_seed = generator();
      if (isOnThisMachine(presynaptic_neuron)) {
        Synapse* synapse = new Synapse();
        synapse->seed = synapse_seed;
        synapse->instantiator = instantiator;
        synapse->presynaptic_neuron = presynaptic_neuron;
        synapse->postsynaptic_neuron = postsynaptic_neuron;
        synapse->location.device = presynaptic_neuron->location.device;
        synapse->location.machine = presynaptic_neuron->location.machine;
        synapses.push_back(synapse);
      } else {
        // We don't care about synapses that don't affect our neurons
      }
    }
  }

  const std::vector<DeviceDescription*>& devices =
    cluster_->getThisMachine()->getDevices();
  unsigned int num_devices = devices.size();
  for (auto key_value : this_machine_synapses_by_type) {
    const std::string& type = key_value.first;
    const std::vector<Synapse*>& synapses = key_value.second;
    bool* device_has_this_type = new bool[num_devices];
    for (unsigned int i = 0; i < num_devices; ++i) {
      device_has_this_type[i] = false;
    }
    for (auto synapse : synapses) {
      device_has_this_type[synapse->location.device] = true;
    }
    SynapsePluginDescription** synapse_plugins_by_device =
      new SynapsePluginDescription*[num_devices];
    unsigned int* synapse_plugin_index = new unsigned int[num_devices];
    for (unsigned int i = 0; i < num_devices; ++i) {
      if (device_has_this_type[i]) {
        synapse_plugins_by_device[i] = devices[i]->getSynapsePlugin(type);
        synapse_plugin_index[i] = devices[i]->getSynapsePluginIndex(type);
      } else {
        synapse_plugins_by_device[i] = nullptr;
      }
    }
    delete [] device_has_this_type;

    for (auto synapse : synapses) {
      unsigned int device_location = synapse->location.device;
      synapse->location.plugin = synapse_plugin_index[device_location];
      synapse->id.plugin =
        synapse_plugins_by_device[device_location]->addSynapse(synapse);
    }

    delete [] synapse_plugins_by_device;
    delete [] synapse_plugin_index;
  }
  unsigned int device_id = 0;
  for (auto device : devices) {
    for (auto plugin : device->getSynapsePlugins()) {
      for (auto synapse : plugin->getSynapses()) {
        synapse->id.device = device_id++;
      }
    }
    device_id = Bit::pad(device_id);
  }
  return true;
}

bool Simulator::initializeVectorExchanger_() {
  // TODO(rvhoang): initialize this after devices are ready
  return true;
}

bool Simulator::initializeDevices_() {
  bool result = true;
  for (auto description : cluster_->getThisMachine()->getDevices()) {
    DeviceBase* device = nullptr;
    switch(description->getDeviceType()) {
    case DeviceType::CUDA:
      device = new CUDADevice(description->getDeviceIndex());
      break;
    case DeviceType::CPU:
      device = new Device<DeviceType::CPU>();
      break;
    case DeviceType::CL:
      // TODO(rvhoang): CL memory ops not implemented yet
      // device = new Device<DeviceType::CL>();
      break;
    default:
      std::cerr << "Unknown device type: " << description->getDeviceType() <<
        std::endl;
      break;
    }
    if (!device) {
      std::cerr << "Failed to create a device." << std::endl;
      return false;
    }
    if (!device->threadInit()) {
      std::cerr << "Failed to initialize device thread." << std::endl;
      delete device;
      return false;
    }
    if (!device->initialize(description,
                            neuron_simulator_generators_,
                            synapse_simulator_generators_,
                            vector_exchanger_,
                            global_neuron_vector_size_)) {
      std::cerr << "Failed to initialize device." << std::endl;
      delete device;
      return false;
    }
    if (!device->threadDestroy()) {
      std::cerr << "Failed to destroy device thread." << std::endl;
      delete device;
      return false;
    }
    devices_.push_back(device);
  }
  return true;
}

int Simulator::getNeuronSeed_() const {
  return neuron_seed_;
}

int Simulator::getSynapseSeed_() const {
  return synapse_seed_;
}

}  // namespace sim

}  // namespace ncs
