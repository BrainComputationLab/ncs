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
#include <ncs/sim/Parallel.h>
#include <ncs/sim/PublisherExtractor.h>
#include <ncs/sim/ReportController.h>
#include <ncs/sim/ReportReceiver.h>
#include <ncs/sim/ReportSyncer.h>
#include <ncs/sim/Signal.h>
#include <ncs/sim/Simulator.h>

#include "ModelParameters.pb.h"

namespace ncs {

namespace sim {

Simulator::Simulator(spec::ModelSpecification* model_specification,
                     spec::SimulationParameters* simulation_parameters)
  : model_specification_(model_specification),
    simulation_parameters_(simulation_parameters),
    communicator_(nullptr),
    vector_communicator_(nullptr),
    neurons_(nullptr),
    vector_exchange_controller_(new VectorExchangeController()),
    global_vector_publisher_(new GlobalVectorPublisher()),
    simulation_controller_(new SimulationController()),
    report_managers_(new ReportManagers()) {
}

bool Simulator::initialize(int argc, char** argv) {
  if (nullptr == model_specification_) {
    std::cerr << "No ModelSpecification was given." << std::endl;
    return false;
  }
  if (nullptr == simulation_parameters_) {
    std::cerr << "Invalid SimulationParameters." << std::endl;
    return false;
  }
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
  vector_communicator_ = Communicator::global();
  if (!communicator_) {
    std::cerr << "Failed to create vector communicator." << std::endl;
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

  std::clog << "Loading input simulator plugins..." << std::endl;
  if (!loadInputSimulatorPlugins_()) {
    std::cerr << "Failed to load input simulator plugins." << std::endl;
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

  std::clog << "Initializing reportables..." << std::endl;
  if (!initializeReporters_()) {
    std::cerr << "Failed to initialize reports." << std::endl;
  }

  std::clog << "Starting device threads..." << std::endl;
  if (!startDevices_()) {
    std::cerr << "Failed to start device threads." << std::endl;
    return false;
  }
  
  std::clog << "Starting global exchangers..." << std::endl;
  if (!vector_exchange_controller_->start()) {
    std::cerr << "Failed to start VectorExchangeController." << std::endl;
    return false;
  }
  if (!global_vector_publisher_->start()) {
    std::cerr << "Failed to start GlobalVectorPublisher." << std::endl;
    return false;
  }
  for (auto extractor : remote_extractors_) {
    if (!extractor->start()) {
      std::cerr << "Failed to start RemoteVectorExtractor." << std::endl;
      return false;
    }
  }
  for (auto publisher : remote_publishers_) {
    if (!publisher->start()) {
      std::cerr << "Failed to start RemoteVectorPublisher." << std::endl;
      return false;
    }
  }
  if (!isMaster()) {
    worker_thread_ = std::thread([this](){ this->workerFunction_(); });
  }

  std::clog << "Initialization complete..." << std::endl;

  return true;
}

bool Simulator::step() {
  if (isMaster()) {
    int command = Step;
    communicator_->bcast(&command, 1, 0);
  }
  return simulation_controller_->step();
}

bool Simulator::addInput(spec::InputGroup* input) {
  simulation_controller_->idle();
  if (isMaster()) {
    int command = AddInput;
    communicator_->bcast(&command, 1, 0);
    com::InputGroup protobuffer;
    input->toProtobuf(&protobuffer);
    std::string buffer;
    protobuffer.SerializeToString(&buffer);
    communicator_->bcast(buffer, 0);
  }
  std::vector<spec::NeuronAlias*> aliases;
  for (const auto& alias_name : input->getNeuronAliases()) {
    auto alias = getNeuronAlias_(alias_name);
    if (nullptr == alias) {
      std::cerr << "Neuron alias " << alias_name << " not found." << 
        std::endl;
      return false;
    }
    aliases.push_back(alias);
  }
  std::vector<spec::NeuronGroup*> potential_groups;
  for (auto alias : aliases) {
    for (auto group : alias->getGroups()) {
      potential_groups.push_back(group);
    }
  }
  std::vector<Neuron*> potential_neurons;
  if (!getNeuronsInGroups_(potential_groups, &potential_neurons)) {
    std::cerr << "Failed to expand all neuron aliases." << std::endl;
    return false;
  }

  spec::ModelParameters* parameters = input->getModelParameters();
  const std::string& type = input->getModelParameters()->getType();
  auto instantiator = input_simulator_generators_->getInstantiator(type);
  if (!instantiator) {
    std::cerr << "Failed to get instantiator for input type " << type <<
      std::endl;
    return false;
  }
  void* instantiator_data = instantiator(input->getModelParameters());
  if (nullptr == instantiator_data) {
    std::cerr << "Failed to build instantiator data for input of type " <<
      type << std::endl;
    return false;
  }

  // TODO(rvhoang): seed this
  ncs::spec::RNG rng(0);
  auto gen = [&](unsigned int i) {
    return std::uniform_int_distribution<unsigned int>(0, i - 1)(rng);
  };
  std::random_shuffle(potential_neurons.begin(), potential_neurons.end(), gen);
  std::vector<std::vector<Input*>> inputs_per_device;
  for (size_t i = 0; i < devices_.size(); ++i) {
    inputs_per_device.push_back(std::vector<Input*>());
  }
  unsigned int num_inputs = input->getProbability() * potential_neurons.size();
  num_inputs = std::min(num_inputs, (unsigned int)potential_neurons.size());
  unsigned int this_machine_index = cluster_->getThisMachineIndex();
  for (unsigned int i = 0; i < num_inputs; ++i) {
    int seed = rng();
    auto neuron = potential_neurons[i];
    if (neuron->location.machine != this_machine_index) {
      continue;
    }
    Input* in = new Input();
    in->seed = seed;
    in->neuron_device_id = neuron->id.device;
    inputs_per_device[neuron->location.device].push_back(in);
  }

  // TODO(rvhoang): thread this
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (inputs_per_device[i].empty()) {
      continue;
    }
    auto device = devices_[i];
    device->threadInit();
    bool result = device->addInput(inputs_per_device[i],
                                   instantiator_data,
                                   type,
                                   input->getStartTime(),
                                   input->getEndTime());
    device->threadDestroy();
  }
  return true;
}

bool Simulator::isMaster() const {
  return 0 == communicator_->getRank();
}

DataSink* Simulator::addReport(spec::Report* report) {
  // Transmit the report spec to all the other nodes
  if (isMaster()) {
    int command = AddReport;
    communicator_->bcast(&command, 1, 0);
    com::Report protobuffer;
    report->toProtobuf(&protobuffer);
    std::string buffer;
    protobuffer.SerializeToString(&buffer);
    communicator_->bcast(buffer, 0);
  }
  // Make a separate communicator for reporting
  Communicator* communicator = Communicator::global();

  if (report->getTarget() == spec::Report::Neuron) {
    // Get all neuron aliases
    std::vector<spec::NeuronAlias*> aliases;
    for (const auto& alias_name : report->getAliases()) {
      auto alias = getNeuronAlias_(alias_name);
      // We can't fail here. It might just be that the alias doesn't exist
      // on this particular machine
      if (alias) {
        aliases.push_back(alias);
      }
    }
    // Get all the groups in those aliases
    std::vector<spec::NeuronGroup*> potential_groups;
    for (auto alias : aliases) {
      for (auto group : alias->getGroups()) {
        potential_groups.push_back(group);
      }
    }
    // Expand all neuron groups into a list of potential neurons 
    // Again, we can't fail out here, we might just have no neurons
    std::vector<Neuron*> potential_neurons;
    getNeuronsInGroups_(potential_groups, &potential_neurons);
    // Shuffle the potential
    // TODO(rvhoang): seed this
    ncs::spec::RNG rng(0);
    auto gen = [&](unsigned int i) {
      return std::uniform_int_distribution<unsigned int>(0, i - 1)(rng);
    };
    std::random_shuffle(potential_neurons.begin(),
                        potential_neurons.end(),
                        gen);
    // Take only as many as we need
    size_t num_to_select =
      potential_neurons.size() * report->getPercentage();
    num_to_select = std::min(num_to_select, potential_neurons.size());
    potential_neurons.resize(num_to_select);

    // Make sure the report type is real
    int my_machine_index = cluster_->getThisMachineIndex();
    const auto neuron_manager = report_managers_->getNeuronManager();
    const auto report_name = report->getReportName();
    const auto data_description = neuron_manager->getDescription(report_name);
    bool status = true;
    if (nullptr == data_description) {
      // If the report doesn't exist on this machine, then we need to be
      // sure that none of the selected neurons exist on this machine as
      // well; otherwise, this is erroneous.
      // A global scope report will be available on every machine despite only
      // the master handling it
      for (auto neuron : potential_neurons) {
        if (neuron->location.machine == my_machine_index) {
          std::cerr << "Report of name " << report_name << " was not found " <<
            "on a machine that requires it." << std::endl;
          status = false;
        }
      }
    } else {
      // If the dataspace is global, clear all neurons as the master will
      // handle it
      if (data_description->getDataSpace() == DataSpace::Global) {
        if (!isMaster()) {
          potential_neurons.clear();
        }
      } else { // Remove all neurons not on this machine
        auto not_on_machine = [my_machine_index](Neuron* n) {
          return n->location.machine != my_machine_index;
        };
        auto new_end = std::remove_if(potential_neurons.begin(),
                                      potential_neurons.end(),
                                      not_on_machine);
        potential_neurons.erase(new_end, potential_neurons.end());
      }
    }

    DataSpace::Space data_space = DataSpace::Unknown;
    DataType::Type data_type = DataType::Unknown;
    if (data_description) {
      data_space = data_description->getDataSpace();
      data_type = data_description->getDataType();
    }
    int space = data_space;
    int type = data_type;
    if (isMaster()) {
      // Get the final data description from all nodes and make sure they're
      // either unknown or the same value
      for (int i = 1; i < communicator->getNumProcesses(); ++i) {
        int remote_space = DataSpace::Unknown;
        int remote_type = DataType::Unknown;
        communicator->recv(remote_space, i);
        communicator->recv(remote_type, i);
        if (space == DataSpace::Unknown) {
          space = remote_space;
        }
        if (type == DataType::Unknown) {
          type = remote_type;
        }
      }
    } else {
      communicator->send(space, 0);
      communicator->send(type, 0);
    }
    communicator->bcast(space, 0);
    communicator->bcast(type, 0);
    if (data_space != DataSpace::Unknown &&
        data_space != space) {
      std::cerr << "Mismatched data spaces." << std::endl;
      status = false;
    }
    if (data_type != DataType::Unknown &&
        data_type != type) {
      std::cerr << "Mismatched data types." << std::endl;
      status = false;
    }
    data_space = static_cast<DataSpace::Space>(space);
    data_type = static_cast<DataType::Type>(type);

    // Make sure we're all good here
    if (!communicator->syncState(status)) {
      std::cerr << "An error occurred setting up this report." << std::endl;
      delete communicator;
      return nullptr;
    }

    // Sort neurons by their location
    auto CompareNeuronLocation = [=](const Neuron* a, const Neuron* b) {
      return a->location < b->location;
    };
    std::sort(potential_neurons.begin(),
              potential_neurons.end(),
              CompareNeuronLocation);

    // Construct a set of accessors based on the dataspace
    std::function<bool(Neuron* a, Neuron* b)> same_location;
    std::function<Location(Neuron* n)> relevant_location;
    std::function<unsigned int(Neuron*)> get_index;
    switch(data_space) {
      case DataSpace::Global:
        same_location = [](Neuron* a, Neuron* b) {
          return true;
        };
        relevant_location = [](Neuron* a) {
          return Location(0, -1, -1);
        };
        get_index = [](Neuron* n) {
          return n->id.global;
        };
        break;
      case DataSpace::Machine:
        same_location = [](Neuron* a, Neuron* b) {
          return a->location.machine == b->location.machine;
        };
        relevant_location = [](Neuron* a) {
          return Location(a->location.machine, -1, -1);
        };
        get_index = [](Neuron* n) {
          return n->id.machine;
        };
        break;
      case DataSpace::Device:
        same_location = [](Neuron* a, Neuron* b) {
          return a->location.machine == b->location.machine &&
            a->location.device == b->location.device;
        };
        relevant_location = [](Neuron* a) {
          return Location(a->location.machine, a->location.device, -1);
        };
        get_index = [](Neuron* n) {
          return n->id.device;
        };
        break;
      case DataSpace::Plugin:
        same_location = [](Neuron* a, Neuron* b) {
          return a->location.machine == b->location.machine &&
            a->location.device == b->location.device &&
            a->location.plugin == b->location.plugin;
        };
        relevant_location = [](Neuron* a) {
          return a->location;
        };
        get_index = [](Neuron* n) {
          return n->id.plugin;
        };
        break;
      default:
        std::cerr << "Invalid data space." << std::endl;
        status = false;
    }

    if (!communicator->syncState(status)) {
      std::cerr << "Invalid data space." << std::endl;
      delete communicator;
      return nullptr;
    }

    // Partition neurons by their relevant location
    // By this point, all Locations are those that reside on this machine
    std::map<Location, std::vector<Neuron*>> neurons_by_location;
    {
      auto it = potential_neurons.begin();
      while (it != potential_neurons.end()) {
        Neuron* base_neuron = *it; 
        Location location = relevant_location(base_neuron);
        auto SameAsCurrentLocation = [base_neuron, same_location](Neuron* n) {
          return same_location(base_neuron, n);
        };
        auto segment_end = std::find_if_not(it,
                                            potential_neurons.end(),
                                            SameAsCurrentLocation);
        neurons_by_location[location] = std::vector<Neuron*>(it, segment_end);
        it = segment_end;
      }
    }

    std::vector<Location> locations;
    for (auto it : neurons_by_location) {
      locations.push_back(it.first);
    }
    unsigned int machine_total_bytes = 0;
    unsigned int num_total_elements = 0;
    unsigned int num_real_elements = 0;
    ReportController* report_controller = new ReportController();
    std::vector<PublisherExtractor*> publisher_extractors;
    for (size_t i = 0; i < locations.size(); ++i) {
      size_t byte_offset = machine_total_bytes;
      Location& l = locations[i];
      std::vector<Neuron*>& n = neurons_by_location[l];
      size_t num_neurons = n.size();
      num_real_elements += num_neurons;
      num_total_elements += DataType::num_padded_elements(num_neurons,
                                                          data_type);
      size_t num_bytes = DataType::num_bytes(num_neurons, data_type);
      machine_total_bytes += num_bytes;
      auto publisher = neuron_manager->getSource(report_name,
                                                 l.machine,
                                                 l.device,
                                                 l.plugin);

      if (!publisher) {
        std::cerr << "Failed to find a publisher for report " <<
          report_name << std::endl;
        status = false;
        continue;
      }
      std::vector<unsigned int> indices;
      for (auto neuron : n) {
        indices.push_back(get_index(neuron));
      }
      PublisherExtractor* extractor = new PublisherExtractor();
      if (!extractor->init(byte_offset,
                           data_type,
                           indices,
                           report_name,
                           publisher,
                           report_controller)) {
        std::cerr << "Failed to initialize PublisherExtractor." << std::endl;
        status = false;
        continue;
      }
      publisher_extractors.push_back(extractor);
    }
    if (!communicator->syncState(status)) {
      for (auto extractor : publisher_extractors) {
        delete extractor;
      }
      delete report_controller;
      delete communicator;
      return nullptr;
    }
    if (isMaster()) {
      std::vector<SpecificPublisher<Signal>*> dependents;
      for (auto extractor : publisher_extractors) {
        dependents.push_back(extractor);
      }
      std::vector<ReportReceiver*> report_receivers;
      for (size_t i = 1; i < communicator->getNumProcesses(); ++i) {
        size_t byte_offset = machine_total_bytes;
        unsigned int remote_num_real_elements = 0;
        unsigned int remote_num_total_elements = 0;
        unsigned int remote_num_bytes = 0; 
        communicator->recv(remote_num_real_elements, i);
        communicator->recv(remote_num_total_elements, i);
        communicator->recv(remote_num_bytes, i);
        num_real_elements += remote_num_real_elements;
        num_total_elements += remote_num_total_elements;
        machine_total_bytes += remote_num_bytes;
        ReportReceiver* receiver = new ReportReceiver();
        if (!receiver->init(byte_offset,
                            remote_num_bytes,
                            communicator,
                            i,
                            report_controller,
                            Constants::num_buffers)) {
          std::cerr << "Failed to initialize ReportReceiver." << std::endl;
          delete receiver;
          status = false;
          continue;
        }
        report_receivers.push_back(receiver);
        dependents.push_back(receiver);
      }
      if (!report_controller->init(machine_total_bytes,
                                   Constants::num_buffers)) {
        std::cerr << "Failed to initialize ReportController." << std::endl;
        status = false;
      }
      MasterReportSyncer* syncer = new MasterReportSyncer();
      if (!syncer->init(publisher_extractors,
                        report_receivers,
                        communicator,
                        report_controller)) {
        std::cerr << "Failed to initialize MasterReportSyncer." << std::endl;
        status = false;
      }
      DataSink* data_sink = new DataSink(DataDescription(data_space, data_type),
                                         num_total_elements - num_real_elements,
                                         num_real_elements,
                                         Constants::num_buffers);
      if (!data_sink->init(dependents,
                           report_controller,
                           syncer)) {
        std::cerr << "Failed to initialize DataSink." << std::endl;
        status = false;
      }

      if (!communicator->syncState(status)) {
        std::cerr << "Report communication failed to initialize." << std::endl;
        delete data_sink;
        return nullptr;
      }
      return data_sink;
    } else {
      // Send the master the number of elements we're responsible for 
      communicator->send(num_real_elements, 0);
      communicator->send(num_total_elements, 0);
      communicator->send(machine_total_bytes, 0);
      std::vector<SpecificPublisher<Signal>*> dependents;
      for (auto extractor : publisher_extractors) {
        dependents.push_back(extractor);
      }
      ReportSender* sender = new ReportSender();
      if (!sender->init(communicator,
                        0,
                        dependents,
                        report_controller)) {
        std::cerr << "Failed to initialize ReportSender." << std::endl;
        status = false;
      }
      WorkerReportSyncer* syncer = new WorkerReportSyncer();
      if (!syncer->init(publisher_extractors,
                        communicator,
                        report_controller,
                        sender)) {
        std::cerr << "Failed to initialize WorkerReportSyncer." << std::endl;
        status = false;
      }
      if (!report_controller->init(machine_total_bytes,
                                   Constants::num_buffers)) {
        std::cerr << "Failed to initialize ReportController." << std::endl;
        status = false;
      }
      if (!communicator->syncState(status)) {
        std::cerr << "Report communication failed to initialize." << std::endl;
        delete syncer;
        return false;
      }
      auto thread_function = [syncer]() {
        syncer->run();
        delete syncer;
      };
      std::thread thread(thread_function);
      thread.detach();
      return nullptr;
    }
  } else if (report->getTarget() == spec::Report::Synapse) {
    // TODO(rvhoang): this is more complicated since synapse information only
    // exists on the machine the synapse resides on
  }
  return nullptr;
}

Simulator::~Simulator() {
  if (isMaster()) {
    int command = Shutdown;
    communicator_->bcast(&command, 1, 0);
  }
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  std::clog << "Shutting down simulation." << std::endl;
  ParallelDelete pd;
  pd.add(simulation_controller_, "SimulationController");
  pd.add(vector_exchange_controller_, "VectorExchangeController");
  pd.add(global_vector_publisher_, "GlobalVectorPublisher");
  pd.add(devices_, "Device");
  pd.add(remote_publishers_, "RemoteVectorPublisher");
  pd.add(remote_extractors_, "RemoteVectorExtractor");
  pd.wait();
  if (communicator_) {
    delete communicator_;
  }
  if (vector_communicator_) {
    delete vector_communicator_;
  }
  MPI::finalize();
  std::clog << "Shut down complete." << std::endl;
}

void Simulator::workerFunction_() {
  int command = -1;
  while (communicator_->bcast(&command, 1, 0)) {
    switch(command) {
      case Shutdown:
        return;
        break;
      case Step:
        step();
        break;
      case AddInput:
        {
          std::string buffer;
          communicator_->bcast(buffer, 0);
          com::InputGroup protobuffer;
          protobuffer.ParseFromString(buffer);
          spec::InputGroup* input_group = 
            spec::InputGroup::fromProtobuf(&protobuffer);
          addInput(input_group);
        }
        break;
      case AddReport:
        {
          std::string buffer;
          communicator_->bcast(buffer, 0);
          com::Report protobuffer;
          protobuffer.ParseFromString(buffer);
          spec::Report* report = spec::Report::fromProtobuf(&protobuffer);
          addReport(report);
        }
        break;
      default:
        std::cerr << "Unrecognized command " << command << std::endl;
        return;
    };
  }
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
    neuron_global_id_offsets_per_machine_.push_back(global_id);
    unsigned int machine_id = 0;
    MachineDescription* machine = cluster_->getMachines()[machine_index];
    for (unsigned int device_index = 0;
         device_index < machine->getDevices().size();
         ++device_index) {
      if (machine_index == cluster_->getThisMachineIndex()) {
        neuron_global_id_offsets_per_my_devices_.push_back(global_id);
      }
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
    size_t machine_vector_size = 
      global_id - neuron_global_id_offsets_per_machine_[machine_index];
    neuron_vector_size_per_machine_.push_back(machine_vector_size);
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
        synapse->location.device = postsynaptic_neuron->location.device;
        synapse->location.machine = postsynaptic_neuron->location.machine;
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
  if (!vector_exchange_controller_->init(global_neuron_vector_size_,
                                         Constants::num_buffers)) {
    std::cerr << "Failed to initialize VectorExchangeController." << std::endl;
    return false;
  }
  // Get the vector extractor for each device
  std::vector<SpecificPublisher<Signal>*> device_vector_extractors;
  for (auto device : devices_) {
    device_vector_extractors.push_back(device->getVectorExtractor());
  }

  // Set up a publisher for each remote machine
  {
    unsigned int this_machine_index = cluster_->getThisMachineIndex();
    size_t vector_size = neuron_vector_size_per_machine_[this_machine_index];
    size_t vector_offset = 
      neuron_global_id_offsets_per_machine_[this_machine_index];
    for (size_t i = 0; i < vector_communicator_->getNumProcesses(); ++i) {
      if (this_machine_index == i) {
        continue;
      }
      auto remote_publisher = new RemoteVectorPublisher();
      if (!remote_publisher->init(vector_offset,
                                  vector_size,
                                  vector_communicator_,
                                  i,
                                  vector_exchange_controller_,
                                  device_vector_extractors)) {
        std::cerr << "Failed to initialize RemoteVectorPublisher." <<
          std::endl;
        delete remote_publisher;
        return false;
      }
      remote_publishers_.push_back(remote_publisher);
    }
  }

  // Set up an extractor for each remote machine
  {
    unsigned int this_machine_index = cluster_->getThisMachineIndex();
    for (size_t i = 0; i < vector_communicator_->getNumProcesses(); ++i) {
      if (i == this_machine_index) {
        continue;
      }
      size_t vector_size = neuron_vector_size_per_machine_[i];
      size_t vector_offset = neuron_global_id_offsets_per_machine_[i];
      auto remote_extractor = new RemoteVectorExtractor();
      if (!remote_extractor->init(vector_offset,
                                  vector_size,
                                  vector_communicator_,
                                  i,
                                  vector_exchange_controller_,
                                  Constants::num_buffers)) {
        std::cerr << "Failed to initialize RemoteVectorExtractor." <<
          std::endl;
        delete remote_extractor;
        return false;
      }
      remote_extractors_.push_back(remote_extractor);
    }
  }
  // Setup the global vector publisher
  {
    std::vector<SpecificPublisher<Signal>*> dependents;
    for (auto device_extractor : device_vector_extractors) {
      dependents.push_back(device_extractor);
    }
    for (auto remote_extractor : remote_extractors_) {
      dependents.push_back(remote_extractor);
    }
    if (!global_vector_publisher_->init(global_neuron_vector_size_,
                                        Constants::num_buffers,
                                        dependents,
                                        vector_exchange_controller_)) {
      std::cerr << "Failed to initialize GlobalVectorPublisher." << std::endl;
      return false;
    }
  }
  return true;
}

bool Simulator::initializeReporters_() {
  int machine_index = cluster_->getThisMachineIndex();
  auto neuron_manager = report_managers_->getNeuronManager();
  bool result = true;
  result &= neuron_manager->addDescription("neuron_fire",
                                           DataDescription(DataSpace::Global,
                                                           DataType::Bit));
  result &= neuron_manager->addSource("neuron_fire",
                                      machine_index,
                                      -1,
                                      -1,
                                      global_vector_publisher_);
  if (!result) {
    std::cerr << "Failed to add Simulator-wide reports." << std::endl;
    return false;
  }
  
  for (int i = 0; i < devices_.size(); ++i) {
    if (!devices_[i]->initializeReporters(machine_index,
                                          i,
                                          report_managers_)) {
      std::cerr << "Failed to initialize device reports." << std::endl;
      return false;
    }
  }
  return true;
}

bool Simulator::loadInputSimulatorPlugins_() {
  auto plugin_path_ptr = std::getenv("NCS_PLUGIN_PATH");
  if (!plugin_path_ptr) {
    std::cerr << "NCS_PLUGIN_PATH was not set." << std::endl;
    return false;
  }
  std::string plugin_path(plugin_path_ptr);
  std::vector<std::string> paths = File::getContents(plugin_path + "/input");
  input_simulator_generators_ =
    PluginLoader<InputSimulator>::loadPaths(paths, "InputSimulator");
  return nullptr != input_simulator_generators_;
  return true;
}

bool Simulator::initializeDevices_() {
  bool result = true;
  auto& my_devices = cluster_->getThisMachine()->getDevices();

  // Instantiate all devices first
  for (size_t i = 0; i < my_devices.size(); ++i) {
    DeviceDescription* description = my_devices[i];
    DeviceBase* device = nullptr;
    switch(description->getDeviceType()) {
#ifdef NCS_CUDA
    case DeviceType::CUDA:
      device = new CUDADevice(description->getDeviceIndex());
      break;
#endif // NCS_CUDA
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
    devices_.push_back(device);
  }
  // Initialize each device
  for (size_t i = 0; i < my_devices.size(); ++i) {
    DeviceDescription* description = my_devices[i];
    DeviceBase* device = devices_[i];

    if (!device->threadInit()) {
      std::cerr << "Failed to initialize device thread." << std::endl;
      delete device;
      return false;
    }
    if (!device->initialize(description,
                            neuron_simulator_generators_,
                            synapse_simulator_generators_,
                            input_simulator_generators_,
                            vector_exchange_controller_,
                            global_vector_publisher_,
                            global_neuron_vector_size_,
                            neuron_global_id_offsets_per_my_devices_[i],
                            simulation_controller_,
                            simulation_parameters_)) {
      std::cerr << "Failed to initialize device." << std::endl;
      delete device;
      return false;
    }
    if (!device->threadDestroy()) {
      std::cerr << "Failed to destroy device thread." << std::endl;
      delete device;
      return false;
    }
  }
  return true;
}

bool Simulator::startDevices_() {
  bool result = true;
  for (auto device : devices_) {
    result &= device->start();
  }
  return result;
}

spec::NeuronAlias* Simulator::getNeuronAlias_(const std::string& alias) const {
  const auto& alias_map = model_specification_->neuron_aliases;
  auto search_result = alias_map.find(alias);
  if (alias_map.end() == search_result) {
    return nullptr;
  }
  return search_result->second;
}

spec::SynapseAlias* Simulator::
getSynapseAlias_(const std::string& alias) const {
  const auto& alias_map = model_specification_->synapse_aliases;
  auto search_result = alias_map.find(alias);
  if (alias_map.end() == search_result) {
    return nullptr;
  }
  return search_result->second;
}

bool Simulator::
getNeuronsInGroups_(const std::vector<spec::NeuronGroup*>& groups,
                    std::vector<Neuron*>* neurons) const {
  bool result = true;
  for (auto group : groups) {
    result &= getNeuronsInGroup_(group, neurons);
  }
  return result;
}

bool Simulator::getNeuronsInGroup_(spec::NeuronGroup* group, 
                          std::vector<Neuron*>* neurons) const {
  auto search_result = neurons_by_group_.find(group);
  if (neurons_by_group_.end() == search_result) {
    return false;
  }
  const auto& neuron_list = search_result->second;
  for (auto neuron : neuron_list) {
    neurons->push_back(neuron);
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
