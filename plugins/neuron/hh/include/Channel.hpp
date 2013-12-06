#include <ncs/sim/DeviceBase.h>

template<ncs::sim::DeviceType::Type MType>
ChannelCurrentBuffer<MType>::ChannelCurrentBuffer() 
  : current_(nullptr),
    reversal_current_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelCurrentBuffer<MType>::init(size_t num_neurons) {
  num_neurons_ = num_neurons;
  if (num_neurons_ > 0) {
    bool result = true;
    result &= ncs::sim::Memory<MType>::malloc(current_, num_neurons_);
    result &= ncs::sim::Memory<MType>::malloc(reversal_current_, num_neurons_);
    return result;
  }
  return true;
}

template<ncs::sim::DeviceType::Type MType>
void ChannelCurrentBuffer<MType>::clear() {
  ncs::sim::Memory<MType>::zero(current_, num_neurons_);
  ncs::sim::Memory<MType>::zero(reversal_current_, num_neurons_);
}

template<ncs::sim::DeviceType::Type MType>
float* ChannelCurrentBuffer<MType>::getCurrent() {
  return current_;
}

template<ncs::sim::DeviceType::Type MType>
float* ChannelCurrentBuffer<MType>::getReversalCurrent() {
  return reversal_current_;
}

template<ncs::sim::DeviceType::Type MType>
ChannelCurrentBuffer<MType>::~ChannelCurrentBuffer() {
  if (current_) {
    ncs::sim::Memory<MType>::free(current_);
  }
  if (reversal_current_) {
    ncs::sim::Memory<MType>::free(reversal_current_);
  }
}

template<ncs::sim::DeviceType::Type MType>
NeuronBuffer<MType>::NeuronBuffer()
  : voltage_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool NeuronBuffer<MType>::init(size_t num_neurons) {
  bool result = true;
  result &= ncs::sim::Memory<MType>::malloc(voltage_, num_neurons);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
float* NeuronBuffer<MType>::getVoltage() {
  return voltage_;
}

template<ncs::sim::DeviceType::Type MType>
NeuronBuffer<MType>::~NeuronBuffer() {
  if (voltage_) {
    ncs::sim::Memory<MType>::free(voltage_);
  }
}

template<ncs::sim::DeviceType::Type MType>
ChannelSimulator<MType>::ChannelSimulator() 
  : neuron_plugin_ids_(nullptr) {
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelSimulator<MType>::initialize() {
  num_channels_ = cpu_neuron_plugin_ids_.size();
  if (!ncs::sim::Memory<MType>::malloc(neuron_plugin_ids_, num_channels_)) {
    std::cerr << "Failed to allocate memory." << std::endl;
    return false;
  }
  const auto CPU = ncs::sim::DeviceType::CPU;
  using namespace ncs::sim;
  if (!ncs::sim::mem::copy<MType, CPU>(neuron_plugin_ids_,
                                       cpu_neuron_plugin_ids_.data(),
                                       num_channels_)) {
    std::cerr << "Failed to copy memory." << std::endl;
    return false;
  }
  return init_();
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelSimulator<MType>::addChannel(void* instantiator,
                                         unsigned int neuron_plugin_id,
                                         int seed) {
  instantiators_.push_back(instantiator);
  cpu_neuron_plugin_ids_.push_back(neuron_plugin_id);
  seeds_.push_back(seed);
  return true;
}

template<ncs::sim::DeviceType::Type MType>
ChannelSimulator<MType>::~ChannelSimulator() {
  if (neuron_plugin_ids_) {
    ncs::sim::Memory<MType>::free(neuron_plugin_ids_);
  }
}

template<ncs::sim::DeviceType::Type MType>
ChannelUpdater<MType>::ChannelUpdater() {
  neuron_subscription_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelUpdater<MType>::
init(std::vector<ChannelSimulator<MType>*> simulators,
     ncs::sim::SpecificPublisher<NeuronBuffer<MType>>* source_publisher,
     const ncs::spec::SimulationParameters* simulation_parameters,
     size_t num_neurons,
     size_t num_buffers) {
  simulators_ = simulators;
  simulation_parameters_ = simulation_parameters;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto blank = new ChannelCurrentBuffer<MType>();
    if (!blank->init(num_neurons)) {
      std::cerr << "Failed to initialize ChannelCurrentBuffer." << std::endl;
      delete blank;
      return false;
    }
    addBlank(blank);
  }
  neuron_subscription_ = source_publisher->subscribe();
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool ChannelUpdater<MType>::start() {
  ncs::sim::DeviceBase* device = ncs::sim::DeviceBase::getThreadDevice();
  if (!device) {
    std::cerr << "ChannelUpdater needs to know thread device." << std::endl;
    return false;
  }
  struct Synchronizer : public ncs::sim::DataBuffer {
    ChannelCurrentBuffer<MType>* channel_buffer;
    NeuronBuffer<MType>* neuron_buffer;
    float simulation_time;
    float time_step;
  };
  auto synchronizer_publisher = 
    new ncs::sim::SpecificPublisher<Synchronizer>();
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto blank = new Synchronizer();
    synchronizer_publisher->addBlank(blank);
  }
  auto master_function = [this, synchronizer_publisher, device]() {
    device->threadInit();
    float simulation_time = 0.0f;
    float time_step = simulation_parameters_->getTimeStep();
    while(true) {
      NeuronBuffer<MType>* neuron_buffer = neuron_subscription_->pull();
      if (nullptr == neuron_buffer) {
        delete synchronizer_publisher;
        break;
      }
      auto channel_buffer = this->getBlank();
      channel_buffer->clear();
      auto synchronizer = synchronizer_publisher->getBlank();
      synchronizer->channel_buffer = channel_buffer;
      synchronizer->neuron_buffer = neuron_buffer;
      synchronizer->simulation_time = simulation_time;
      synchronizer->time_step = time_step;
      auto prerelease_function = [this, channel_buffer, neuron_buffer]() {
        neuron_buffer->release();
        this->publish(channel_buffer);
      };
      synchronizer->setPrereleaseFunction(prerelease_function);
      synchronizer_publisher->publish(synchronizer);
      simulation_time += time_step;
    }
    device->threadDestroy();
  };
  master_thread_ = std::thread(master_function);

  for (auto simulator : simulators_) {
    auto subscription = synchronizer_publisher->subscribe();
    auto worker_function = [subscription, simulator, device]() {
      device->threadInit();
      while(true) {
        auto synchronizer = subscription->pull();
        if (nullptr == synchronizer) {
          delete subscription;
          break;
        }
        ChannelUpdateParameters parameters;
        parameters.voltage = synchronizer->neuron_buffer->getVoltage();
        parameters.current = synchronizer->channel_buffer->getCurrent();
        parameters.reversal_current = synchronizer->channel_buffer->getReversalCurrent();
        parameters.simulation_time = synchronizer->simulation_time;
        parameters.time_step = synchronizer->time_step;
        parameters.write_lock = synchronizer->channel_buffer->getWriteLock();
        if (!simulator->update(&parameters)) {
          std::cerr << "An error occurred updating a ChannelSimulator." <<
            std::endl;
        }
        synchronizer->release();
      }
      device->threadDestroy();
    };
    worker_threads_.push_back(std::thread(worker_function));
  }
  return true;
}

template<ncs::sim::DeviceType::Type MType>
ChannelUpdater<MType>::~ChannelUpdater() {
  if (master_thread_.joinable()) {
    master_thread_.join();
  }
  for (auto& thread : worker_threads_) {
    thread.join();
  }
  if (neuron_subscription_) {
    delete neuron_subscription_;
  }
}

