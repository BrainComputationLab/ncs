namespace ncs {

namespace sim {

template<DeviceType::Type MType>
SynapseSimulatorUpdater<MType>::SynapseSimulatorUpdater() 
  : fire_subscription_(nullptr),
    neuron_state_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool SynapseSimulatorUpdater<MType>::
setFireVectorPublisher(FireVectorPublisher* publisher) {
  if (fire_subscription_ != nullptr) {
    std::cerr << "A FireVectorPublisher was already set." << std::endl;
    return false;
  }
  fire_subscription_ = publisher->subscribe();
  return fire_subscription_ != nullptr;
}

template<DeviceType::Type MType>
bool SynapseSimulatorUpdater<MType>::
setNeuronStatePublisher(NeuronStatePublisher* publisher) {
  if (neuron_state_subscription_ != nullptr) {
    std::cerr << "A NeuronStatePublisher was already set." << std::endl;
    return false;
  }
  neuron_state_subscription_ = publisher->subscribe();
  return neuron_state_subscription_ != nullptr;
}

template<DeviceType::Type MType>
bool SynapseSimulatorUpdater<MType>::
init(const std::vector<SynapseSimulator<MType>*>& simulators,
     const std::vector<size_t>& device_synaptic_vector_offsets,
     const spec::SimulationParameters* simulation_parameters,
     size_t neuron_device_vector_size,
     size_t num_buffers) {
  if (nullptr == fire_subscription_) {
    std::cerr << "No FireVectorPublisher was set." << std::endl;
    return false;
  }
  if (nullptr == neuron_state_subscription_) {
    std::cerr << "No NeuronStatePublisher was set." << std::endl;
    return false;
  }
  num_buffers_ = num_buffers;
  neuron_device_vector_size_ = neuron_device_vector_size;
  device_synaptic_vector_offsets_ = device_synaptic_vector_offsets;
  simulators_ = simulators;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto blank = new SynapticCurrentBuffer<MType>(neuron_device_vector_size_);
    if (!blank->init()) {
      std::cerr << "Failed to initialize SynapticCurrentBuffer." << std::endl;
      delete blank;
      return false;
    }
    this->addBlank(blank);
  }
  simulation_parameters_ = simulation_parameters;
  return true;
}

template<DeviceType::Type MType>
bool SynapseSimulatorUpdater<MType>::
start(std::function<bool()> thread_init,
      std::function<bool()> thread_destroy) {
  struct Synchronizer : public DataBuffer {
    DeviceNeuronStateBuffer<MType>* neuron_state;
    SynapticFireVectorBuffer<MType>* synaptic_fire;
    SynapticCurrentBuffer<MType>* synaptic_current;
    float simulation_time;
    float time_step;
  };
  auto synchronizer_publisher = new SpecificPublisher<Synchronizer>();
  for (size_t i = 0; i < num_buffers_; ++i) {
    synchronizer_publisher->addBlank(new Synchronizer());
  }
  auto master_function = [this,
                          synchronizer_publisher,
                          thread_init,
                          thread_destroy]() {
    thread_init();
    float simulation_time = 0.0f;
    float time_step = simulation_parameters_->getTimeStep();
    unsigned int simulation_step = 0;
    Mailbox mailbox;
    while(true) {
      DeviceNeuronStateBuffer<MType>* neuron_state = nullptr;
      neuron_state_subscription_->pull(&neuron_state, &mailbox);
      SynapticFireVectorBuffer<MType>* synaptic_fire = nullptr;
      fire_subscription_->pull(&synaptic_fire, &mailbox);
      if (!mailbox.wait(&neuron_state, &synaptic_fire)) {
        neuron_state_subscription_->cancel();
        fire_subscription_->cancel();
        if (neuron_state) {
          neuron_state->release();
        }
        if (synaptic_fire) {
          synaptic_fire->release();
        }
        delete synchronizer_publisher;
        break;
      }
      auto synchronizer = synchronizer_publisher->getBlank();
      auto synaptic_current = this->getBlank();
      synaptic_current->simulation_step = simulation_step;
      if (!synaptic_current->clear()) {
        std::cerr << "Failed to clear SynapticCurrentBuffer." <<
          std::endl;
      }
      synchronizer->neuron_state = neuron_state;
      synchronizer->synaptic_fire = synaptic_fire;
      synchronizer->synaptic_current = synaptic_current;
      synchronizer->time_step = time_step;
      synchronizer->simulation_time = simulation_time;
      auto prerelease_function = [this, synchronizer]() {
        this->publish(synchronizer->synaptic_current);
        synchronizer->neuron_state->release();
        synchronizer->synaptic_fire->release();
      };
      synchronizer->setPrereleaseFunction(prerelease_function);
      synchronizer_publisher->publish(synchronizer);
      simulation_time += time_step;
      ++simulation_step;
    }
    thread_destroy();
  };
  master_thread_ = std::thread(master_function);

  for (size_t i = 0; i < simulators_.size(); ++i) {
    auto simulator = simulators_[i];
    auto unit_offset = device_synaptic_vector_offsets_[i];
    auto subscription = synchronizer_publisher->subscribe();
    auto worker_function = [subscription, 
                            simulator, 
                            unit_offset,
                            thread_init,
                            thread_destroy]() {
      thread_init();
      auto word_offset = Bit::num_words(unit_offset);
      while (true) {
        auto synchronizer = subscription->pull();
        if (nullptr == synchronizer) {
          delete subscription;
          break;
        }
        SynapseUpdateParameters parameters;
        parameters.synaptic_fire =
          synchronizer->synaptic_fire->getFireBits() + word_offset;
        parameters.neuron_voltage =
          synchronizer->neuron_state->getVoltages();
        parameters.device_neuron_fire =
          synchronizer->neuron_state->getFireBits();
        parameters.synaptic_current =
          synchronizer->synaptic_current->getCurrents();
        parameters.write_lock =
          synchronizer->synaptic_current->getWriteLock();
        parameters.simulation_time = synchronizer->simulation_time;
        parameters.time_step = synchronizer->time_step;
        if (!simulator->update(&parameters)) {
          std::cerr << "An error occurred updating a SynapseSimulator." <<
            std::endl;
        }
        synchronizer->release();
      }
      thread_destroy();
    };
    worker_threads_.push_back(std::thread(worker_function));
  }
  return true;
}

template<DeviceType::Type MType>
SynapseSimulatorUpdater<MType>::~SynapseSimulatorUpdater() {
  if (master_thread_.joinable()){
    master_thread_.join();
  }
  for (auto& thread : worker_threads_) {
    thread.join();
  }
  if (fire_subscription_) {
    delete fire_subscription_;
  }
  if (neuron_state_subscription_) {
    delete neuron_state_subscription_;
  }
}

} // namespace sim

} // namespace ncs
