namespace ncs {

namespace sim {

template<DeviceType::Type MType>
NeuronSimulatorUpdater<MType>::NeuronSimulatorUpdater()
  : neuron_state_subscription_(nullptr),
    input_subscription_(nullptr),
    synaptic_current_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool NeuronSimulatorUpdater<MType>::
init(InputPublisher* input_publisher,
     SynapticCurrentPublisher* synaptic_current_publisher,
     const std::vector<NeuronSimulator<MType>*>& neuron_simulators,
     const std::vector<size_t>& device_id_offsets,
     size_t neuron_device_vector_size,
     size_t num_buffers) {
  num_buffers_ = num_buffers;
  neuron_simulators_ = neuron_simulators;
  device_id_offsets_ = device_id_offsets;
  neuron_device_vector_size_ = neuron_device_vector_size;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer = 
      new DeviceNeuronStateBuffer<MType>(neuron_device_vector_size_);
    if (!buffer->isValid()) {
      std::cerr << "Failed to initialize DeviceNeuronStateBuffer." <<
        std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  input_subscription_ = input_publisher->subscribe();
  synaptic_current_subscription_ = synaptic_current_publisher->subscribe();
  neuron_state_subscription_ = this->subscribe();

  return true;
}

template<DeviceType::Type MType>
bool NeuronSimulatorUpdater<MType>::
start(std::function<bool()> thread_init,
      std::function<bool()> thread_destroy) {
  thread_init();
  // Initialize the first voltage vector
  auto buffer = this->getBlank();
  if (!Memory<MType>::zero(buffer->getFireBits(),
                           Bit::num_words(buffer->getVectorSize()))) {
    std::cerr << "Failed to zero initial DeviceNeuronStateBuffer." <<
      std::endl;
    return false;
  }

  for (size_t i = 0; i < neuron_simulators_.size(); ++i) {
    auto simulator = neuron_simulators_[i];
    auto offset = device_id_offsets_[i];
    if (!simulator->initializeVoltages(buffer->getVoltages() + offset)) {
      std::cerr << "Failed to initialize voltages." << std::endl;
      buffer->release();
      return false;
    }
  }
  buffer->simulation_step = 0;
  this->publish(buffer);
  thread_destroy();

  struct Synchronizer : public DataBuffer {
    DeviceNeuronStateBuffer<MType>* previous_neuron_state;
    SynapticCurrentBuffer<MType>* synaptic_current;
    InputBuffer<MType>* input;
    DeviceNeuronStateBuffer<MType>* current_neuron_state;
  };
  auto synchronizer_publisher = new SpecificPublisher<Synchronizer>();
  for (size_t i = 0; i < num_buffers_; ++i) {
    synchronizer_publisher->addBlank(new Synchronizer());
  }
  NeuronSimulatorUpdater<MType>* self = this;
  auto master_function = [self, 
                          synchronizer_publisher,
                          thread_init,
                          thread_destroy]() {
    thread_init();
    Mailbox mailbox;
    unsigned int simulation_step = 1;
    while(true) {
      InputBuffer<MType>* input_buffer = nullptr;
      self->input_subscription_->pull(&input_buffer, &mailbox);
      DeviceNeuronStateBuffer<MType>* previous_state_buffer = nullptr;
      self->neuron_state_subscription_->pull(&previous_state_buffer, &mailbox);
      SynapticCurrentBuffer<MType>* synaptic_current_buffer = nullptr;
      self->synaptic_current_subscription_->pull(&synaptic_current_buffer,
                                                 &mailbox);
      if (!mailbox.wait(&input_buffer, 
                        &previous_state_buffer, 
                        &synaptic_current_buffer)) {
        self->input_subscription_->cancel();
        self->neuron_state_subscription_->cancel();
        self->synaptic_current_subscription_->cancel();
        if (input_buffer) {
          input_buffer->release();
        }
        if (previous_state_buffer) {
          previous_state_buffer->release();
        }
        if (synaptic_current_buffer) {
          synaptic_current_buffer->release();
        }
        delete synchronizer_publisher;
        break;
      }
      auto synchronizer = synchronizer_publisher->getBlank();
      auto current_neuron_state = self->getBlank();
      current_neuron_state->simulation_step = simulation_step;
      synchronizer->current_neuron_state = current_neuron_state;
      synchronizer->input = input_buffer;
      synchronizer->previous_neuron_state = previous_state_buffer;
      synchronizer->synaptic_current = synaptic_current_buffer;
      auto prerelease_function = [self, synchronizer]() {
        self->publish(synchronizer->current_neuron_state);
        synchronizer->input->release();
        synchronizer->previous_neuron_state->release();
        synchronizer->synaptic_current->release();
      };
      synchronizer->setPrereleaseFunction(prerelease_function);
      synchronizer_publisher->publish(synchronizer);
      ++simulation_step;
    }
    thread_destroy();
  };
  master_thread_ = std::thread(master_function);

  for (size_t i = 0; i < neuron_simulators_.size(); ++i) {
    auto simulator = neuron_simulators_[i];
    auto unit_offset = device_id_offsets_[i];
    auto subscription = synchronizer_publisher->subscribe();
    auto worker_function = [subscription,
                            simulator, 
                            unit_offset,
                            thread_init,
                            thread_destroy]() {
      thread_init();
      while(true) {
        auto synchronizer = subscription->pull();
        auto word_offset = Bit::num_words(unit_offset);
        if (nullptr == synchronizer) {
          delete subscription;
          break;
        }
        NeuronUpdateParameters parameters;
        parameters.input_current = 
          synchronizer->input->getInputCurrent() + unit_offset;
        parameters.clamp_voltage_values = 
          synchronizer->input->getVoltageClampValues() + unit_offset;
        parameters.voltage_clamp_bits = 
          synchronizer->input->getVoltageClampBits() + word_offset;
        parameters.previous_neuron_voltage = 
          synchronizer->previous_neuron_state->getVoltages() + unit_offset;
        parameters.synaptic_current = 
          synchronizer->synaptic_current->getCurrents() + unit_offset;
        parameters.neuron_voltage = 
          synchronizer->current_neuron_state->getVoltages() + unit_offset;
        parameters.neuron_fire_bits = 
          synchronizer->current_neuron_state->getFireBits() + word_offset;
        parameters.write_lock = 
          synchronizer->current_neuron_state->getWriteLock();
        if (!simulator->update(&parameters)) {
          std::cerr << "An error occurred updating a NeuronSimulator." <<
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
NeuronSimulatorUpdater<MType>::~NeuronSimulatorUpdater() {
  if (master_thread_.joinable()) {
    master_thread_.join();
  }
  for (auto& thread : worker_threads_) {
    thread.join();
  }
  if (neuron_state_subscription_) {
    delete neuron_state_subscription_;
  }
  if (input_subscription_) {
    delete input_subscription_;
  }
  if (synaptic_current_subscription_) {
    delete synaptic_current_subscription_;
  }
}

} // namespace sim

} // namespace ncs
