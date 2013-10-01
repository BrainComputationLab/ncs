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

  // Initialize the first voltage vector
  auto buffer = this->getBlank();
  for (size_t i = 0; i < neuron_simulators_.size(); ++i) {
    auto simulator = neuron_simulators_[i];
    auto offset = device_id_offsets_[i];
    if (!simulator->initializeVoltages(buffer->getVoltages() + offset)) {
      std::cerr << "Failed to initialize voltages." << std::endl;
      buffer->release();
      return false;
    }
  }
  this->publish(buffer);
  return true;
}

template<DeviceType::Type MType>
bool NeuronSimulatorUpdater<MType>::start() {
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
  auto master_function = [this, synchronizer_publisher]() {
    Mailbox mailbox;
    while(true) {
      auto synchronizer = synchronizer_publisher->getBlank();
      std::unique_lock<std::mutex> lock(mailbox.mutex);
      input_subscription_->pull(&(synchronizer->input), &mailbox);
      neuron_state_subscription_->pull(&(synchronizer->previous_neuron_state),
                                       &mailbox);
      while((!synchronizer->input ||
             !synchronizer->previous_neuron_state) &&
            !mailbox.failed) {
        mailbox.arrival.wait(lock);
      }
      lock.unlock();
      if (mailbox.failed) {
        input_subscription_->cancel();
        neuron_state_subscription_->cancel();
        if (synchronizer->input) {
          synchronizer->input->release();
        }
        if (synchronizer->previous_neuron_state) {
          synchronizer->previous_neuron_state->release();
        }
        delete synchronizer;
        return;
      }
      auto current_neuron_state = this->getBlank();
      synchronizer->current_neuron_state = current_neuron_state;
      auto prerelease_function = [this, synchronizer]() {
        this->publish(synchronizer->current_neuron_state);
        synchronizer->input->release();
        synchronizer->previous_neuron_state->release();
      };
      synchronizer->setPrereleaseFunction(prerelease_function);
      synchronizer_publisher->publish(synchronizer);
    }
  };
  master_thread_ = std::thread(master_function);

  // TODO(rvhoang): setup workers
  for (size_t i = 0; i < neuron_simulators_.size(); ++i) {
    auto simulator = neuron_simulators_[i];
    auto unit_offset = device_id_offsets_[i];
    auto subscription = synchronizer_publisher->subscribe();
    auto worker_function = [subscription, simulator, unit_offset]() {
      while(true) {
        auto synchronizer = subscription->pull();
        auto word_offset = Bit::num_words(unit_offset);
        if (nullptr == synchronizer) {
          delete subscription;
          return;
        }
        NeuronUpdateParameters parameters;
        auto input = synchronizer->input;
        parameters.input_current = input->getInputCurrent();
        parameters.clamp_voltage_values = input->getVoltageClampValues();
        parameters.voltage_clamp_bits = input->getVoltageClampBits();
        auto previous_neuron_state = synchronizer->previous_neuron_state;
        parameters.previous_neuron_voltage = 
          previous_neuron_state->getVoltages();
        // TODO(rvhoang): add synaptic current as the cycle is completed
        auto current_neuron_state = synchronizer->current_neuron_state;
        parameters.neuron_voltage = current_neuron_state->getVoltages();
        parameters.neuron_fire_bits = current_neuron_state->getFireBits();
        parameters.write_lock = current_neuron_state->getWriteLock();
        if (!simulator->update(&parameters)) {
          std::cerr << "An error occurred updating a NeuronSimulator." <<
            std::endl;
        }
        synchronizer->release();
      }
    };
    worker_threads_.push_back(std::thread(worker_function));
  }
  return true;
}


template<DeviceType::Type MType>
NeuronSimulatorUpdater<MType>::~NeuronSimulatorUpdater() {
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
