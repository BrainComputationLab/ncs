#include <ncs/sim/InputUpdateParameters.h>

namespace ncs {

namespace sim {

template<DeviceType::Type MType>
InputUpdater<MType>::InputUpdater()
  : step_subscription_(nullptr) {
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::
init(SpecificPublisher<StepSignal>* signal_publisher,
     size_t num_buffers,
     size_t device_neuron_vector_size,
     FactoryMap<InputSimulator>* input_plugins,
     const spec::SimulationParameters* simulation_parameters) {
  simulation_parameters_ = simulation_parameters;
  num_buffers_ = num_buffers;
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer = new InputBuffer<MType>(device_neuron_vector_size);
    if (!buffer->init()) {
      std::cerr << "Failed to initialize InputBuffer." << std::endl;
      delete buffer;
      return false;
    }
    addBlank(buffer);
  }
  step_subscription_ = signal_publisher->subscribe();
  if (nullptr == step_subscription_) {
    std::cerr << "Failed to subscribe to StepSignal generator." << std::endl;
    return false;
  }
  std::vector<std::string> input_types = input_plugins->getTypes();
  for (auto type : input_types) {
    auto simulator_generator = input_plugins->getProducer<MType>(type);
    if (!simulator_generator) {
      std::cerr << "Failed to get simulator for input type " << type <<
        std::endl;
      return false;
    }
    InputSimulator<MType>* simulator = simulator_generator();
    if (!simulator->initialize(simulation_parameters_)) {
      std::cerr << "Failed to initialize InputSimulator for type " << type <<
        std::endl;
      delete simulator;
      return false;
    }
    simulator_type_indices_[type] = simulators_.size();
    simulators_.push_back(simulator);
  }
  return true;
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::step() {
  auto step_signal = step_subscription_->pull();
  if (nullptr == step_signal) {
    return false;
  }
  auto buffer = this->getBlank();
  this->publish(buffer);
  step_signal->release();
  return true;
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::addInputs(const std::vector<Input*>& inputs,
                                    void* instantiator,
                                    const std::string& type,
                                    float start_time,
                                    float end_time) {
  auto search_result = simulator_type_indices_.find(type);
  if (simulator_type_indices_.end() == search_result) {
    std::cerr << "Failed to find an InputSimulator for type " << type <<
      std::endl;
    return false;
  }
  auto simulator = simulators_[search_result->second];
  return simulator->addInputs(inputs,
                              instantiator,
                              start_time,
                              end_time);
}

template<DeviceType::Type MType>
bool InputUpdater<MType>::start() {
  struct Synchronizer : public DataBuffer {
    InputBuffer<MType>* input_buffer;
    float simulation_time;
    float time_step;
  };
  auto synchronizer_publisher = new SpecificPublisher<Synchronizer>();
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto blank = new Synchronizer();
    synchronizer_publisher->addBlank(blank);
  }
  auto master_function = [this, synchronizer_publisher]() {
    float simulation_time = 0.0f;
    float time_step = simulation_parameters_->getTimeStep();
    while(true) {
      auto step_signal = this->step_subscription_->pull();
      if (nullptr == step_signal) {
        delete synchronizer_publisher;
        return;
      }
      auto input_buffer = this->getBlank();
      input_buffer->clear();
      auto synchronizer = synchronizer_publisher->getBlank();
      synchronizer->input_buffer = input_buffer;
      synchronizer->time_step = time_step;
      synchronizer->simulation_time = simulation_time;
      auto prerelease_function = [this, input_buffer, step_signal]() {
        this->publish(input_buffer);
        step_signal->release();
      };
      synchronizer->setPrereleaseFunction(prerelease_function);
      synchronizer_publisher->publish(synchronizer);
      simulation_time += time_step;
    }
  };
  master_thread_ = std::thread(master_function);
  
  for (auto simulator : simulators_) {
    auto subscription = synchronizer_publisher->subscribe();
    auto worker_function = [subscription, simulator]() {
      while(true) {
        auto synchronizer = subscription->pull();
        if (nullptr == synchronizer) {
          delete subscription;
          return;
        }
        InputUpdateParameters parameters;
        auto buffer = synchronizer->input_buffer;
        parameters.input_current = buffer->getInputCurrent();
        parameters.clamp_voltage_values = buffer->getVoltageClampValues();
        parameters.voltage_clamp_bits = buffer->getVoltageClampBits();
        parameters.write_lock = buffer->getWriteLock();
        parameters.simulation_time = synchronizer->simulation_time;
        parameters.time_step = synchronizer->time_step;
        if (!simulator->update(&parameters)) {
          std::cerr << "An error occurred updating an InputSimulator." <<
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
InputUpdater<MType>::~InputUpdater() {
  if (master_thread_.joinable()) {
    master_thread_.join();
  }
  for (auto& thread : worker_threads_) {
    thread.join();
  }
  if (step_subscription_) {
    delete step_subscription_;
  }
}

} // namespace sim

} // namespace ncs
