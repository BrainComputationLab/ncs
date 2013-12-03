#include <ncs/sim/AtomicWriter.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool LinearSimulator<MType, IType>::
BatchStartsLater::operator()(const Batch* a, const Batch* b) const {
  return a->start_time > b->start_time;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool LinearSimulator<MType, IType>::
addInputs(const std::vector<ncs::sim::Input*>& inputs,
          void* instantiator,
          float start_time,
          float end_time) {
  Instantiator* i = (Instantiator*)instantiator;
  auto starting_amplitude_generator = i->starting_amplitude;
  auto ending_amplitude_generator = i->ending_amplitude;
  std::vector<float> starting_amplitudes;
  std::vector<float> slopes;
  std::vector<unsigned int> device_neuron_ids;
  float duration = end_time - start_time;
  float one_over_duration = 1.0f / duration;
  for (auto input : inputs) {
    device_neuron_ids.push_back(input->neuron_device_id);
    ncs::spec::RNG rng(input->seed);
    float starting_amplitude =
      starting_amplitude_generator->generateDouble(&rng);
    starting_amplitudes.push_back(starting_amplitude);
    float ending_amplitude =
      ending_amplitude_generator->generateDouble(&rng);
    float slope = (ending_amplitude - starting_amplitude) * one_over_duration;
    slopes.push_back(slope);
  }
  Batch* batch = new Batch();
  batch->count = starting_amplitudes.size();
  batch->start_time = start_time;
  batch->end_time = end_time;
  bool result = true;
  result &= ncs::sim::mem::clone<MType>(batch->starting_amplitude, 
                                        starting_amplitudes);
  result &= ncs::sim::mem::clone<MType>(batch->slope, 
                                        slopes);
  result &= ncs::sim::mem::clone<MType>(batch->device_neuron_id,
                                        device_neuron_ids);
  if (!result) {
    delete batch;
    std::cerr << "Failed to allocate memory for input." << std::endl;
    return false;
  }
  future_batches_.push(batch);
  return true;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool LinearSimulator<MType, IType>::
initialize(const ncs::spec::SimulationParameters* simulation_parameters) {
  return true;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool LinearSimulator<MType, IType>::
update(ncs::sim::InputUpdateParameters* parameters) {
  const auto simulation_time = parameters->simulation_time;
  auto BatchIsDone = [simulation_time](Batch* b) {
    return b->end_time < simulation_time;
  };
  for (auto batch : active_batches_) {
    if (BatchIsDone(batch)) {
      delete batch;
    }
  }
  active_batches_.remove_if(BatchIsDone);

  while (!future_batches_.empty() &&
         future_batches_.top()->start_time <= simulation_time) {
    Batch* batch = future_batches_.top();
    future_batches_.pop();
    active_batches_.push_back(batch);
  }

  return update_(parameters);
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
LinearSimulator<MType, IType>::Batch::~Batch() {
  if (starting_amplitude) {
    ncs::sim::Memory<MType>::free(starting_amplitude);
  }
  if (slope) {
    ncs::sim::Memory<MType>::free(slope);
  }
  if (device_neuron_id) {
    ncs::sim::Memory<MType>::free(device_neuron_id);
  }
}
