#include <ncs/sim/AtomicWriter.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool SineSimulator<MType, IType>::
BatchStartsLater::operator()(const Batch* a, const Batch* b) const {
  return a->start_time > b->start_time;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool SineSimulator<MType, IType>::
addInputs(const std::vector<ncs::sim::Input*>& inputs,
          void* instantiator,
          float start_time,
          float end_time) {
  Instantiator* ins = (Instantiator*)instantiator;
  std::vector<float> amplitude_scale(inputs.size());
  std::vector<float> time_scale(inputs.size());
  std::vector<float> phase(inputs.size());
  std::vector<float> amplitude_shift(inputs.size());
  std::vector<unsigned int> device_neuron_ids(inputs.size());
  float duration = end_time - start_time;
  float one_over_duration = 1.0f / duration;
  for (size_t i = 0; i < inputs.size(); ++i) {
    device_neuron_ids[i] = inputs[i]->neuron_device_id;
    ncs::spec::RNG rng(inputs[i]->seed);
    amplitude_scale[i] = ins->amplitude_scale->generateDouble(&rng);
    time_scale[i] = ins->time_scale->generateDouble(&rng);
    phase[i] = ins->phase->generateDouble(&rng);
    amplitude_shift[i] = ins->amplitude_shift->generateDouble(&rng);
  }
  Batch* batch = new Batch();
  batch->count = inputs.size();
  batch->start_time = start_time;
  batch->end_time = end_time;
  bool result = true;
  result &= ncs::sim::mem::clone<MType>(batch->amplitude_scale, 
                                        amplitude_scale);
  result &= ncs::sim::mem::clone<MType>(batch->time_scale, 
                                        time_scale);
  result &= ncs::sim::mem::clone<MType>(batch->phase, 
                                        phase);
  result &= ncs::sim::mem::clone<MType>(batch->amplitude_shift, 
                                        amplitude_shift);
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
bool SineSimulator<MType, IType>::
initialize(const ncs::spec::SimulationParameters* simulation_parameters) {
  return true;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool SineSimulator<MType, IType>::
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
SineSimulator<MType, IType>::Batch::~Batch() {
  if (amplitude_scale) {
    ncs::sim::Memory<MType>::free(amplitude_scale);
  }
  if (time_scale) {
    ncs::sim::Memory<MType>::free(time_scale);
  }
  if (phase) {
    ncs::sim::Memory<MType>::free(phase);
  }
  if (amplitude_shift) {
    ncs::sim::Memory<MType>::free(amplitude_shift);
  }
  if (device_neuron_id) {
    ncs::sim::Memory<MType>::free(device_neuron_id);
  }
}
