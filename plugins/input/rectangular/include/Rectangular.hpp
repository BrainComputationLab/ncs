#include <ncs/sim/AtomicWriter.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool RectangularSimulator<MType, IType>::
BatchStartsLater::operator()(const Batch* a, const Batch* b) const {
  return a->start_time > b->start_time;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool RectangularSimulator<MType, IType>::
addInputs(const std::vector<ncs::sim::Input*>& inputs,
          void* instantiator,
          float start_time,
          float end_time) {
  Instantiator* i = (Instantiator*)instantiator;
  auto amplitude_generator = i->amplitude;
  std::vector<float> amplitudes;
  std::vector<unsigned int> device_neuron_ids;
  for (auto input : inputs) {
    device_neuron_ids.push_back(input->neuron_device_id);
    ncs::spec::RNG rng(input->seed);
    amplitudes.push_back(amplitude_generator->generateDouble(&rng));
  }
  Batch* batch = new Batch();
  batch->start_time = start_time;
  batch->end_time = end_time;
  bool result = true;
  result &= ncs::sim::mem::clone<MType>(batch->amplitude, amplitudes);
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
bool RectangularSimulator<MType, IType>::initialize() {
  return true;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
bool RectangularSimulator<MType, IType>::
update(ncs::sim::InputUpdateParameters* parameters) {
  // TODO(rvhoang): do update
  return true;
}

template<ncs::sim::DeviceType::Type MType, InputType IType>
RectangularSimulator<MType, IType>::Batch::~Batch() {
  if (amplitude) {
    ncs::sim::Memory<MType>::free(amplitude);
  }
  if (device_neuron_id) {
    ncs::sim::Memory<MType>::free(device_neuron_id);
  }
}
