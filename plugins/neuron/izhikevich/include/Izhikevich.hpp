#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
IzhikevichSimulator<MType>::IzhikevichSimulator() {
  buffers_ = new Buffers();
  a_ = nullptr;
  b_ = nullptr;
  c_ = nullptr;
  d_ = nullptr;
  u_ = nullptr;
  v_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool IzhikevichSimulator<MType>::
addNeuron(ncs::sim::Neuron* neuron) {
  Instantiator* i = (Instantiator*)(neuron->instantiator);
  ncs::spec::RNG rng(neuron->seed);
  buffers_->a.push_back(i->a->generateDouble(&rng));
  buffers_->b.push_back(i->b->generateDouble(&rng));
  buffers_->c.push_back(i->c->generateDouble(&rng));
  buffers_->d.push_back(i->d->generateDouble(&rng));
  buffers_->u.push_back(i->u->generateDouble(&rng));
  buffers_->v.push_back(i->v->generateDouble(&rng));
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool IzhikevichSimulator<MType>::initialize() {
  using ncs::sim::Memory;
  num_neurons_ = buffers_->a.size();
  bool result = true;
  result &= Memory<MType>::malloc(a_, num_neurons_);
  result &= Memory<MType>::malloc(b_, num_neurons_);
  result &= Memory<MType>::malloc(c_, num_neurons_);
  result &= Memory<MType>::malloc(d_, num_neurons_);
  result &= Memory<MType>::malloc(u_, num_neurons_);
  result &= Memory<MType>::malloc(v_, num_neurons_);
  if (!result) {
    std::cerr << "Memory allocation failed." << std::endl;
    return false;
  }
  const auto CPU = ncs::sim::DeviceType::CPU;
  result &= Memory<CPU>::To<MType>::copy(buffers_->a.data(), a_, num_neurons_);
  result &= Memory<CPU>::To<MType>::copy(buffers_->b.data(), b_, num_neurons_);
  result &= Memory<CPU>::To<MType>::copy(buffers_->c.data(), c_, num_neurons_);
  result &= Memory<CPU>::To<MType>::copy(buffers_->d.data(), d_, num_neurons_);
  result &= Memory<CPU>::To<MType>::copy(buffers_->u.data(), u_, num_neurons_);
  result &= Memory<CPU>::To<MType>::copy(buffers_->v.data(), v_, num_neurons_);
  if (!result) {
    std::cerr << "Failed to copy data to device." << std::endl;
    return false;
  }

  delete buffers_;
  buffers_ = nullptr;
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool IzhikevichSimulator<MType>::initializeVoltages(float* plugin_voltages) {
  using ncs::sim::Memory;
  if (!ncs::sim::mem::copy<MType, MType>(plugin_voltages, v_, num_neurons_)) {
    std::cerr << "Failed to copy voltages." << std::endl;
    return false;
  }
  return true;
}

template<ncs::sim::DeviceType::Type MType>
IzhikevichSimulator<MType>::~IzhikevichSimulator() {
  if (buffers_) {
    delete buffers_;
    buffers_ = nullptr;
  }
  ncs::sim::Memory<MType>::free(a_);
  ncs::sim::Memory<MType>::free(b_);
  ncs::sim::Memory<MType>::free(c_);
  ncs::sim::Memory<MType>::free(d_);
  ncs::sim::Memory<MType>::free(u_);
  ncs::sim::Memory<MType>::free(v_);
}
