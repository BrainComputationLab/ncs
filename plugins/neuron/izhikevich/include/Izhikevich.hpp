#include <ncs/sim/Constants.h>
#include <ncs/sim/Memory.h>

template<ncs::sim::DeviceType::Type MType>
IzhikevichBuffer<MType>::IzhikevichBuffer() {
  u_ = nullptr;
}

template<ncs::sim::DeviceType::Type MType>
bool IzhikevichBuffer<MType>::init(size_t num_neurons) {
  if (0 == num_neurons) {
    return true;
  }
  bool result = true;
  result &= ncs::sim::Memory<MType>::malloc(u_, num_neurons);
  return result;
}

template<ncs::sim::DeviceType::Type MType>
float* IzhikevichBuffer<MType>::getU() {
  return u_;
}

template<ncs::sim::DeviceType::Type MType>
IzhikevichBuffer<MType>::~IzhikevichBuffer() {
  if (u_) {
    ncs::sim::Memory<MType>::free(u_);
  }
}

template<ncs::sim::DeviceType::Type MType>
IzhikevichSimulator<MType>::IzhikevichSimulator() {
  subscription_ = nullptr;
  buffers_ = new Buffers();
  a_ = nullptr;
  b_ = nullptr;
  c_ = nullptr;
  d_ = nullptr;
  v_ = nullptr;
  threshold_ = nullptr;
  step_dt_ = 0.5f;
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
  buffers_->threshold.push_back(i->threshold->generateDouble(&rng));
  return true;
}

template<ncs::sim::DeviceType::Type MType>
bool IzhikevichSimulator<MType>::
initialize(const ncs::spec::SimulationParameters* simulation_parameters) {
  using ncs::sim::Memory;
  num_neurons_ = buffers_->a.size();
  bool result = true;
  result &= ncs::sim::mem::clone<MType>(a_, buffers_->a);
  result &= ncs::sim::mem::clone<MType>(b_, buffers_->b);
  result &= ncs::sim::mem::clone<MType>(c_, buffers_->c);
  result &= ncs::sim::mem::clone<MType>(d_, buffers_->d);
  result &= ncs::sim::mem::clone<MType>(v_, buffers_->v);
  result &= ncs::sim::mem::clone<MType>(threshold_, buffers_->threshold);
  if (!result) {
    std::cerr << "Failed to copy data to device." << std::endl;
    return false;
  }

  for (size_t i = 0; i < ncs::sim::Constants::num_buffers; ++i) {
    auto blank = new IzhikevichBuffer<MType>();
    if (!blank->init(num_neurons_)) {
      std::cerr << "Faield to initialize IzhikevichBuffer." << std::endl;
      delete blank;
      return false;
    }
    addBlank(blank);
  }
  subscription_ = this->subscribe();
  auto blank = this->getBlank();
  using ncs::sim::mem::copy;
  if (!copy<MType, ncs::sim::DeviceType::CPU>(blank->getU(),
                                              buffers_->u.data(),
                                              buffers_->u.size())) {
    std::cerr << "Failed to transfer U." << std::endl;
    this->publish(blank);
    return false;
  }
  this->publish(blank);
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
bool IzhikevichSimulator<MType>::
update(ncs::sim::NeuronUpdateParameters* parameters) {
  std::cerr << "IzhikevichSimulator<MType>::update is not implemented." <<
    std::endl;
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
  ncs::sim::Memory<MType>::free(v_);
  if (subscription_) {
    delete subscription_;
  }
}
