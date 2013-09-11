template<ncs::sim::DeviceType::Type MemoryType>
bool IzhikevichSimulator<MemoryType>::
addNeuron(ncs::sim::Neuron* neuron) {
  Instantiator* i = (Instantiator*)(neuron->instantiator);
  ncs::spec::RNG rng(neuron->seed);
  a_.push_back(i->a->generateDouble(&rng));
  b_.push_back(i->b->generateDouble(&rng));
  c_.push_back(i->c->generateDouble(&rng));
  d_.push_back(i->d->generateDouble(&rng));
  u_.push_back(i->u->generateDouble(&rng));
  v_.push_back(i->v->generateDouble(&rng));
  return true;
}

template<ncs::sim::DeviceType::Type MemoryType>
bool IzhikevichSimulator<MemoryType>::initialize() {
}
