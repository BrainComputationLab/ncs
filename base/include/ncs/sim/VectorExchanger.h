#pragma once
#include <ncs/sim/Bit.h>
#include <ncs/sim/DeviceNeuronStateBuffer.h>

namespace ncs {

namespace sim {

class DeviceVectorExtractorBase {
public:
  virtual bool pull(Bit::Word* dst) = 0;
  virtual ~DeviceVectorExtractorBase();
private:
};

template<DeviceType::Type MType>
class DeviceVectorExtractor : public DeviceVectorExtractorBase {
public:
  typedef SpecificPublisher<DeviceNeuronStateBuffer<MType>> StatePublisher;
  DeviceVectorExtractor();
  bool init(StatePublisher* publisher);
  virtual bool pull(Bit::Word* dst);
  virtual ~DeviceVectorExtractor();
private:
  typename StatePublisher::Subscription* state_subscription_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/VectorExchanger.hpp>
