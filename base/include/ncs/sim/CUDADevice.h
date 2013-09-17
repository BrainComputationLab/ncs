#pragma once
#include <ncs/sim/Device.h>

namespace ncs {

namespace sim {

class CUDADevice : public Device<DeviceType::CUDA> {
public:
  CUDADevice(int device_number);
  virtual bool threadInit();
  virtual bool threadDestroy();
private:
  int device_number_;
};

} // namespace sim

} // namespace ncs
