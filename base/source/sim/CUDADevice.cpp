#ifdef NCS_CUDA
#include <ncs/sim/CUDADevice.h>

namespace ncs {

namespace sim {

CUDADevice::CUDADevice(int device_number)
  : device_number_(device_number) {
}

bool CUDADevice::threadInit() {
  bool result = DeviceBase::setThreadDevice(this);
  result &= CUDA::setDevice(device_number_);
  result &= CUDA::initStream();
  return result;
}

bool CUDADevice::threadDestroy() {
  return CUDA::endStream();
}

} // namespace sim

} // namespace ncs
#endif
