#pragma once
#ifdef NCS_CUDA
#include <cuda_runtime.h>

namespace ncs {

namespace sim {

class CUDA
{
public:
  static bool setDevice(int device_number);
  static int getDevice();
  static bool initStream();
  static cudaStream_t& getStream();
  static bool synchronize();
  static bool endStream();
private:
	static __thread cudaStream_t stream_;
};

} // namespace sim

} // namespace ncs

#endif // NCS_CUDA
