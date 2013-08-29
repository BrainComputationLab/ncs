#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

#include <ncs/sim/DeviceDescription.h>

namespace ncs {

namespace sim {

DeviceDescription::
DeviceDescription(bool on_this_machine, double power, DeviceType::Type device_type)
  : on_this_machine_(on_this_machine),
    power_(power),
    device_type_(device_type) {
}

bool DeviceDescription::isOnThisMachine() const {
  return on_this_machine_;
}

double DeviceDescription::getPower() const {
  return power_;
}

DeviceType::Type DeviceDescription::getDeviceType() const {
  return device_type_;
}

const std::vector<NeuronPluginDescription*>&
DeviceDescription::getNeuronPlugins() const {
  return neuron_plugins_;
}

unsigned int DeviceDescription::getNeuronPluginIndex(const std::string& type) {
  if (neuron_type_to_plugin_index_.count(type) == 0) {
    neuron_plugins_.push_back(new NeuronPluginDescription(type));
    neuron_type_to_plugin_index_[type] = neuron_plugins_.size() - 1;
  }
  return neuron_type_to_plugin_index_[type];
}

NeuronPluginDescription*
DeviceDescription::getNeuronPlugin(const std::string& type) {
  unsigned int index = getNeuronPluginIndex(type);
  return neuron_plugins_[index];
}

std::vector<DeviceDescription*> 
DeviceDescription::getDevicesOnThisMachine(unsigned int enabled_device_types) {
  std::vector<DeviceDescription*> results;
  // Get CPU devices
  if (enabled_device_types & DeviceType::CPU) {
    // TODO(rvhoang): Find a reasonable estimation of the number of cores and
    // clock rate per CPU
    unsigned int num_cores = std::thread::hardware_concurrency();
    // For now, use 2MHz per core
    double power = num_cores * 2000000.0;
    results.push_back(new DeviceDescription(true, power, DeviceType::CPU));
  }

  // Get CUDA devices
  if (enabled_device_types & DeviceType::CUDA) {
#ifdef NCS_CUDA
    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    for (int i = 0; i < num_cuda_devices; ++i) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, i);
      double power = properties.clockRate * properties.multiProcessorCount;
      if (properties.major >= 3) { // Each SMX is 192 cores
        power *= 192;
      }
      else { // Each SP is 32 cores
        power *= 32;
      }
      results.push_back(new DeviceDescription(true, power, DeviceType::CUDA));
    }
#else // NCS_CUDA
    std::cerr << "This version of NCS was not compiled with CUDA support." <<
      " Skipping CUDA devices." << std::endl;
#endif // NCS_CUDA
  }

  // TODO(rvhoang): Get CL devices
  if (enabled_device_types & DeviceType::CL) {
    std::cerr << "OpenCL is not yet implemented. Skipping CL devices." <<
      std::endl;
  }
  
  return results;
}

} // namespace sim

} // namespace ncs
