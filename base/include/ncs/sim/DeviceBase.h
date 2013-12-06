#pragma once
#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DeviceDescription.h>
#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/InputUpdater.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/NeuronSimulatorUpdater.h>
#include <ncs/sim/PluginDescription.h>
#include <ncs/sim/ReportManager.h>
#include <ncs/sim/Signal.h>
#include <ncs/sim/SynapseSimulator.h>
#include <ncs/sim/SynapseSimulatorUpdater.h>
#include <ncs/spec/SimulationParameters.h>

namespace ncs {

namespace sim {

class DeviceBase {
public:
  virtual DeviceType::Type getDeviceType() const = 0;
  virtual int getNeuronTypeIndex(const std::string& type) const = 0;
  virtual bool 
    initialize(DeviceDescription* description,
               FactoryMap<NeuronSimulator>* neuron_plugins,
               FactoryMap<SynapseSimulator>* synapse_plugins,
               FactoryMap<InputSimulator>* input_plugins,
               class VectorExchangeController* vector_exchange_controller,
               class GlobalVectorPublisher* global_vector_publisher,
               size_t global_neuron_vector_size,
               size_t global_neuron_vector_offset,
               SpecificPublisher<StepSignal>* signal_publisher,
               const spec::SimulationParameters* simulation_parameters) = 0;
  virtual bool initializeReporters(int machine_location,
                                   int device_location,
                                   ReportManagers* report_managers) = 0;
  virtual bool threadInit() = 0;
  virtual bool threadDestroy() = 0;
  virtual bool start() = 0;
  virtual bool addInput(const std::vector<Input*>& inputs,
                        void* instantiator,
                        const std::string& type,
                        float start_time,
                        float end_time) = 0;
  virtual SpecificPublisher<Signal>* getVectorExtractor() = 0;
  virtual ~DeviceBase() = 0;
  static bool setThreadDevice(DeviceBase* device);
  static DeviceBase* getThreadDevice();
private:
  static __thread DeviceBase* thread_device_;
};

} // namespace sim

} // namespace ncs
