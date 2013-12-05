#pragma once

#include <thread>

#include <ncs/sim/DataBuffer.h>
#include <ncs/sim/DeviceDescription.h>
#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>
#include <ncs/sim/FireTable.h>
#include <ncs/sim/FireTableUpdater.h>
#include <ncs/sim/InputUpdater.h>
#include <ncs/sim/NeuronSimulator.h>
#include <ncs/sim/NeuronSimulatorUpdater.h>
#include <ncs/sim/PluginDescription.h>
#include <ncs/sim/ReportManager.h>
#include <ncs/sim/SynapseSimulator.h>
#include <ncs/sim/SynapseSimulatorUpdater.h>
#include <ncs/sim/VectorExchanger.h>
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
               VectorExchangeController* vector_exchange_controller,
               GlobalVectorPublisher* global_vector_publisher,
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

template<DeviceType::Type MType>
class Device : public DeviceBase {
public:
  virtual DeviceType::Type getDeviceType() const;
  virtual int getNeuronTypeIndex(const std::string& type) const;
  virtual bool 
    initialize(DeviceDescription* description,
               FactoryMap<NeuronSimulator>* neuron_plugins,
               FactoryMap<SynapseSimulator>* synapse_plugins,
               FactoryMap<InputSimulator>* input_plugins,
               VectorExchangeController* vector_exchange_controller,
               GlobalVectorPublisher* global_vector_publisher,
               size_t global_neuron_vector_size,
               size_t global_neuron_vector_offset,
               SpecificPublisher<StepSignal>* signal_publisher,
               const spec::SimulationParameters* simulation_parameters);
  virtual bool initializeReporters(int machine_location,
                                   int device_location,
                                   ReportManagers* report_managers);
  virtual bool threadInit();
  virtual bool threadDestroy();
  virtual bool start();
  virtual bool addInput(const std::vector<Input*>& inputs,
                        void* instantiator,
                        const std::string& type,
                        float start_time,
                        float end_time);
  virtual SpecificPublisher<Signal>* getVectorExtractor();
  virtual ~Device();
private:
  bool allocateUpdaters_();

  bool initializeNeurons_(DeviceDescription* description,
                          FactoryMap<NeuronSimulator>* neuron_plugins);
  bool initializeNeuronSimulator_(NeuronSimulator<MType>* simulator,
                                  NeuronPluginDescription* description);
  bool initializeNeuronUpdater_();

  bool initializeVectorExtractor_(VectorExchangeController* controller,
                                  size_t global_neuron_vector_offset);
  bool initializeVectorInjector_(GlobalVectorPublisher* publisher,
                                 size_t global_vector_size);

  bool initializeSynapses_(DeviceDescription* description,
                           FactoryMap<SynapseSimulator>* synapse_plugins);
  bool initializeSynapseSimulator_(SynapseSimulator<MType>* simulator,
                                  SynapsePluginDescription* description);
  bool initializeSynapseUpdater_();

  bool initializeFireTable_();
  bool initializeFireTableUpdater_(DeviceDescription* description);

  bool initializeInputUpdater_(SpecificPublisher<StepSignal>* signal_publisher,
                               FactoryMap<InputSimulator>* input_plugins);

  std::map<std::string, int> neuron_type_map_;
  std::vector<NeuronSimulator<MType>*> neuron_simulators_;
  std::vector<size_t> neuron_device_id_offsets_;
  size_t neuron_device_vector_size_;
  NeuronSimulatorUpdater<MType>* neuron_simulator_updater_;

  DeviceVectorExtractor<MType>* fire_vector_extractor_;
  GlobalVectorInjector<MType>* global_vector_injector_;

  FireTable<MType>* fire_table_;
  unsigned int min_synaptic_delay_;
  unsigned int max_synaptic_delay_;

  FireTableUpdater<MType>* fire_table_updater_;

  std::vector<SynapseSimulator<MType>*> synapse_simulators_;
  std::vector<size_t> synapse_device_id_offsets_;
  size_t device_synaptic_vector_size_;
  SynapseSimulatorUpdater<MType>* synapse_simulator_updater_;

  InputUpdater<MType>* input_updater_;

  const spec::SimulationParameters* simulation_parameters_;
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/Device.hpp>
