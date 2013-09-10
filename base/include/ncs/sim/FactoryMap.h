#pragma once
#include <functional>
#include <iostream>
#include <map>

#include <ncs/sim/DeviceType.h>
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace sim {

template<template<DeviceType::Type> class Product>
class FactoryMap {
public:
  FactoryMap(const std::string& product_type);
  typedef std::function<Product<DeviceType::CUDA>*()> CUDAProducer;
  typedef std::function<Product<DeviceType::CPU>*()> CPUProducer;
  typedef std::function<Product<DeviceType::CL>*()> CLProducer;
  typedef std::function<void*(spec::ModelParameters*)> Instantiator;
  typedef std::function<void(void*)> InstantiatorDestructor;
  bool registerCUDAProducer(const std::string& type, CUDAProducer producer);
  bool registerCPUProducer(const std::string& type, CPUProducer producer);
  bool registerCLProducer(const std::string& type, CLProducer producer);
  bool registerInstantiator(const std::string& type,
                            Instantiator instantiator);
  Instantiator getInstantiator(const std::string& type);
  bool registerInstantiatorDestructor(const std::string& type,
                                      InstantiatorDestructor destructor);
  template<DeviceType::Type MType>
  std::function<Product<MType>*()> getProducer(const std::string& type);

  template<DeviceType::Type MType>
    friend class FactoryMapTypeExtractor;
private:
  template<DeviceType::Type MType>
  bool register_(const std::string& type,
                 std::function<Product<MType>*()> producer,
                 std::map<std::string,
                          std::function<Product<MType>*()>>& map);
  template<DeviceType::Type MType>
  std::map<std::string, std::function<Product<MType>*()>>& getMap_();

  std::map<std::string, CUDAProducer> cuda_factories_;
  std::map<std::string, CPUProducer> cpu_factories_;
  std::map<std::string, CLProducer> cl_factories_;
  std::map<std::string, Instantiator> instantiators_;
  std::map<std::string, InstantiatorDestructor> destructors_;
  std::string product_type_;
};

template<DeviceType::Type MType>
class FactoryMapTypeExtractor {
public:
  template<template<DeviceType::Type> class Product>
  static std::map<std::string, std::function<Product<MType>*()>>&
    extract(FactoryMap<Product>& factory);
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/FactoryMap.hpp>
