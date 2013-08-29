#pragma once
#include <dlfcn.h>

#include <ncs/sim/DeviceType.h>
#include <ncs/sim/FactoryMap.h>

namespace ncs {

namespace sim {

template<template<DeviceType::Type> class PluginType>
class PluginLoader {
public:
  static FactoryMap<PluginType>*
    loadPaths(const std::vector<std::string>& paths,
              const std::string& plugin_type);
private:
  static bool loadPlugin_(const std::string& path,
                          FactoryMap<PluginType>* plugin_map);
};

} // namespace sim

} // namespace ncs

#include <ncs/sim/PluginLoader.hpp>
