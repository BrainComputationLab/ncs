namespace ncs {

namespace sim {

template<template<DeviceType::Type> class PluginType>
FactoryMap<PluginType>*
PluginLoader<PluginType>::loadPaths(const std::vector<std::string>& paths,
                                    const std::string& plugin_type) {
  FactoryMap<PluginType>* plugin_map = new FactoryMap<PluginType>(plugin_type);
  for (auto path : paths) {
    if (!loadPlugin_(path, plugin_map)) {
      std::cerr << "Failed to load " << plugin_type << " from file " <<
        path << std::endl;
      //delete plugin_map;
      //return nullptr;
    }
  }
  return plugin_map;
}

template<template<DeviceType::Type> class PluginType>
bool PluginLoader<PluginType>::
loadPlugin_(const std::string& path,
            FactoryMap<PluginType>* plugin_map) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    std::cerr << "Failed to open plugin " << path << std::endl;
    std::cerr << "Reason:" << dlerror() << std::endl;
    return false;
  }
  bool (*function)(FactoryMap<PluginType>*);
  *(void**)(&function) = dlsym(handle, "load");
  if (nullptr == function) {
    dlclose(handle);
    std::cerr << "Failed to find load function in file " << path << std::endl;
    return false;
  }
  if (!(*function)(plugin_map)) {
    std::cerr << "An error occurred loading plugin " << path << std::endl;
    dlclose(handle);
    return false;
  }
  return true;
}

} // namespace sim

} // namespace ncs
