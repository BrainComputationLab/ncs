#pragma once

#ifndef SWIG
#include <map>
#include <string>
#endif // SWIG
#include <ncs/spec/Generator.h>

namespace ncs {

namespace spec {

/**
  Contains all the parameters needed to generate some sort of model, be it
  a neuron, synapse, input, or whatever else.
*/
class ModelParameters {
public:
  /**
    Constructor.

    @param type The type of model these parameters are for.
    @param parameters Specifies how to generate each value for the model.
  */
  ModelParameters(const std::string& type,
                  std::map<std::string, Generator*> parameters);

  /**
    Returns the type of model this set of parameters is for.

    @return The type of model.
  */
  const std::string& getType() const;

  /**
    Returns the generator for a particular parameter name.

    @param parameter_name The name of the parameter.
    @return The corresponding Generator if there is one. Null otherwise.
  */
  Generator* getGenerator(const std::string& parameter_name);
private:
  /// The model type
  std::string type_;
  
  /// A mapping from parameter name to the value generator
  std::map<std::string, Generator*> parameters_;
};

} // namespace spec

} // namespace ncs
