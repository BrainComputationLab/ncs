%module pyncs
%{
#include <include/ncs/sim/Simulation.h>
#include <include/ncs/spec/BoxGenerator.h>
#include <include/ncs/spec/DataSource.h>
#include <include/ncs/spec/ExactGenerator.h>
#include <include/ncs/spec/Generator.h>
#include <include/ncs/spec/InputGroup.h>
#include <include/ncs/spec/ModelSpecification.h>
#include <include/ncs/spec/NormalGenerator.h>
#include <include/ncs/spec/NullSink.h>
#include <include/ncs/spec/Report.h>
#include <include/ncs/spec/UniformGenerator.h>
%}

%include <stdint.i>
%include <std_map.i>
%include <std_string.i>
%include <std_vector.i>
%include <include/ncs/sim/Simulation.h>
%include <include/ncs/spec/BoxGenerator.h>
%include <include/ncs/spec/DataSource.h>
%include <include/ncs/spec/ExactGenerator.h>
%include <include/ncs/spec/Generator.h>
%include <include/ncs/spec/InputGroup.h>
%include <include/ncs/spec/ModelSpecification.h>
%include <include/ncs/spec/NormalGenerator.h>
%include <include/ncs/spec/NullSink.h>
%include <include/ncs/spec/Report.h>
%include <include/ncs/spec/UniformGenerator.h>

%template(string_to_generator_map) std::map<std::string, ncs::spec::Generator*>;
%{
  namespace swig {
    template<>  struct traits<ncs::spec::Generator> {
      typedef pointer_category category;
      static const char* type_name() {
        return "ncs::spec::Generator";
      }
    };
  }
%}

%template(string_to_model_parameters_map) std::map<std::string, ncs::spec::ModelParameters*>;
%{
  namespace swig {
    template<>  struct traits<ncs::spec::ModelParameters> {
      typedef pointer_category category;
      static const char* type_name() {
        return "ncs::spec::ModelParameters";
      }
    };
  }
%}

%template(neuron_group_list) std::vector<ncs::spec::NeuronGroup*>;
%template(string_to_neuron_group_map) std::map<std::string, ncs::spec::NeuronGroup*>;
%template(neuron_alias_list) std::vector<ncs::spec::NeuronAlias*>;
%template(string_to_neuron_alias_map) std::map<std::string, ncs::spec::NeuronAlias*>;
%template(synapse_group_list) std::vector<ncs::spec::SynapseGroup*>;
%template(string_to_synapse_group_map) std::map<std::string, ncs::spec::SynapseGroup*>;
%template(synapse_alias_list) std::vector<ncs::spec::SynapseAlias*>;
%template(string_to_synapse_alias_map) std::map<std::string, ncs::spec::SynapseAlias*>;
%template(string_list) std::vector<std::string>;
