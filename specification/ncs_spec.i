%module ncs_spec
%{
#include <Generator.h> 
#include <ExactGenerator.h>
#include <Specification.h>
typedef slug::spec::Specification Specification;
#include <NeuronSpecification.h>
#include <SynapseSpecification.h>
%}

%include <stdint.i>
%include <std_map.i>
%include <std_string.i>
%include <Generator.h>
%include <ExactGenerator.h>
%include <Specification.h>
%template(Specification) std::map<std::string, slug::spec::Generator*>;
typedef slug::spec::Specification Specification;
%include <NeuronSpecification.h>
#include <SynapseSpecification.h>
%{
  namespace swig {
    template<>  struct traits<slug::spec::Generator> {
      typedef pointer_category category;
      static const char* type_name() {
        return "slug::spec::Generator";
      }
    };
  }
%}
