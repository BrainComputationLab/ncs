#include <iostream>

#include <ncs/spec/Generator.h>
#include <ncs/spec/ExactGenerator.h>
#include <ncs/spec/NormalGenerator.h>
#include <ncs/spec/UniformGenerator.h>

#include "ModelParameters.pb.h"

namespace ncs {

namespace spec {

std::string Generator::generateString(RNG* rng) {
  std::cerr << name() << " cannot generate strings" << std::endl;
  std::terminate();
}

long Generator::generateInt(RNG* rng) {
  std::cerr << name() << " cannot generate integers" << std::endl;
  std::terminate();
}

double Generator::generateDouble(RNG* rng) {
  std::cerr << name() << " cannot generate doubles" << std::endl;
  std::terminate();
}

std::vector<Generator*> Generator::generateList(RNG* rng) {
  std::vector<Generator*> self;
  self.push_back(this);
  return self;
}

ModelParameters* Generator::generateParameters(RNG* rng) {
  std::cerr << name() << " cannot generate Parameters." << std::endl;
  std::terminate();
}

const std::string& Generator::name() const {
  static std::string s = "Unknown";
  return s;
}

Generator* Generator::fromProtobuf(com::Generator* gen) {
  switch(gen->distribution()) {
    case com::Generator::Exact:
      switch(gen->value_type()) {
        case com::Generator::Integer:
          return new ExactInteger(gen->exact_integer());
          break;
        case com::Generator::Double:
          return new ExactDouble(gen->exact_double());
          break;
        case com::Generator::String:
          return new ExactString(gen->exact_string());
          break;
        case com::Generator::List:
          {
            int num_elements = gen->exact_list_size();
            std::vector<Generator*> generators;
            for (int i = 0; i < num_elements; ++i) {
              com::Generator* sub_gen = gen->mutable_exact_list(i);
              generators.push_back(fromProtobuf(sub_gen));
            }
            return new ExactList(generators);
          }
          break;
        case com::Generator::Parameters:
          {
            com::ModelParameters* mp = gen->mutable_exact_parameters();
          }
          break;
        default:
          std::cerr << "Unrecognized exact value type " << 
            gen->value_type() << std::endl;
          return nullptr;
          break;
      };
      break;
    case com::Generator::Uniform:
      break;
    case com::Generator::Normal:
      break;
    default:
      std::cerr << "Unrecognized distribution type " << 
        gen->distribution() << std::endl;
      return nullptr;
      break;
  };
}

} // namespace spec

} // namespace ncs

