#include <iostream>

#include <ncs/sim/Bit.h>
#include <ncs/sim/DataType.h>

namespace ncs {

namespace sim {

size_t DataType::num_bytes(size_t count, Type t) {
  switch(t) {
    case Type::Bit:
      return ncs::sim::Bit::num_words(count) * sizeof(Bit::Word);
      break;
    case Float:
      return sizeof(float) * count;
      break;
    case Integer:
      return sizeof(int32_t) * count;
      break;
    case Unknown:
      std::cerr << "Warning: Unknown datatype size requested." << std::endl;
      return 0;
      break;
  };
}

size_t DataType::num_padded_elements(size_t count, Type t) {
  switch(t) {
    case Type::Bit:
      return ncs::sim::Bit::pad(count);
      break;
    case Float:
      return count;
      break;
    case Integer:
      return count;
      break;
    case Unknown:
      std::cerr << "Warning: Unknown datatype size requested." << std::endl;
      return 0;
      break;
  };
}

} // namespace sim

} // namespace ncs
