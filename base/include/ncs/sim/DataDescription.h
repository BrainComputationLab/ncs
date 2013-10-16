#pragma once
#include <ncs/sim/DataType.h>
#include <ncs/sim/DataSpace.h>

namespace ncs {

namespace sim {

class DataDescription {
public:
  DataDescription(DataSpace::Space dataspace,
                  DataType::Type datatype);
  DataDescription(const DataDescription& source);
  DataSpace::Space getDataSpace() const;
  DataType::Type getDataType() const;
  bool operator==(const DataDescription& rhs) const;
  bool operator!=(const DataDescription& rhs) const;
private:
  DataSpace::Space dataspace_;
  DataType::Type datatype_;
};


} // namespace sim

} // namespace ncs
