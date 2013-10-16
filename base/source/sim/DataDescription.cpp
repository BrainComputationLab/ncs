#include <ncs/sim/DataDescription.h>

namespace ncs {

namespace sim {

DataDescription::DataDescription(DataSpace::Space dataspace,
                                 DataType::Type datatype)
  : dataspace_(dataspace),
    datatype_(datatype) {
}

DataDescription::DataDescription(const DataDescription& source)
  : dataspace_(source.dataspace_),
    datatype_(source.datatype_) {
}

DataSpace::Space DataDescription::getDataSpace() const {
  return dataspace_;
}

DataType::Type DataDescription::getDataType() const {
  return datatype_;
}

bool DataDescription::operator==(const DataDescription& rhs) const {
  return datatype_ == rhs.datatype_ &&
    dataspace_ == rhs.dataspace_;
}

bool DataDescription::operator!=(const DataDescription& rhs) const {
  return datatype_ != rhs.datatype_ ||
    dataspace_ != rhs.dataspace_;
}

} // namespace sim

} // namespace ncs
