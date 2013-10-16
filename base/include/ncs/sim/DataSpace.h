#pragma once

namespace ncs {

namespace sim {

class DataSpace {
public:
  enum Space {
    Unknown = 0,
    Global = 1,
    Machine = 2,
    Device = 3,
    Plugin = 4
  };
};

} // namespace sim

} // namespace ncs
