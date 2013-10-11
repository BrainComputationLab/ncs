#pragma once

namespace ncs {

namespace sim {

class Dataspace {
public:
  enum Space {
    Global = 0,
    Machine = 1,
    Device = 2,
    Plugin = 3
  };
};

} // namespace sim

} // namespace ncs
