#pragma once
namespace ncs {

namespace sim {

class Location {
public:
  Location();
  Location(int m, int d, int p);
  bool operator==(const Location& r) const;
  bool operator<(const Location& r) const;
  int machine;
  int device;
  int plugin;
};

} // namespace sim

} // namespace ncs
