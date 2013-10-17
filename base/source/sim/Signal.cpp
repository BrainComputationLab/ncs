#include <ncs/sim/Signal.h>

namespace ncs {

namespace sim {

Signal::Signal()
  : status_(true) {
}

bool Signal::getStatus() const {
  return status_;
}

void Signal::setStatus(bool status) {
  status_ = status;
}

Signal::~Signal() {
}

} // namespace sim

} // namespace ncs
