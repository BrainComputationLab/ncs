#include <ncs/sim/Storage.h>

namespace ncs {

namespace sim {

namespace cuda {

template<typename T>
void extract(const typename Storage<T>::type* source,
             typename Storage<T>::type* destination,
             const unsigned int* indices,
             unsigned int num_indices);

} // namespace cuda

} // namespace sim

} // namespace ncs
