#include <ncs/sim/Bit.h>

namespace cuda {

bool updateNeurons(const float* as,
                   const float* bs,
                   const float* cs,
                   const float* ds,
                   const float* thresholds,
                   const float* synaptic_current,
                   const float* input_current,
                   const float* old_u,
                   const float* old_v,
                   float* new_u,
                   float* new_v,
                   ncs::sim::Bit::Word* fire_vector,
                   float step_dt,
                   unsigned int num_neurons);

} // namespace cuda
