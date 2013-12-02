#include <ncs/sim/Bit.h>

namespace cuda {

bool updateFlat(const ncs::sim::Bit::Word* synaptic_fire,
                const unsigned int* device_neuron_device_ids,
                const float* device_current,
                float* synaptic_current,
                unsigned int num_synapses);

} // namespace cuda
