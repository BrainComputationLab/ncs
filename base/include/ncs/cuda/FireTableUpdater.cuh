#include <ncs/sim/Bit.h>

namespace ncs {

namespace sim {

namespace cuda {

void updateFireTable(Bit::Word* neuron_fire_vector,
                     Bit::Word* synapse_fire_table,
                     unsigned int synaptic_vector_size,
                     unsigned int row,
                     unsigned int num_rows,
                     unsigned int* presynaptic_neuron_ids,
                     unsigned int* synaptic_delays,
                     unsigned int num_synapses);

} // namespace cuda

} // namespace sim

} // namespace ncs
