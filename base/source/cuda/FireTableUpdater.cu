#include <ncs/cuda/CUDA.h>
#include <ncs/cuda/FireTableUpdater.cuh>
#include <ncs/sim/CUDA.h>

namespace ncs {

namespace sim {

namespace cuda {

__global__  void updateTableKernel(ncs::sim::Bit::Word* neuron_fire_vector,
                                   ncs::sim::Bit::Word* synapse_fire_table,
                                   unsigned int synaptic_vector_size,
                                   unsigned int row,
                                   unsigned int num_rows,
                                   unsigned int* presynaptic_neuron_ids,
                                   unsigned int* synaptic_delays,
                                   unsigned int num_synapses) {
	unsigned int index = grid::thread();
	unsigned int stride = grid::stride();
	for (; index < num_synapses; index += stride) {
		unsigned int pre_id = presynaptic_neuron_ids[index];
		if (pre_id == 0xFFFFFFFF) {
			continue;
    }
		unsigned int word = bit::word(pre_id);
		unsigned int mask = bit::mask(pre_id);
		if (neuron_fire_vector[word] & mask) {
			unsigned int delay = synaptic_delays[index];
			unsigned int event_row = row + delay;
			if (event_row >= num_rows) {
				event_row -= num_rows;
      }
			unsigned int event_mask = bit::mask(index);
			unsigned int* event_word =
			    synapse_fire_table + event_row * synaptic_vector_size;
			atomicOr(event_word + bit::word(index), event_mask);
		}
	}
}

void updateFireTable( Bit::Word* neuron_fire_vector,
                     Bit::Word* synapse_fire_table,
                     unsigned int synaptic_vector_size,
                     unsigned int row,
                     unsigned int num_rows,
                     unsigned int* presynaptic_neuron_ids,
                     unsigned int* synaptic_delays,
                     unsigned int num_synapses) {
	updateTableKernel<<<CUDA::getThreadsPerBlock(num_synapses), 
	                    CUDA::getNumberOfBlocks(num_synapses),
                      0,
                      CUDA::getStream()>>>(neuron_fire_vector,
                                           synapse_fire_table,
                                           synaptic_vector_size,
                                           row,
                                           num_rows,
                                           presynaptic_neuron_ids,
                                           synaptic_delays,
                                           num_synapses);
	CUDA::synchronize();
}

} // namespace cuda

} // namespace sim

} // namespace ncs
