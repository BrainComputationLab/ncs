#include <ncs/spec/NeuronGroup.h>

namespace ncs {

namespace spec {

NeuronGroup::NeuronGroup(unsigned int num_cells,
                         ModelParameters* model_parameters,
                         GeometryGenerator* geometry_generator)
  : num_cells_(num_cells),
    model_parameters_(model_parameters),
    geometry_generator_(geometry_generator) {
}

unsigned int NeuronGroup::getNumberOfCells() const {
  return num_cells_;
}

ModelParameters* NeuronGroup::getModelParameters() {
  return model_parameters_;
}

GeometryGenerator* NeuronGroup::getGeometryGenerator() {
  return geometry_generator_;
}

} // namespace spec

} // namespace ncs
