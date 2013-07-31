#pragma once

#include <ncs/spec/GeometryGenerator.h>
#include <ncs/spec/ModelParameters.h>

namespace ncs {

namespace spec {

/**
  Represents an actual set of Neurons.
*/
class NeuronGroup {
public:
  /**
    Constructor.

    @param num_cells The number of cells in this group.
    @param model_parameters The parameters used to generate this group of cells.
    @param geometry_generator Specifies how to position each neuron.
  */
  NeuronGroup(unsigned int num_cells,
              ModelParameters* model_parameters,
              GeometryGenerator* geometry_generator);

  /**
    Returns the number of cells in this group.
    
    @return The number of cells in this group.
  */
  unsigned int getNumberOfCells() const;

  /**
    Returns the parameters used to generate cells in this group.
  */
  ModelParameters* getModelParameters();

  /**
    Returns how neurons are positioned in this group.
  */
  GeometryGenerator* getGeometryGenerator();
private:
  /// The number of cells in this group.
  unsigned int num_cells_;

  /// The parameters used to generate cells in this group.
  ModelParameters* model_parameters_;

  /// Generates positions for each neuron
  GeometryGenerator* geometry_generator_;
};

} // namespace spec

} // namespace ncs
