import ncs

import synapses_pcrain

def add(sim):
  synapses_pcrain.add(sim)
  # Channel Parameters
  ahp_PCR = {
    "type": "calcium_dependent",
    "m_initial": 0.0,
    "reversal_potential": -80,
    "m_power": 2,
    "conductance": 6.0 * 0.00044,
    "forward_scale": 0.000125,
    "forward_exponent": 2,
    "backwards_rate": 2.5,
    "tau_scale": 0.01,
  }

  # spike shapes
  AP_Hoff = [-38, 30, -43, -60, -60]
  AP_OLM = [-38, 30, -43, -60, -60, -60, -60, -60, -60, -60]

  # Cell Parameters
  soma_exc_PCR = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 5.0,
    "calcium_spike_increment": 100.0,
    "tau_calcium": 0.07,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": AP_Hoff,
    "channels": [ ahp_PCR ],
  }

  soma_inhib_PCR = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 0.0,
    "calcium_spike_increment": 0.0,
    "tau_calcium": 0.0,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": AP_Hoff,
  }

  soma_probe_HP = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 0.0,
    "calcium_spike_increment": 0.0,
    "tau_calcium": 0.0,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": AP_Hoff,
  }

  soma_inhib_IP = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 0.0,
    "calcium_spike_increment": 0.0,
    "tau_calcium": 0.0,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": AP_Hoff,
  }

  soma_inhib_OLM = {
    "threshold": -50.0,
    "resting_potential": -60.0,
    "calcium": 0.0,
    "calcium_spike_increment": 0.0,
    "tau_calcium": 0.0,
    "leak_reversal_potential": 0.0,
    "tau_membrane": 0.02,
    "r_membrane": 200.0,
    "spike_shape": AP_OLM,
  }

  # Register the cell parameters
  sim.addModelParameters("soma_exc_PCR", "ncs", soma_exc_PCR)
  sim.addModelParameters("soma_inhib_PCR", "ncs", soma_exc_PCR)
  sim.addModelParameters("soma_probe_HP", "ncs", soma_probe_HP)
  sim.addModelParameters("soma_inhib_IP", "ncs", soma_inhib_IP)
  sim.addModelParameters("soma_inhib_OLM", "ncs", soma_inhib_OLM)

  # layer_SP
  for i in range(1,15):
    # excitatory
    e_name = "CA_COLUMN:layer_SP:PCR%i_E:somaE_PCR" % i
    e_group = sim.addCellGroup(e_name, 2600, "soma_exc_PCR", None)
    # inhibitory
    i_name = "CA_COLUMN:layer_SP:PCR%i_I:somaI_PCR" % i
    i_group = sim.addCellGroup(i_name, 600, "soma_inhib_PCR", None)

    # connections in layer
    sim.connect("CA_COLUMN:layer_SP:PCR%i_E:PCR%i_E" % (i, i),
                e_group,
                e_group,
                0.03,
                "synEE_PCR")
    sim.connect("CA_COLUMN:layer_SP:PCR%i_E:PCR%i_I" % (i, i),
                e_group,
                i_group,
                0.03,
                "synEI_PCR")
    sim.connect("CA_COLUMN:layer_SP:PCR%i_I:PCR%i_E" % (i, i),
                i_group,
                e_group,
                0.03,
                "synIE_PCR")
    sim.connect("CA_COLUMN:layer_SP:PCR%i_I:PCR%i_I" % (i, i),
                i_group,
                i_group,
                0.04,
                "synII_PCR")
#  sim.addCellGroup("CA_COLUMN:HP_probe:somaE", 100, "soma_probe_HP", None)

  # layer_SO
#  sim.addCellGroup("CA_COLUMN:layer_SO:O_LM:somaI_OLM", "soma_inhib_OLM", None)
