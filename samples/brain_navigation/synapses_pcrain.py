import ncs

def add(sim):
  # TODO(rvhoang): these values aren't yet accurate
  synEE_PCR = {
    "utilization": ncs.Normal(0.5, 0.05),
    "redistribution": 1.0,
    "last_prefire_time": 0.0,
    "last_postfire_time": 0.0,
    "tau_facilitation": 0.0,
    "tau_depression": 0.0,
    "tau_ltp": 0.015,
    "tau_ltd": 0.03,
    "A_ltp_minimum": 0.003,
    "A_ltd_minimum": 0.003,
    "max_conductance": 0.004,
    "reversal_potential": 0.0,
    "tau_postsynaptic_conductance": 0.025,
    "psg_waveform_duration": 0.05,
    "delay": ncs.Uniform(1,5),
  }
  
  synEI_PCR = {
    "utilization": ncs.Normal(0.5, 0.05),
    "redistribution": 1.0,
    "last_prefire_time": 0.0,
    "last_postfire_time": 0.0,
    "tau_facilitation": 0.0,
    "tau_depression": 0.0,
    "tau_ltp": 0.005,
    "tau_ltd": 0.005,
    "A_ltp_minimum": 0.02,
    "A_ltd_minimum": 0.01,
    "max_conductance": 0.004,
    "reversal_potential": 0.0,
    "tau_postsynaptic_conductance": 0.025,
    "psg_waveform_duration": 0.05,
    "delay": ncs.Uniform(1,5),
  }

  synIE_PCR = {
    "utilization": ncs.Normal(0.5, 0.05),
    "redistribution": 1.0,
    "last_prefire_time": 0.0,
    "last_postfire_time": 0.0,
    "tau_facilitation": 0.0,
    "tau_depression": 0.0,
    "tau_ltp": 0.005,
    "tau_ltd": 0.005,
    "A_ltp_minimum": 0.02,
    "A_ltd_minimum": 0.01,
    "max_conductance": 0.004,
    "reversal_potential": 0.0,
    "tau_postsynaptic_conductance": 0.025,
    "psg_waveform_duration": 0.05,
    "delay": ncs.Uniform(1,5),
  }
  synII_PCR = {
    "utilization": ncs.Normal(0.5, 0.05),
    "redistribution": 1.0,
    "last_prefire_time": 0.0,
    "last_postfire_time": 0.0,
    "tau_facilitation": 0.0,
    "tau_depression": 0.0,
    "tau_ltp": 0.005,
    "tau_ltd": 0.005,
    "A_ltp_minimum": 0.02,
    "A_ltd_minimum": 0.01,
    "max_conductance": 0.004,
    "reversal_potential": 0.0,
    "tau_postsynaptic_conductance": 0.025,
    "psg_waveform_duration": 0.05,
    "delay": ncs.Uniform(1,5),
  }
  sim.addModelParameters("synEE_PCR", "ncs", synEE_PCR)
  sim.addModelParameters("synEI_PCR", "ncs", synEI_PCR)
  sim.addModelParameters("synIE_PCR", "ncs", synIE_PCR)
  sim.addModelParameters("synII_PCR", "ncs", synII_PCR)
