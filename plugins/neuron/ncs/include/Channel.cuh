namespace cuda {

bool updateVoltageGatedIon(const unsigned int* neuron_plugin_ids,
                           const float* neuron_voltages,
                           const float* v_half,
                           const float* deactivation_scale,
                           const float* activation_scale,
                           const float* equilibrium_scale,
                           const float* tau_scale_factor,
                           const float* old_m,
                           const float* reversal_potential,
                           const float* conductance,
                           float* new_m,
                           float* channel_current,
                           float dt,
                           unsigned int num_channels);

bool updateCalciumDependent(const unsigned int* neuron_plugin_ids,
                            const float* neuron_voltages,
                            const float* neuron_calcium,
                            const float* forward_scale,
                            const float* forward_exponent,
                            const float* backwards_rate,
                            const float* tau_scale,
                            const float* m_power,
                            const float* conductance,
                            const float* reversal_potential,
                            const float* old_m,
                            float* new_m,
                            float* channel_current,
                            float dt,
                            unsigned int num_channels);

} // namespace cuda 
