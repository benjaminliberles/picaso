import streamlit as st
import os
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_bokeh import streamlit_bokeh

import picaso.driver as go
from picaso import justplotit as jpi
from  picaso import retrieval as ret

# HEADER
st.logo('https://natashabatalha.github.io/picaso/_images/logo.png', size="large", link="https://github.com/natashabatalha/picaso")
st.header('Retrieval Analysis', divider='rainbow')

uploaded_file = st.file_uploader("Upload driver configuration TOML", type=["toml"])

if uploaded_file is not None:
    # Read the file as a string
    try:
        string_data = uploaded_file.read().decode("utf-8")
        config = toml.loads(string_data)
    except: 
        st.error("Could not prase retrieval toml file")
        st.stop()
    
    st.success(f"Successfully loaded configuration file!")
    
    # Check retrieval output
    ret_out = config.get('InputOutput', {}).get('retrieval_output', '')
    if not ret_out:
        st.warning("The uploaded config does not have 'retrieval_output' defined in 'InputOutput'. Please specify the directory path below:")
    
    retrieval_dir = st.text_input("Retrieval Output Directory", value=ret_out)
    
    if retrieval_dir:
        if not os.path.exists(retrieval_dir):
            st.error(f"The path '{retrieval_dir}' does not exist on this machine.")
        else:
            # Get parameters
            ret_section = config.get('retrieval', {})
            if not ret_section:
                st.error("No 'retrieval' section found in the configuration.")
            else:
                fitpars = go.prior_finder(ret_section)
                params = list(fitpars.keys())
                
                st.write(f"Parameters in retrieval: `{params}`")
                
                if st.button("Read Retrievals & Analyze"):
                    with st.spinner("Reading retrieval samples..."):
                        info = ret.read_retrievals(retrieval_dir, params)
                        st.success("Successfully loaded retrieval outputs!")
                        
                        # 3) Display corner plot
                        st.subheader("Corner Plot")
                        fig, ax = ret.plot_pair(info['samples_equal'], info['param_names'])
                        st.pyplot(fig)
                        
                        # 5) Max log likelihood model and plot against data
                        st.subheader("Max Log Likelihood Model vs Data")
                        max_logl_point = info['max_logl_point']
                        out = go.check_model_samples(
                            config, N=1, 
                            samples=np.atleast_2d(max_logl_point),
                            full_likelihood=True
                            )
                        chi2=out['chi_sq_per_pt'][0]        
                        # Create spectrum and errorbar plots
                        fig_spec = jpi.spectrum(
                                    [out['xdata']], 
                                    [out['ydata'][0]], 
                                    legend=["Max LogL Model"], 
                                    title=f"Chi-sq = {chi2:.2f}"
                                )
                        fig_spec = jpi.plot_errorbar(1e4/out['xdata'], out['ydata'][0], out['edata'][0], 
                                                     plot=fig_spec)
                                
                        streamlit_bokeh(fig_spec, theme="streamlit", key=f"spectrum_maxlogl")
                    
                        st.subheader("Generate Banded Profiles and Spectra")
                        N = st.slider('Number of Samples',value=10, min_value=10, max_value=1000)
                        options = [i for i in out['profiles'][0].keys() if 'pressure' not in i]
                        selected_pressure_bands = st.multiselect("Generate bands for: ", options, 
                                                    default=options[0:2])

                        if st.button("Generate"):
                            with st.spinner(f"Running {N} evaluations..."):
                                returns = ret.get_bands(config, info,
                                                    pressure_bands=selected_pressure_bands)

                                f_spec, a_spec = ret.plot_spectra_bands(returns)
                                st.pyplot(f_spec)
                                f_chem, a_chem = ret.plot_pressure_bands(returns)
                                st.pyplot(f_chem)
                            