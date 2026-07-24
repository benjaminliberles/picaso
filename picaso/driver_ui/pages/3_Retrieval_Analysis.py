import streamlit as st
import os
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_bokeh import streamlit_bokeh

import picaso.driver as go
from picaso import justplotit as jpi
from picaso import retrieval as ret

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
    
    # Check if retrieval directory has changed to clear state
    if "last_retrieval_dir" in st.session_state and st.session_state["last_retrieval_dir"] != retrieval_dir:
        st.session_state.pop("retrieval_info", None)
        st.session_state.pop("max_logl_out", None)
        st.session_state.pop("chi2", None)
        st.session_state.pop("fig_spec", None)
        st.session_state.pop("last_retrieval_dir", None)
        st.session_state.pop("banded_returns", None)
    
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
                        
                        max_logl_point = info['max_logl_point']
                        out = go.check_model_samples(
                            config, N=1, 
                            samples=np.atleast_2d(max_logl_point),
                            full_likelihood=True
                        )
                        chi2 = out['chi_sq_per_pt'][0]        
                        # Create spectrum and errorbar plots
                        fig_spec = jpi.spectrum(
                                    [out['xdata']], 
                                    [out['ymodel'][0]], 
                                    legend=["Max LogL Model"], 
                                    title=f"Chi-sq = {chi2:.2f}"
                                )
                        fig_spec = jpi.plot_errorbar(1e4/out['xdata'], out['ydata'][0], out['edata'][0], 
                                                     plot=fig_spec)
                        
                        st.session_state['retrieval_info'] = info
                        st.session_state['max_logl_out'] = out
                        st.session_state['chi2'] = chi2
                        st.session_state['fig_spec'] = fig_spec
                        st.session_state['last_retrieval_dir'] = retrieval_dir
                        st.session_state.pop("banded_returns", None)
                        st.rerun()

                # Conditionally render results if they are in the session state
                if st.session_state.get('retrieval_info') is not None and st.session_state.get('last_retrieval_dir') == retrieval_dir:
                    info = st.session_state['retrieval_info']
                    out = st.session_state['max_logl_out']
                    chi2 = st.session_state['chi2']
                    fig_spec = st.session_state['fig_spec']
                    params = info['param_names']

                    # 3) Display corner plot
                    st.subheader("Corner Plot")
                    
                    # Create editable dataframe for pretty labels and ranges
                    with st.expander("Customize Corner Plot Labels and Ranges", expanded=False):
                        st.markdown("Customize parameter labels and ranges for the corner plot.")
                        
                        # Initialize or load parameters from session state
                        if "corner_params_df" not in st.session_state or st.session_state.get("last_retrieval_dir_params") != retrieval_dir:
                            default_labels = []
                            default_mins = []
                            default_maxs = []
                            for i, ip in enumerate(params):
                                default_labels.append(ip)
                                vals = info['samples_equal'][:, i]
                                default_mins.append(float(np.min(vals)))
                                default_maxs.append(float(np.max(vals)))
                            
                            st.session_state["corner_params_df"] = pd.DataFrame({
                                "Parameter": params,
                                "Pretty Label": default_labels,
                                "Min Range": default_mins,
                                "Max Range": default_maxs
                            })
                            st.session_state["last_retrieval_dir_params"] = retrieval_dir
                        
                        edited_df = st.data_editor(
                            st.session_state["corner_params_df"],
                            key="corner_params_editor",
                            disabled=["Parameter"]
                        )
                        st.session_state["corner_params_df"] = edited_df
                    
                    # Construct pretty_labels and ranges dictionaries
                    pretty_labels = {}
                    ranges = {}
                    for _, row in edited_df.iterrows():
                        param_name = row["Parameter"]
                        pretty_labels[param_name] = row["Pretty Label"]
                        ranges[param_name] = [row["Min Range"], row["Max Range"]]

                    fig, ax = ret.plot_pair(
                        info['samples_equal'], 
                        info['param_names'], 
                        pretty_labels=pretty_labels, 
                        ranges=ranges
                    )
                    st.pyplot(fig)

                    # Export figure options for manuscript publication
                    st.markdown("#### Export Corner Plot")
                    col1, col2 = st.columns(2)
                    with col1:
                        dpi_val = st.number_input("Resolution (DPI)", value=300, min_value=100, max_value=1200, step=100)
                    with col2:
                        img_format = st.selectbox("Image Format", options=["png", "pdf", "svg"], index=0)
                    
                    # Convert matplotlib figure to bytes for download
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format=img_format, dpi=dpi_val, bbox_inches='tight')
                    buf.seek(0)
                    
                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        st.download_button(
                            label="Download High-Resolution Corner Plot",
                            data=buf,
                            file_name=f"corner_plot.{img_format}",
                            mime=f"image/{img_format}" if img_format != "svg" else "image/svg+xml"
                        )
                    with col_save2:
                        if st.button("Save Figure Locally in Retrieval Directory"):
                            local_path = os.path.join(retrieval_dir, f"corner_plot_manuscript.{img_format}")
                            fig.savefig(local_path, format=img_format, dpi=dpi_val, bbox_inches='tight')
                            st.success(f"Successfully saved figure locally at: `{local_path}`")
                    
                    # 5) Max log likelihood model and plot against data
                    st.subheader("Max Log Likelihood Model vs Data")
                    streamlit_bokeh(fig_spec, theme="streamlit", key=f"spectrum_maxlogl")
                
                    st.subheader("Generate Banded Profiles and Spectra")
                    N = st.slider('Number of Samples', value=10, min_value=10, max_value=1000)
                    options = [i for i in out['profiles'][0].keys() if 'pressure' not in i]
                    selected_pressure_bands = st.multiselect(
                        "Generate bands for: ", 
                        options, 
                        default=options[0:2] if 'selected_pressure_bands_widget' not in st.session_state else None,
                        key="selected_pressure_bands_widget"
                    )

                    if st.button("Generate"):
                        with st.spinner(f"Running {N} evaluations..."):
                            returns = ret.get_bands(
                                config, info,
                                N=N,
                                pressure_bands=selected_pressure_bands
                            )
                            st.session_state['banded_returns'] = returns
                            st.rerun()

                    if 'banded_returns' in st.session_state:
                        returns = st.session_state['banded_returns']
                        f_spec, a_spec = ret.plot_spectra_bands(returns)
                        st.pyplot(f_spec)
                        f_chem, a_chem = ret.plot_pressure_bands(returns)
                        st.pyplot(f_chem)
