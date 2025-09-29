"""Streamlit UI for PRODX RAG App

This is a chat interface using Streamlit to interact with the RAG system via the FastAPI backend.
Provides natural language queries for PRODX analysis, dependency impacts, etc.

Features:
- Chat interface for RAG queries.
- Separate tabs for dependency analysis and log analysis.
- Displays sources and suggestions.

Run with: streamlit run streamlit_ui.py

Requires: streamlit, requests (for API calls), pandas (for optional display).
Business Context: Enables tenants to query the RAG app via web UI, improving productivity without CLI/API expertise.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any
import pandas as pd

# API base URL (local dev)
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(page_title="PRODX RAG Assistant", page_icon="🏦", layout="wide")

st.title("🏦 PRODX RAG Assistant")
st.markdown("**Ask questions about the PRODX framework, job failures, update impacts, or analyze dependencies/logs.**")

# Sidebar for API config
st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL, help="FastAPI server URL")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["💬 RAG Chat", "🔍 Dependency Analysis", "📋 Log Analysis"])

with tab1:
    st.header("Natural Language Query")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Query input
    if prompt := st.chat_input("Enter your query about PRODX (e.g., 'Impact of updating cryptography?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = requests.post(f"{api_url}/query", json={"query": prompt, "k": 3})
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(data['response'])
                        st.markdown("**Sources:**")
                        sources_df = pd.DataFrame(data['sources'])
                        st.dataframe(sources_df)
                        confidence = f"Confidence: {data['confidence']:.2f}"
                        st.caption(confidence)
                        
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": data['response'] + "\n\n" + confidence})
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

with tab2:
    st.header("Dependency Update Impact Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        package = st.text_input("Package Name", value="cryptography", help="e.g., cryptography")
        current_version = st.text_input("Current Version", value="3.2.0")
        new_version = st.text_input("New Version", value="3.4.8")
    with col2:
        lob = st.selectbox("Line of Business (LOB)", options=["", "009", "003", "017"], index=0)
    
    if st.button("Analyze Impact"):
        if package and current_version and new_version:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{api_url}/analyze_dependency", json={
                        "package": package,
                        "current_version": current_version,
                        "new_version": new_version,
                        "lob": lob if lob else None
                    })
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Risk Level: **{data['risk_level']}**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Conflicts:**")
                            for conflict in data['conflicts']:
                                st.error(conflict)
                        with col_b:
                            st.markdown("**Affected LOBs:**")
                            for l in data['affected_lobs']:
                                st.warning(f"LOB {l}")
                        
                        st.markdown("**Suggestions:**")
                        for suggestion in data['suggestions']:
                            st.success(suggestion)
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please fill in package and versions.")

with tab3:
    st.header("Log Impact Analysis")
    
    log_file = st.selectbox("Log File", options=["sample_airflow_log.txt"], help="Select from data/ directory")
    update_type = st.selectbox("Update Type", options=["general", "schema_evolution", "athena_parser"])
    
    if st.button("Analyze Log"):
        with st.spinner("Analyzing log..."):
            try:
                response = requests.post(f"{api_url}/analyze_log", json={
                    "log_file": log_file,
                    "update_type": update_type
                })
                if response.status_code == 200:
                    data = response.json()
                    st.subheader(f"Severity: **{data['severity']}**")
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown("**Impacts:**")
                        for impact in data['impacts']:
                            st.warning(impact)
                    with col_d:
                        st.markdown("**Suggested Fixes:**")
                        for fix in data['fixes']:
                            st.success(fix)
                    
                    if data['matches']:
                        st.markdown("**Detected Patterns:**")
                        for match in data['matches']:
                            st.code(match)
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("Built with modular RAG for ABC Bank PRODX framework. Source code in src/ modules.")