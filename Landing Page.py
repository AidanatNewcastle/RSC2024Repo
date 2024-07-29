import streamlit as st
import jax.numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import base64
import textwrap

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Robust Kernelised Composite Goodness-of-Fit Testing for Conditional Relationships Supplemental Web App',
    page_icon=':globe_with_meridians:', 
    layout = "wide"
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

def render_svg(svg): #Credit to https://gist.github.com/treuille for this function!
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

st.markdown("# Landing Page :globe_with_meridians:")
st.sidebar.markdown("# Landing Page :globe_with_meridians:")

# -----------------------------------------------------------------------------
# Draw the actual page
lcol, rcol = st.columns(2)

# Set the title that appears at the top of the page.
with lcol:
    '''
    # :globe_with_meridians: Robust Kernelised Composite Goodness-of-Fit Testing for Conditional Relationships Supplemental Web App

    Hello! This a Streamlit-powered web app to support a poster made for RSC Exeter 2024. This contextualises a figure profiling the power of a statistical test presented on the poster, namely by letting the reader see the data plotted, point-by-point, next to the underlying test each plotted point represents with the ability to select and deselect elements of a demonstrative Plotly chart. The principle image of the figure is included here as well. 

    Only keeping the app open if it's in use would be deeply appreciated, as the overall resources available to this app are limited.
    '''

    # Add some spacing
    ''
    ''

with rcol:
    st.plotly_chart(makeplottwo(cLselector,MFS,LFS,[p1,p2,p3,p4,p5,p6,p7]), use_container_width=True)

    f = open("GoingPowerProfile1.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)
