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

def quickget(ds,h5f):
    return np.array(h5f.get(ds))

@st.cache_data
def preparecLspace():

    i = 0;
    cLspace = [None]*400

    stribuff = str(i)
    Abuffer = 'PowerTestData' + stribuff + '.h5'
    with h5.File(Abuffer,'r') as A:
        cL = quickget('cL',A)
        A.close()
    cLspace[i] = cL

    for i in range(1,400):
        stribuff = str(i)
        Abuffer = 'PowerTestData' + stribuff + '.h5'
        with h5.File(Abuffer,'r') as A:
            cL = quickget('cL',A)
            A.close()
        cLspace[i] = cL

    return np.array(cLspace).flatten()

@st.cache_data
def preparetestlevelsandfits():

    Bbuffer = 'PowerTestFits.h5'
    with h5.File(Bbuffer,'r') as B:
        mmdlevelspace = quickget('MMDLevel',B)
        lsqlevelspace = quickget('LSQLevel',B)
        mmdfitspace = quickget('MMDFit',B)
        lsqfitspace = quickget('LSQFit',B)
        B.close()

    return mmdlevelspace, lsqlevelspace, mmdfitspace, lsqfitspace

@st.cache_data
def prepareplotdata(rs):

    Cbuffer = 'PowerTestData' + rs + '.h5'

    with h5.File(Cbuffer,'r') as C:
        a = quickget('a',C)
        b = quickget('b',C)
        L = quickget('L',C)
        cL = quickget('cL',C)
        tsample = quickget('TSample',C)
        xsample = quickget('XSample',C) 
        C.close()
    
    yspace = a + b*np.exp(np.linspace(-cL,cL,100))

    return tsample,xsample,yspace.flatten(),L

def makeplot(id,mfs,lfs):
    
    ids = str(id)
    tsam,xsam,yspa,l = prepareplotdata(ids)
    xspa = np.linspace(-l,l,100).flatten()

    mp = mfs[id]
    mspa = mp[0]*(xspa**2) + mp[1]*xspa + mp[2]
    lp = lfs[id]
    lspa = lp[0]*(xspa**2) + lp[1]*xspa + lp[2]

    xspa = np.linspace(-1,1,100)
    tsam = tsam/l

    fig, ax = plt.subplots()

    ax.scatter(tsam,xsam, s = [10.0]*200, color = (0.0,0.0,0.0), label = 'Generated Data')
    ax.plot(xspa,yspa, color = (0.0,0.0,0.0), label = 'True Trend')
    ax.plot(xspa,mspa, color = (0.0,0.247058823529412,0.447058823529412), label = 'MMD-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None), s = [10.0]*200, color = (0.0,0.247058823529412,0.447058823529412), label = 'MMD-Trained Error')
    ax.plot(xspa,lspa, color = (0.776470588235294,0.0470588235294118,0.188235294117647), label = 'LSQ-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None), s = [10.0]*200, color = (0.776470588235294,0.0470588235294118,0.188235294117647), label = 'LSQ-Trained Error')
    ax.set_xlabel('t/L') 
    ax.set_ylabel('x') 
    ax.legend()

    return fig

def makeplottwo(id,mfs,lfs):

    ids = str(id)
    tsam,xsam,yspa,l = prepareplotdata(ids)
    xspa = np.linspace(-l,l,100).flatten()

    mp = mfs[id]
    mspa = mp[0]*(xspa**2) + mp[1]*xspa + mp[2]
    lp = lfs[id]
    lspa = lp[0]*(xspa**2) + lp[1]*xspa + lp[2]

    xspa = np.linspace(-1,1,100)
    tsam = tsam/l

    # Add traces
    t1 = (go.Scatter(x=tsam.flatten(), y=xsam.flatten(),
                    mode='markers',
                    marker=dict(size=[4]*200),
                    name='Generated Data',
                    marker_color = 'rgba(0,0,0,1.0)'
    ))
    
    t2 = (go.Scatter(x=xspa, y=yspa,
                    mode='lines',
                    name='True Trend',
                    marker_color = 'rgba(0,0,0,1.0)'
    ))

    t3 = (go.Scatter(x=xspa, y=mspa,
                    mode='lines',
                    name='MMD-Trained Quadratic',
                    marker_color = 'rgba(0,63,114,1.0)'
    ))

    t4 = (go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None),
                    mode='markers',
                    marker=dict(size=[4]*200),
                    name='MMD-Trained Error',
                    marker_color = 'rgba(0,63,114,1.0)'
    ))

    t5 = (go.Scatter(x=xspa, y=lspa,
                    mode='lines',
                    name='LSQ-Trained Quadratic',
                    marker_color = 'rgba(198,12,48,1.0)'
    ))
    
    t6 = (go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None),
                    mode='markers',
                    marker=dict(size=[4]*200, line=dict(width=0)),
                    name='LSQ-Trained Error',
                    marker_color = 'rgba(198,12,48,1.0)'
    ))

    fig = go.Figure(data = [t1,t2,t3,t4,t5,t6])

    fig.update_layout(xaxis_title="t / L", yaxis_title="x")

    return fig

MLS, LLS, MFS, LFS = preparetestlevelsandfits()
cLs = preparecLspace()
# -----------------------------------------------------------------------------
# Draw the actual page
lcol, rcol = st.columns(2)

# Set the title that appears at the top of the page.
with lcol:
    '''
    # :globe_with_meridians: Robust Kernelised Composite Goodness-of-Fit Testing for Conditional Relationships Supplemental Web App

    Hello! This a Streamlit-powered web app to support a poster made for RSC Exeter 2024. This contextualises a figure profiling the power of a statistical test presented on the poster, namely by letting the reader see the data plotted, point-by-point, next to the underlying test each plotted point represents. The principle image of the figure is included here as well. 

    Only keeping the app open if it's in use would be deeply appreciated, as the overall resources available to this app are limited.
    '''

    # Add some spacing
    ''
    ''
    cLselector = st.slider(label = 'Slide a test number to get a better look at the test! Drawing the slider up increases the cL tested.',min_value=0,max_value=399,value = 101,step=1)
    st.metric('Power Test No.',(cLselector))
    st.metric('cL', (cLs[cLselector]))
    st.metric('Proposed Test Confidence in the Null Hypothesis.',(MLS[cLselector][0]))
    st.metric('MEP-CvM Test Confidence in the Null Hypothesis.' ,(LLS[cLselector][0]))

with rcol:
    st.plotly_chart(makeplottwo(cLselector,MFS,LFS), use_container_width=True)

    f = open("GoingPowerProfile1.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)
