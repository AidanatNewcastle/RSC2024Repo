import streamlit as st
import jax.numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Robust Kernelised Composite Goodness-of-Fit Testing for Conditional Relationships Supplemental Web App',
    page_icon=':iphone:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

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

    ax.scatter(tsam,xsam, s = [0.5]*200, color = (0.0,0.0,0.0), label = 'Generated Data')
    ax.plot(xspa,yspa, color = (0.0,0.0,0.0), label = 'True Trend')
    ax.plot(xspa,mspa, color = (0.0,0.247058823529412,0.447058823529412), label = 'MMD-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None), s = [0.5]*200, color = (0.0,0.247058823529412,0.447058823529412), label = 'MMD-Trained Error')
    ax.plot(xspa,lspa, color = (0.776470588235294,0.0470588235294118,0.188235294117647), label = 'LSQ-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None), s = [0.5]*200, color = (0.776470588235294,0.0470588235294118,0.188235294117647), label = 'LSQ-Trained Error')
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

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=tsam, y=xsam,
                    mode='markers',
                    name='Generated Data'
                    marker_color = 'rgba(0,0,0,1.0)'
    ))
    
    fig.add_trace(go.Scatter(x=xspa, y=yspa,
                    mode='lines',
                    name='True Trend'
                    marker_color = 'rgba(0,0,0,1.0)'
    ))

    fig.add_trace(go.Scatter(x=xspa, y=mspa,
                    mode='lines',
                    name='MMD-Trained Quadratic'
                    marker_color = 'rgba(0,63,114,1.0)'
    ))

    fig.add_trace(go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None),
                    mode='markers',
                    name='MMD-Trained Error'
                    marker_color = 'rgba(0,63,114,1.0)'
    ))

    fig.add_trace(go.Scatter(x=xspa, y=mspa,
                    mode='lines',
                    name='LSQ-Trained Quadratic'
                    marker_color = 'rgba(198,12,48,1.0)'
    ))
    
    fig.add_trace(go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None),
                    mode='markers',
                    name='LSQ-Trained Error'
                    marker_color = 'rgba(198,12,48,1.0)'
    ))

    return fig

def prepareplots(mfspp,lfspp):

    plotsarray = [None]*400
    
    for i in range(0,400)
        plotsarray[i] = makeplot(i,mfspp,lfspp)

    return plotsarray
# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :iphone: Robust Kernelised Composite Goodness-of-Fit Testing for Conditional Relationships Supplemental Web App

Hello! This a Streamlit-powered web app to support a poster made for RSC Exeter 2024. This contextualises a figure profiling the power of a statistical test presented on the poster, namely by letting the reader see the data and the underlying behaviour thereof as well as the quadratic models fitted by the proposed and conventional methods and the corresponding Goodness-of-Fit confidences recorded. 

Only keeping the app open if it's in use would be deeply appreciated, as the overall resources available to this app are limited.
'''

# Add some spacing
''
''

MLS, LLS, MFS, LFS = preparetestlevelsandfits()
cLs = preparecLspace()
cLselector = st.slider(label = 'Slide a test number to get a better look at the test! Drawing the slider up increases the cL tested.',min_value=0,max_value=399,value = 20,step=1)
st.write('Power Test No.'+ str(cLselector) + ', cL = ', str(cLs[cLselector]))
st.write('Proposed Test Confidence in the Null Hypothesis: ' + str(MLS[cLselector][0]))
st.write('MEP-CvM Test Confidence in the Null Hypothesis: ' + str(LLS[cLselector][0]))
st.pyplot(makeplot(cLselector,MFS,LFS))
