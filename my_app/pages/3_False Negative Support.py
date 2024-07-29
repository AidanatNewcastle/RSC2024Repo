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
    page_title='False Negative Support',
    page_icon=':two:', 
    layout = "wide"
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

st.markdown("# False Negative Support :two:")
st.sidebar.markdown("# False Negative Support :two:")

def quickget2(ds,h5f):
    return np.array(h5f.get(ds))

@st.cache_data
def preparepsispace():

    i = 0;
    psispace = [None]*400
    
    stribuff = str(i)
    Abuffer = 'FalseNegativeTestData' + stribuff + '.h5'
    with h5.File(Abuffer,'r') as A:
        psi = quickget2('psi',A)
        A.close()
    psispace[i] = psi

    for i in range(1,400):
        stribuff = str(i)
        Abuffer = 'FalseNegativeTestData' + stribuff + '.h5'
        with h5.File(Abuffer,'r') as A:
            psi = quickget2('psi',A)
            A.close()
        psispace[i] = psi

    return np.array(psispace).flatten()

@st.cache_data
def preparetestlevelsandfits2():

    Bbuffer = 'FNPowerTestFits.h5'
    with h5.File(Bbuffer,'r') as B:
        mmdlevelspace = quickget2('MMDLevel',B)
        lsqlevelspace = quickget2('LSQLevel',B)
        mmdfitspace = quickget2('MMDFit',B)
        lsqfitspace = quickget2('LSQFit',B)
        B.close()

    return mmdlevelspace, lsqlevelspace, mmdfitspace, lsqfitspace

@st.cache_data
def prepareplotdata2(rs):

    Cbuffer = 'FalseNegativeTestData' + rs + '.h5'

    with h5.File(Cbuffer,'r') as C:
        a = quickget2('a',C)
        b = quickget2('b',C)
        c = quickget2('c',C)
        L = quickget2('L',C)
        psi = quickget2('psi',C)
        tsample = quickget2('TSample',C)
        xsample = quickget2('XSample',C) 
        C.close()
    
    xspace = np.linspace(-L,L,100).flatten()
    yspace = a*(xspace**2) + b*(xspace) + c

    return tsample,xsample,xspace,yspace.flatten(),L

def makeplot2(id,mfs,lfs,tlist):

    plotlist = []
    ids = str(id)
    tsam,xsam,xspa,yspa,l = prepareplotdata2(ids)

    mp = mfs[id]
    mspa = mp[0]*(xspa**2) + mp[1]*xspa + mp[2]
    lp = lfs[id]
    lspa = lp[0]*(xspa**2) + lp[1]*xspa + lp[2]

    tsam = tsam/l
    xspa = xspa/l

    # Add traces
    if tlist[0]:
        
        t1 = (go.Scatter(x=tsam.flatten(), y=xsam.flatten(),
                    mode='markers',
                    marker=dict(size=[7]*200,symbol = 'x', line=dict(width=0)),
                    name='Generated Data',
                    marker_color = 'rgba(125,124,123,1.0)'
        ))
        plotlist.append(t1)

    if tlist[1]:

        t2 = (go.Scatter(x=xspa, y=yspa,
                    mode='lines',
                    name='True Trend',
                    marker_color = 'rgba(125,124,123,1.0)'
        ))
        plotlist.append(t2)
        
    if tlist[2]:
        
        t3 = (go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((yspa+1),(yspa-1)), axis=None),
                    mode='markers',
                    marker=dict(size=[3]*200, line=dict(width=0)),
                    name='True Error',
                    marker_color = 'rgba(125,124,123,1.0)'
        ))
        plotlist.append(t3)

    if tlist[3]:
        
        t4 = (go.Scatter(x=xspa, y=mspa,
                    mode='lines',
                    name='MMD-Trained Quadratic',
                    marker_color = 'rgba(0,63,114,1.0)'
        ))
        plotlist.append(t4)

    if tlist[4]:

        t5 = (go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None),
                    mode='markers',
                    marker=dict(size=[3]*200, line=dict(width=0)),
                    name='MMD-Trained Error',
                    marker_color = 'rgba(0,63,114,1.0)'
        ))
        plotlist.append(t5)

    if tlist[5]:

        t6 = (go.Scatter(x=xspa, y=lspa,
                    mode='lines',
                    name='LSQ-Trained Quadratic',
                    marker_color = 'rgba(198,12,48,1.0)'
        ))
        plotlist.append(t6)

    if tlist[6]:
        
        t7 = (go.Scatter(x=np.concatenate((xspa,xspa), axis=None), y=np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None),
                    mode='markers',
                    marker=dict(size=[3]*200, line=dict(width=0)),
                    name='LSQ-Trained Error',
                    marker_color = 'rgba(198,12,48,1.0)'
        ))
        plotlist.append(t7)

    fig = go.Figure(data = plotlist)

    if plotlist == []:
        fig.update_layout(yaxis_range=[-1,1],xaxis_range=[-1,1])
    else:
        fig.update_layout(xaxis_title="t / L", yaxis_title="x")

    return fig

MLS, LLS, MFS, LFS = preparetestlevelsandfits2()
psis = preparepsispace()
# -----------------------------------------------------------------------------
# Draw the actual page
lcol, rcol = st.columns(2)

# Set the title that appears at the top of the page.
with lcol:
    '''
    # :two: False Negative Support : 
    '''

    # Add some spacing
    ''
    ''
    cLselector = st.slider(label = 'Slide a test number to get a better look at the test! Drawing the slider up increases the cL tested.',min_value=0,max_value=399,value = 101,step=1)
    st.metric('Power Test No.',(cLselector))
    st.metric('ψ', (cLs[cLselector]))
    st.metric('Proposed Test Confidence in the Null Hypothesis.',(MLS[cLselector][0]))
    st.metric('MEP-CvM Test Confidence in the Null Hypothesis.' ,(LLS[cLselector][0]))
    st.write('Select the data shown below.')
    p1 = st.checkbox('Generated Data', value=True)
    p2 = st.checkbox('True Trend', value=True)
    p3 = st.checkbox('True Error', value=False)
    p4 = st.checkbox('MMD-Trained Quadratic', value=True)
    p5 = st.checkbox('MMD-Trained Error', value=True)
    p6 = st.checkbox('LSQ-Trained Quadratic', value=False)
    p7 = st.checkbox('LSQ-Trained Error', value=False)
    #st.metric('MMD-Trained Standard Deviation from Trained Trend.',np.abs((MFS[cLselector][-1])))
    #st.metric('LSQ-Trained Standard Deviation from Trained Trend.',(LFS[cLselector][-1]))

with rcol:
    st.plotly_chart(makeplot2(cLselector,MFS,LFS,[p1,p2,p3,p4,p5,p6,p7]), use_container_width=True)
