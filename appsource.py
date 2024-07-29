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

def preparepsispace():
    
    i = 0;
    psispace = [None]*400

    stribuff = str(i)
    Abuffer = 'FalseNegativeTestData' + stribuff + '.h5'
    with h5.File(Abuffer,'r') as A:
        psi = quickget('psi',A)
        A.close()
    psispace[i] = psi

    for i in range(1,400):
        stribuff = str(i)
        Abuffer = 'FalseNegativeTestData' + stribuff + '.h5'
        with h5.File(Abuffer,'r') as A:
            psi = quickget('psi',A)
            A.close()
        psispace[i] = psi

    return np.array(psispace).flatten()

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
def preparetestlevelsandfitstwo():

    Bbuffer = 'FNPowerTestFits.h5'
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

@st.cache_data
def prepareplotdatatwo(rs):

    Cbuffer = 'FalseNegativeTestData' + rs + '.h5'

    with h5.File(Cbuffer,'r') as C:
        a = quickget('a',C)
        b = quickget('b',C)
        c = quickget('c',C)
        L = quickget('L',C)
        psi = quickget('psi',C)
        tsample = quickget('TSample',C)
        xsample = quickget('XSample',C) 
        C.close()

    xspace = np.linspace(-L,L,100).flatten()
    yspace = a*(xspace**2) + b*(xspace) + c

    return tsample,xsample,xspace,yspace.flatten(),L


def makeplot(id,mfs,lfs,tlist):

    plotlist = []
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

def makeplottwo(id,mfs,lfs,tlist):

    plotlist = []
    ids = str(id)
    tsam,xsam,xspa,yspa,l = prepareplotdatatwo(ids)

    mp = mfs[id]
    mspa = mp[0]*(xspa**2) + mp[1]*xspa + mp[2]
    lp = lfs[id]
    lspa = lp[0]*(xspa**2) + lp[1]*xspa + lp[2]

    xspa = np.linspace(-1,1,100)
    tsam = tsam/l

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

MLS, LLS, MFS, LFS = preparetestlevelsandfits()
MLS2, LLS2, MFS2, LFS2 = preparetestlevelsandfitstwo()
cLs = preparecLspace()
psis = preparepsispace()
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
    FSFNswitch = st.toggle(label = 'False Positive - False Negative Toggle.', value = False, help = 'Use this switch to toggle between looking at the data underlying the False Positive and False Negative Profiles!')
    if FSFNswitch = False:
        
        reportstr = 'False Positive Profile'
        jobselectorhelpstr = 'Drawing the slider up increases the cL tested.'
        
    else:

        reportstr = 'False Negative Profile'
        jobselectorhelpstr = 'Drawing the slider up increases the ψ tested.'
        
    jobselector = st.slider(label = 'Slide a test number to get a better look at the test!',min_value=0,max_value=399,value = 101,step=1,help = jobselectorhelpstr)
    st.metric('Power Test No.',(jobselector))
    st.metric('cL', (cLs[jobselector]))
    st.metric('Proposed Test Confidence in the Null Hypothesis.',(MLS[jobselector][0]))
    st.metric('MEP-CvM Test Confidence in the Null Hypothesis.' ,(LLS[jobselector][0]))
    st.write('Select the data shown below.')
    p1 = st.checkbox('Generated Data', value=True)
    p2 = st.checkbox('True Trend', value=True)
    p3 = st.checkbox('True Error', value=False)
    p4 = st.checkbox('MMD-Trained Quadratic', value=True)
    p5 = st.checkbox('MMD-Trained Error', value=True)
    p6 = st.checkbox('LSQ-Trained Quadratic', value=False)
    p7 = st.checkbox('LSQ-Trained Error', value=False)

with rcol:
    
    st.plotly_chart(makeplot(jobselector,MFS,LFS,[p1,p2,p3,p4,p5,p6,p7]), use_container_width=True)

    f = open("GoingPowerProfile1.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)
