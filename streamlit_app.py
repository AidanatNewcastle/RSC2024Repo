import streamlit as st
import pandas as pd
import math
import jax.numpy as np
from jax import jit
import h5py as h5
import matplotlib.pyplot as plt
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Robust Kernelised Composite Goodness-of Fit Testing for Conditional Relationships Supplemental Web App',
    page_icon=':iphone:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

def quickget(ds,h5f):
    return np.array(h5f.get(ds))

@jit
@st.cache_data
def preparecLspace():

    i = 0;
    cLspace = [None]*400

    stribuff = str(i)
    Abuffer = 'PowerTestData' + stribuff + '.h5'
    with A as h5.File(Abuffer,'r'):
        cL = quickget('cL',A)
        A.close()
    cLspace[i] = cL

    for i in range(1,400):
            stribuff = str(i)
        Abuffer = 'PowerTestData' + stribuff + '.h5'
        with A as h5.File(Abuffer,'r'):
            cL = quickget('cL',A)
            A.close()
        cLspace[i] = cL

    return np.array(cLspace).flatten()

@jit
@st.cache_data
def preparetestlevelsandfits():

    Bbuffer = 'PowerTestFits.h5'
    with B as h5.File(Bbuffer,'r')
        mmdlevelspace = quickget('MMDLevel',B)
        lsqlevelspace = quickget('LSQLevel',B)
        mmdfitspace = quickget('MMDFit',B)
        lsqfitspace = quickget('LSQFit',B)
        B.close()

    return mmdlevelspace, lsqlevelspace, mmdfitspace, lsqfitspace

@jit
@st.cache_data
def prepareplotdata(rs):

    Cbuffer = 'PowerTestData' + rs + '.h5'

    with C as h5.File(Cbuffer,'r'):
        a = quickget('a',C)
        b = quickget('b',C)
        L = quickget('L',C)
        cL = quickget('cL',C)
        tsample = quickget('TSample',C)
        xsample = quickget('XSample',C) 
        C.close()
    
    yspace = a + b*np.exp(np.linspace(-cL,cL,1000))

    return tsample,xsample,yspace,L

def makeplot(id,mfs,lfs):
    
    ids = str(id)
    tsam,xsam,yspa,l = prepareplotdata(ids)
    xspa = np.linspace(-l,l,1000)

    mp = mfs[:,id]
    mspa = mp[0]*(xspa**2) + mp[1]*xspa + mp[2]
    lp = lfs[:,id]
    lspa = lp[0]*(xspa**2) + lp[1]*xspa + lp[2]

    fig, ax = plt.subplots()

    ax.scatter(tsam,xsam, color = (0.0,0.0,0.0), label = 'Generated Data')
    ax.plot(xspa,yspa, color = (0.0,0.0,0.0) label = 'True Trend')
    ax.plot(xspa,mspa, color = (0.0,0.247058823529412,0.447058823529412), label = 'MMD-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((mspa+mp[3]),(mspa-mp[3])), axis=None), c = np.array([[0.0,0.247058823529412,0.447058823529412]]), label = 'MMD-Trained Error')
    ax.plot(xspa,lspa, color = (0.980392156862745,0.976470588235294,0.964705882352941), label = 'LSQ-Trained Quadratic')
    ax.scatter(np.concatenate((xspa,xspa), axis=None),np.concatenate(((lspa+lp[3]),(lspa-lp[3])), axis=None), c = np.array([[0.980392156862745,0.976470588235294,0.964705882352941]]), label = 'LSQ-Trained Error')
    ax.xlabel('t/L') 
    ax.ylabel('x') 
    ax.legend()

    return fig

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

MLS, LLS, MFS, LFS = preparetestlevelsandfits()
cLs = preparecLspace()
cLselector = st.slider(label = 'Slide a test number to get a better look at the test! Drawing the slider up increases the cL tested.',min_value=0,max_value=399,value = 20,step=1)
cLselected = cLs[cLsselector]
st.write('Power Test No.'+ str(cLselector) + ', cL = ', str(cLs[cLselector]))
st.write('Proposed Test Level = ' + str(MLS[cLselector]))
st.write('MEP-CvM Test Level = ' + str(LLS[cLselector]))
st.pyplot(makeplot(cLselector,MFS,LFS))
