import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
#matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from copy import copy
from utils.utils import dot
from utils.utils import mag
from utils.utils import mag2
from utils.utils import cosTheta
from utils.utils import angle
from utils.utils import cross
from utils.utils import vecAdd
from utils.utils import pi0Energy
from utils.utils import pi0InvMass
from utils.utils import getPhi
from utils.utils import getTheta
from utils.utils import getEnergy
from utils.utils import readFile
from utils import make_histos
from utils import histo_plotting
import matplotlib as mpl

# 1.) Necessary imports.    
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

from matplotlib import pylab as plt

import numpy as np
from scipy.optimize import minimize


find_beamQ = False

if find_beamQ:
    #df = pd.read_pickle("pickled_data/f18_out_with_logistics.pkl")
    #df = pd.read_pickle("pickled_data/f18_inbending_with_logi.pkl")
    df = pd.read_pickle("pickled_data/exp_outbending_183_with_logi.pkl")

    total_q = 0
    for run_num in df.RunNum.unique():
        run_q = df.query('RunNum == {}'.format(run_num)).beamQ.max()-df.query('RunNum == {}'.format(run_num)).beamQ.min()
        total_q+=run_q
        print(run_q)

    print(total_q) # Observe  1065038.810546875 for outbending small sample size
                # observe    42909169.32714844 for inbendining
                # observe    35441123.45703125 for outbending large sample size


#beam_Q = 1065038.810546875 #out
beam_Q = 35441123.45703125 #outbending large sample size
#beam_Q = 42909169.32714844 # in
N_a = 6E23
e = 1.6E-19
l = 5
rho = 0.07
units_conversion_factor = 1E-9

Lumi = N_a*l*rho*beam_Q/e*units_conversion_factor
print(Lumi)

