import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
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
from utils import filestruct
pd.set_option('mode.chained_assignment', None)

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
E = 10.6

def makeDVpi0(df_epgg):

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = df_epgg.loc[:, "Q2"] > 1  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        # proton reconstruction quality
        #cut_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
        cut_proton = True
        cut_FD_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
        cut_CD_proton = (df_epgg.loc[:, "Psector"]>7) & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)
        #cut_proton = (cut_FD_proton)|(cut_CD_proton)
        #cut_proton = cut_FD_proton
        #cut_proton = cut_CD_proton

        #Experimental cuts
        cut_gtheta = df_epgg.loc[:, "Gtheta"] > 0.5
        cut_gtheta2 = df_epgg.loc[:, "Gtheta2"] > 0.5
        cut_etheta = df_epgg.loc[:, "Etheta"] > 0.8
        cut_ep = df_epgg.loc[:, "Ep"] < 77

        #cut_genQ2 = df_epgg.loc[:, "GenQ2"] < 100  # mmep
        #cut_genW = df_epgg.loc[:, "GenW"] < 1.975  # mmep

        

        # Exclusivity cuts
        cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.1  # mmep
        cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.1  # meepgg
        cut_mpt = df_epgg.loc[:, "MPt"] < 0.1  # mpt
        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.161 #0.2
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.115 #0.07
        #cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])
        #cut_Vz = np.abs(df_epgg["Evz"] - df_epgg["Pvz"]) < 2.5 + 2.5 / mag([df_epgg["Ppx"], df_epgg["Ppy"], df_epgg["Ppz"]])

        df_dvpi0 = df_epgg.loc[cut_gtheta & cut_gtheta2 & cut_ep &  cut_etheta & cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_proton & cut_mmep & cut_meepgg & cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower, :]

        #For an event, there can be two gg's passed conditions above.
        #Take only one gg's that makes pi0 invariant mass
        #This case is very rare.
        #For now, duplicated proton is not considered.
        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        

        return df_dvpi0


if __name__ == "__main__":
        df_loc_rec = "pickled_data/f18_bkmrg_in_rec.pkl"
        df_rec = pd.read_pickle(df_loc_rec)
        ic(df_rec.shape)
        df_dvpp_rec = makeDVpi0(df_rec)
        ic(df_dvpp_rec.shape)
        df_dvpp_rec.to_pickle("pickled_dvpip/f18_bkmrg_in_dvpp_rec_noseccut.pkl")














