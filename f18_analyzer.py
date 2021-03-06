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

df = pd.read_pickle("f18_in_rec.pkl")
print(df.head())
print(df.shape)

for item in df.columns:
    print(item)

sys.exit()

#raw_f2018_in_data_epgg_no_cuts_no_corrections.pkl
#raw_f2018_in_data_epgg_no_cuts_with_corrections.pkl
df_epgg = pd.read_pickle("raw_f2018_in_data_epgg_no_cuts_no_corrections.pkl")
#df_epgg = pd.read_pickle("raw_f2018_in_data_epgg_no_cuts_with_corrections.pkl")
print(df_epgg.shape)
#print(df_epgg.Psector)

df_epgg.loc[:,"DeltaT"] = df_epgg['t1'] - df_epgg['t2']

df_dvpi0p = df_epgg




"""
Old cuts:
cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])

df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_mmep & cut_meepgg &
                    cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]


"""
cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.7  # mmep
cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.7  # meepgg
cut_mpt_FD = df_dvpi0p["MPt"] < 0.2  # mpt
cut_recon_FD = df_dvpi0p["reconPi"] < 2  # recon gam angle
cut_pi_mass_min = df_dvpi0p["Mpi0"] > 0.07
cut_pi_mass_max = df_dvpi0p["Mpi0"] < 0.20
cut_Esector1 = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]
cut_Esector2 = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector2"]
cut_Psector_FD = df_dvpi0p.Psector<7

cut_total = cut_Psector_FD & cut_mmep1_FD & cut_meepgg1_FD & cut_mpt_FD & cut_recon_FD & cut_pi_mass_min & cut_pi_mass_max & cut_Esector1 & cut_Esector2

df_dvpi0p = df_dvpi0p[cut_total]


df_dvpi0p.to_pickle("f18_fd_only_dvpi0p_no_corrs_old_cuts.pkl")

print(df_dvpi0p.shape)

"""
#common cuts
cut_xBupper = df_dvpi0p["xB"] < 1  # xB
cut_xBlower = df_dvpi0p["xB"] > 0  # xB
cut_Q2 = df_dvpi0p["Q2"] > 1  # Q2
cut_W = df_dvpi0p["W"] > 2  # W
cut_Ee = df_dvpi0p["Ee"] > 2  # Ee
cut_Ge = df_dvpi0p["Ge"] > 3  # Ge
#cut_Esector = True
cut_Esector = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]
cut_pi_mass_min = df_dvpi0p["Mpi0"] > 0.07
cut_pi_mass_max = df_dvpi0p["Mpi0"] < 0.20
cut_p_FD = df_dvpi0p["Psector"] < 3000
#cut_Ppmax = df_dvpi0p.Pp < 0.8  # Pp
# cut_Vz = np.abs(df_dvpi0p["Evz"] - df_dvpi0p["Pvz"]) < 2.5 + 2.5 / mag([df_dvpi0p["Ppx"], pi0SimInb_forDVCS["Ppy"], pi0SimInb_forDVCS["Ppz"]])
cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Esector & cut_pi_mass_min & cut_pi_mass_max & cut_p_FD

df_dvpi0p = df_dvpi0p[cut_common]

print(df_dvpi0p.shape)


cut_Pp1_FD = df_dvpi0p.Pp > 0.35  # Pp
cut_Psector_FD = df_dvpi0p.Psector<7
cut_Ptheta_FD = df_dvpi0p.Ptheta>2.477
cut_Gsector_FD = df_dvpi0p.Gsector<7
cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.569  # mmep
cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.167  # mpi0
cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.104  # mpi0
cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.771  # mmegg
cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > -0.0598  # mmegg
cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.805  # meepgg
cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.813  # meepgg
cut_mpt_FD = df_dvpi0p["MPt"] < 0.231  # mpt
cut_recon_FD = df_dvpi0p["reconPi"] < 1.098  # recon gam angle
cut_mmepgg2_FD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0287  # mmepgg

cut_FD = (cut_Psector_FD & cut_mmep1_FD)
#cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta_FD & cut_Gsector_FD &
#            cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
#            cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
#            cut_mpt_FD & cut_recon_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

df_dvpi0p = df_dvpi0p[cut_FD]

"""

sys.exit()



x_data = df_dvpi0p["Ppz"]
plot_title = "Proton Z Momentum vs. t"

#plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

y_data = df_dvpi0p["t1"]
y_range = [0,1,100]





# y_data = df["Ptheta"]
# y_range = [0,50,100]

var_names = ["t2","Ppz"]

ranges = [[0,2,100],y_range]
make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title=plot_title,logger=False,first_label="rad",
            filename="ExamplePlot",units=["GeV","GeV^2"])





sys.exit()

fs = filestruct.fs()


from utils.epg import epgFromROOT

fname = "test_new_filter.root"
converter = epgFromROOT(fname, entry_stop = None, mc = False, rec = False)
df_after_cuts = converter.df_epgg

"""

"""

print(df_after_cuts.head(4))



sys.exit()

df = pd.read_pickle("F18_All_DVPi0_Events.pkl")

df_epgg = df.filter(['Epx','Epy','Epz','Ppx','Ppy','Ppz','Gpx','Gpy','Gpz','Gpx2','Gpy2','Gpz2'], axis=1)

print(df.columns)

# useful objects
ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
df_epgg.loc[:, 'Ep'] = mag(ele)
df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
df_epgg.loc[:, 'Etheta'] = getTheta(ele)
df_epgg.loc[:, 'Ephi'] = getPhi(ele)

pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
df_epgg.loc[:, 'Pp'] = mag(pro)
df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
df_epgg.loc[:, 'Pphi'] = getPhi(pro)

gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
df_epgg.loc[:, 'Gp'] = mag(gam)
df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
df_epgg.loc[:, 'Gphi'] = getPhi(gam)

gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
df_epgg.loc[:, 'Gp2'] = mag(gam2)
df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

Ppt = mag([df_epgg['Ppx'], df_epgg['Ppy'], 0])

pi0 = vecAdd(gam, gam2)
VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
v3l = cross(beam, ele)
v3h = cross(pro, VGS)
costheta = cosTheta(VGS, gam)

v3g = cross(VGS, gam)
VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
            df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
            df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
            -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
            pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]

df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

# binning kinematics
df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
df_epgg.loc[:,'t1'] = 2 * M * (df_epgg['Pe'] - M)
df_epgg.loc[:,'t2'] = (M * df_epgg['Q2'] + 2 * M * df_epgg['nu'] * (df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta))\
/ (M + df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta)

df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
                            (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)

# exclusivity variables
df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
                        df_epgg["Pe"])**2 - mag2(VmissPi0)
df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
                        df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
                        df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
df_epgg.loc[:,'reconPi'] = angle(VmissPi0, pi0)
df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']

df_epgg.loc[:,"DeltaT"] = df_epgg['t1'] - df_epgg['t2']




x_data = df_epgg["Pp"]
plot_title = "Proton Z Momentum vs. t"

#plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

y_data = df_epgg["DeltaT"]
y_range = [-0.4,0.4,70]





# y_data = df["Ptheta"]
# y_range = [0,50,100]

var_names = ["t2","Ppz"]

ranges = [[0,2,70],y_range]
make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title=plot_title,logger=False,first_label="rad",
            filename="ExamplePlot",units=["GeV","GeV^2"])
