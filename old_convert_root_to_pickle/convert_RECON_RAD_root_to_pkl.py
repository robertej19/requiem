#!/usr/bin/env python3
"""code is modified from Sangbaek's initial work on converting root to pandas"""
import uproot
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from copy import copy
import sys 

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.6 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector

def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

def mag2(vec1):
    # square of L2 norm
    return  dot(vec1, vec1)

def cosTheta(vec1, vec2):
    # cosine angle between two 3d vectors
    return dot(vec1,vec2)/np.sqrt(mag2(vec1) * mag2(vec2))

def angle(vec1, vec2):
    # angle between two 3d vectors
    return 180/np.pi*np.arccos(np.minimum(1, cosTheta(vec1, vec2)))

def cross(vec1, vec2):
    # cross product of two 3d vectors
    return [vec1[1]*vec2[2]-vec1[2]*vec2[1], vec1[2]*vec2[0]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0]]

def vecAdd(gam1, gam2):
    # add two 3d vectors
    return [gam1[0]+gam2[0], gam1[1]+gam2[1], gam1[2]+gam2[2]]

def pi0Energy(gam1, gam2):
    # reconstructed pi0 energy of two 3d photon momenta
    return mag(gam1)+mag(gam2)

def pi0InvMass(gam1, gam2):
    # pi0 invariant mass of two 3d photon momenta
    pi0mass2 = pi0Energy(gam1, gam2)**2-mag2(vecAdd(gam1, gam2))
    pi0mass2 = np.where(pi0mass2 >= 0, pi0mass2, 10**6)
    pi0mass = np.sqrt(pi0mass2)
    pi0mass = np.where(pi0mass > 100, -1000, pi0mass)
    return pi0mass

def getPhi(vec1):
    # azimuthal angle of one 3d vector
    return 180/np.pi*np.arctan2(vec1[1], vec1[0])

def getTheta(vec1):
    # polar angle of one 3d vector
    return 180/np.pi*np.arctan2(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]), vec1[2])

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)

def readFile(fname):
    #read root using uproot
    ffile = uproot.open(fname)
    tree = ffile["T"]
    return tree

def readEPGG(tree, entry_stop = None):
    
    print(tree.keys())

    # data frames and their keys to read Z part
    df_electronGen = pd.DataFrame()
    df_protonGen = pd.DataFrame()
    df_gammaGen = pd.DataFrame()
    df_piGen = pd.DataFrame()

    eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
    proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
    gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
    piKeysGen = ["GenPipx", "GenPipy", "GenPipz"]

    # read keys
    for key in eleKeysGen:
        df_electronGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in proKeysGen:
        df_protonGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in gamKeysGen:
        df_gammaGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in piKeysGen:
        df_piGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)

    print(tree['GenEpx'].array())
    print(tree['Gpx'].array())

    
    
    print(df_electronGen)
    print(df_protonGen)
    print(df_gammaGen)


    #convert data type to standard double
    df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
    df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
    df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})
    df_piGen = df_piGen.astype({"GenPipx": float, "GenPipy": float, "GenPipz": float})


    #set up a dummy index for merging
    df_electronGen.loc[:,'event'] = df_electronGen.index
    df_protonGen.loc[:,'event'] = df_protonGen.index
    df_piGen.loc[:,'event'] = df_piGen.index

    df_gammaGen.loc[:,'event'] = df_gammaGen.index.get_level_values('entry')

    #sort columns for readability
    df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz"]]

    #two g's to one gg.
    gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
    df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)

    gam1 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
    gam1 = gam1.reset_index(drop=True)
    gam2 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==1]
    gam2 = gam2.reset_index(drop=True)

    gam1.loc[:,"GenGp2"] = gam2.loc[:,"GenGp"]
    gam1.loc[:,"GenGpx2"] = gam2.loc[:,"GenGpx"]
    gam1.loc[:,"GenGpy2"] = gam2.loc[:,"GenGpy"]
    gam1.loc[:,"GenGpz2"] = gam2.loc[:,"GenGpz"]
    df_gammaGen = gam1

    #sort GenG indices so that GenGp > GenGp2. This is because Gp > Gp2 at reconstruction level.
    df_gammaGencopy = copy(df_gammaGen)
    df_gammaGencopy.loc[:, "GenGp"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp"], df_gammaGen.loc[:, "GenGp2"])
    df_gammaGencopy.loc[:, "GenGpx"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx"], df_gammaGen.loc[:, "GenGpx2"])
    df_gammaGencopy.loc[:, "GenGpy"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy"], df_gammaGen.loc[:, "GenGpy2"])
    df_gammaGencopy.loc[:, "GenGpz"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz"], df_gammaGen.loc[:, "GenGpz2"])
    df_gammaGencopy.loc[:, "GenGp2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp2"], df_gammaGen.loc[:, "GenGp"])
    df_gammaGencopy.loc[:, "GenGpx2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx2"], df_gammaGen.loc[:, "GenGpx"])
    df_gammaGencopy.loc[:, "GenGpy2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy2"], df_gammaGen.loc[:, "GenGpy"])
    df_gammaGencopy.loc[:, "GenGpz2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz2"], df_gammaGen.loc[:, "GenGpz"])
    df_gammaGen = df_gammaGencopy


    #spherical coordinates
    eleGen = [df_electronGen["GenEpx"], df_electronGen["GenEpy"], df_electronGen["GenEpz"]]
    df_electronGen.loc[:, 'GenEp'] = mag(eleGen)
    df_electronGen.loc[:, 'GenEtheta'] = getTheta(eleGen)
    df_electronGen.loc[:, 'GenEphi'] = getPhi(eleGen)

    piGen = [df_piGen["GenPipx"], df_piGen["GenPipy"], df_piGen["GenPipz"]]
    df_piGen.loc[:, 'GenPip'] = mag(piGen)
    df_piGen.loc[:, 'GenPitheta'] = getTheta(piGen)
    df_piGen.loc[:, 'GenPiphi'] = getPhi(piGen)

    proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
    df_protonGen.loc[:, 'GenPp'] = mag(proGen)
    df_protonGen.loc[:, 'GenPtheta'] = getTheta(proGen)
    df_protonGen.loc[:, 'GenPphi'] = getPhi(proGen)

    gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
    df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
    df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
    df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

    gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
    debug = df_gammaGen.loc[:, 'GenGp2'] == mag(gamGen2)
    df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
    df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

    df_z = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
    df_z = pd.merge(df_z, df_piGen, how='inner', on='event')
    df_z = pd.merge(df_z, df_gammaGen, how='inner', on='event')



    # data frames and their keys to read X part
    df_electronRec = pd.DataFrame()
    df_protonRec = pd.DataFrame()
    df_gammaRec = pd.DataFrame()
    eleKeysRec = ["Epx", "Epy", "Epz", "Esector"]
    proKeysRec = ["Ppx", "Ppy", "Ppz", "Psector"]
    gamKeysRec = ["Gpx", "Gpy", "Gpz", "Gsector"]
    # read them
    for key in eleKeysRec:
        df_electronRec[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in proKeysRec:
        df_protonRec[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in gamKeysRec:
        df_gammaRec[key] = tree[key].array(library="pd", entry_stop=entry_stop)

    #convert data type to standard double
    df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float})
    df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float})
    df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float})

    #set up a dummy index for merging
    df_electronRec.loc[:,'event'] = df_electronRec.index
    df_protonRec.loc[:,'event'] = df_protonRec.index.get_level_values('entry')
    df_gammaRec.loc[:,'event'] = df_gammaRec.index.get_level_values('entry')
    df_gammaRec.loc[:,'GIndex'] = df_gammaRec.index.get_level_values('subentry')

    #save only FD protons and photons
    df_protonRec = df_protonRec[df_protonRec["Psector"]<7]
    df_gammaRec = df_gammaRec[df_gammaRec["Gsector"]<7]

    df_gg = pd.merge(df_gammaRec, df_gammaRec,
                        how='outer', on='event', suffixes=("", "2"))
    df_gg = df_gg[df_gg["GIndex"] < df_gg["GIndex2"]]
    df_ep = pd.merge(df_electronRec, df_protonRec, how='outer', on='event')

    df_epgg = pd.merge(df_ep, df_gg, how='outer', on='event')
    df_epgg = df_epgg[~np.isnan(df_epgg["Ppx"])]
    df_epgg = df_epgg[~np.isnan(df_epgg["Gpx"])]
    df_epgg = df_epgg[~np.isnan(df_epgg["Gpx2"])]

    return df_z, df_epgg
    

def saveDVpi0vars(df_epgg):

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

    pi0 = vecAdd(gam, gam2)
    VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
    v3l = cross(beam, ele)
    v3h = cross(pro, VGS)
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
    df_epgg.loc[:,'t'] = 2 * M * (df_epgg['Pe'] - M)
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
    
    df_math_epgg = df_epgg

    return df_math_epgg


"""
def makeDVpi0(self):
    #make dvpi0 pairs
    df_epgg = self.df_epgg

    df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

    cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
    cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
    cut_Q2 = df_epgg.loc[:, "Q2"] > 1  # Q2
    cut_W = df_epgg.loc[:, "W"] > 2  # W

    # Exclusivity cuts
    cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
    cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
    cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
    cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
    cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
    cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
    cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])

    df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_mmep & cut_meepgg &
                        cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]

    #For an event, there can be two gg's passed conditions above.
    #Take only one gg's that makes pi0 invariant mass
    #This case is very rare.
    #For now, duplicated proton is not considered.
    df_dvpi0.sort_values(by='closeness', ascending=False)
    df_dvpi0.sort_values(by='event')        
    df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]

    df_x = df_dvpi0.loc[:, ["event", "Epx", "Epy", "Epz", "Ep", "Ephi", "Etheta", "Ppx", "Ppy", "Ppz", "Pp", "Pphi", "Ptheta", "Gpx", "Gpy", "Gpz", "Gp", "Gtheta", "Gphi", "Gpx2", "Gpy2", "Gpz2", "Gp2", "Gtheta2", "Gphi2"]]
    self.df_x = df_x #done with saving x
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    parser.add_argument("-t","--test", help="use to enable testing flag", action='store_true',default=False)
    
    args = parser.parse_args()


    if args.test:
        #test_file = "tests/sample_radrec_1.root"
        test_file = "tests/merged_Fall_2018_Inbending_recon_10radtest.root"
        print("test enabled, using {}")
        args.fname = test_file

    fname_base = args.fname.split(".")[0]


    tree = readFile(args.fname)

    print(tree.keys())
    df_gen, df_rec = readEPGG(tree)
    #df_gen = readEPGG(tree)
    print(df_rec.shape)
    print(df_rec.head(20))
    # a = df_rec.query(
    # "W>2")

    # df_math = saveDVpi0vars(df_rec)
    # a = df_math.query(
    # "W>1")

    # hist = a.hist(bins=30)
    # plt.show()

    df_rec.to_pickle(fname_base+"_recon_recon.pkl")
    df_gen.to_pickle(fname_base+"_recon_gen.pkl")
