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

from utils import filestruct
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

# For analysis flow
from make_dvpip_cuts import makeDVpi0
from bin_events import bin_df

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
Clas6_Sim_BeamTime = 11922445
Clas12_Sim_BeamTime = 16047494
#Clas12_exp_luminosity = 5.5E40 #Fall 2018 inbending
Clas12_exp_luminosity = 1.3978634388427737e+39 #Fall 2018 outbending
Clas12_exp_luminosity = 4.651647453735352e+40 #Fall 2018 outbending large sample size

fs = filestruct.fs()


generator_type = "rad"
generator_type = "norad"
mag_config = "in"
mag_config = "out"

datafile_base_dir = "/mnt/d/GLOBUS/CLAS12/APS2022/"
roots_dir = "raw_roots/"
raw_data_dir = "pickled_data/"


if generator_type == "rad":
    if mag_config == "in":
        path_to_exp_root = fs.path_to_exp_inbending_root
        path_to_rec_root = fs.path_to_rec_inbending_rad_root
        path_to_gen_root = fs.path_to_gen_inbending_rad_root
    elif mag_config == "out":
        path_to_exp_root = fs.path_to_exp_outbending_root
        path_to_rec_root = fs.path_to_rec_outbending_rad_root
        path_to_gen_root = fs.path_to_gen_outbending_rad_root
elif generator_type == "norad":
    if mag_config == "in":
        path_to_exp_root = fs.path_to_exp_inbending_root
        path_to_rec_root = fs.path_to_rec_inbending_norad_root
        path_to_gen_root = fs.path_to_gen_inbending_norad_root
    elif mag_config == "out":
        path_to_exp_root = fs.path_to_exp_outbending_root
        path_to_rec_root = fs.path_to_rec_outbending_norad_root
        path_to_gen_root = fs.path_to_gen_outbending_norad_root


print(datafile_base_dir+roots_dir+path_to_exp_root)
print(datafile_base_dir+roots_dir+path_to_rec_root)
print(datafile_base_dir+roots_dir+path_to_gen_root)



sys.exit()

#run_identifiyer = "_CD_Included"
#run_identifiyer = "_CD_ONLY"
#run_identifiyer = "_Sangbaek_rad_CD_sim"
#run_identifiyer = "_rad_bkmrg"
run_identifiyer = "_outbending"

#### Naming constants:
dvpip_data_dir = "pickled_dvpip/"
binned_data_dir = "binned_dvpip/"
final_xsec_dir = "final_data_files/"

exp_data_name = "f18_in_exp"
#rec_data_name = "f18_in_rec"
#rec_data_name = "f18_bkmrg_in_rec"
rec_data_name = "f18_bkmrg_in_rec_with_cd"
#rec_data_name = "sangbaek_sim_rec_with_cd"


gen_data_name = "f18_bkmrg_in_gen"
#gen_data_name  = "genOnly_sangbaek_sim_genONLY"
merged_data_name = "merged_total"
final_output_name = "full_xsection"

exp_common_name = "fall 2018 inbending"
rec_common_name = "sim rec rad inbending"


#exp_data_name = "f18_outbending_exp"
exp_data_name = "exp_outbending_183_no_logi"
rec_data_name = "1933_rad_f18_outbending_recon"
gen_data_name = "1933_rad_f18_outbending_gen"
exp_common_name = "fall 2018 outbending"
rec_common_name = "sim rec rad outbending"


bin_all_events = True
make_exclusive_cuts = True
bin_gen = True

# merged_data_name = "merged_total_rad_outb"
# final_output_name = "full_xsection_rad_outb"


def get_gamma(x,q2,BeamE):
    a8p = 1/137*(1/(8*3.14159))
    energies = [BeamE]
    for e in energies:
        y = q2/(2*x*e*mp)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]


def expand_clas6(df):
    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins
    print(df)

    qrange = [q2bins[0], q2bins[-1]]
    xBrange = [xBbins[0], xBbins[-1]]
    trange = [tbins[0], tbins[-1]]

    data_vals = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        query = "qmin=={}".format(qmin)
        df_q = df.query(query)

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            print("        xB bin: {} to {}".format(xmin,xmax))
            query = "xmin=={}".format(xmin)
            df_qx = df_q.query(query)

            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
                #print("                 t bin: {} to {}".format(tmin,tmax))
                query = "tmin=={}".format(tmin)
                df_qxt = df_qx.query(query)

                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    #print("                             p bin: {} to {}".format(pmin,pmax))
                    query = "pmin=={}".format(pmin)
                    df_qxtp =  df_qxt.query(query)
                    if df_qxtp.empty:
                        data_vals.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,qmin,xmin,tmin,pmin,qmax,xmax,tmax,pmax])
                    else:                    
                        list_value = [df_qxtp["q"].values[0],df_qxtp["x"].values[0],df_qxtp["t"].values[0],df_qxtp["p"].values[0],df_qxtp["dsdtdp"].values[0],df_qxtp["stat"].values[0],df_qxtp["sys"].values[0],qmin,xmin,tmin,pmin,qmax,xmax,tmax,pmax]
                        print(list_value)
                        data_vals.append(list_value)

    df_spaced = pd.DataFrame(data_vals, columns = ['q','x','t','p','dsdtdp','stat','sys','qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax'])
    # df_minibin = pd.DataFrame(num_counts, columns = ['qmin','xmin','tmin','pmin','qave','xave','tave','pave',prefix+'counts'])
    # print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
    # print("Total number of original events: {}".format(total_num))
    return df_spaced


if make_exclusive_cuts:
    #### APPLY EXCLUSIVITY CUTS
    print(
        "Applying exclusive cuts to dataframe..."
    )

    ########################################

    print(raw_data_dir+rec_data_name+".pkl")
    df_exp_epgg = pd.read_pickle(raw_data_dir+exp_data_name+".pkl")
    df_rec_epgg = pd.read_pickle(raw_data_dir+rec_data_name+".pkl")
    print("There are {} exp epgg events".format(df_exp_epgg.shape[0]))
    print("There are {} rec epgg events".format(df_rec_epgg.shape[0]))

    # x_data = df_exp_epgg["Pphi"]
    # y_data = df_exp_epgg["Ptheta"]

    # # df_gen = pd.read_pickle(raw_data_dir+"f18_bkmrg_in_gen.pkl")
    # # y_data = df_gen["GenPtheta"]
    # # x_data = df_gen["GenPphi"]

    # # var_names = ["phi","theta"]
    # # ranges = [[-360,360,100],[0,100,100]]
    # # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
    # #             saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
    # #             filename="ExamplePlot",units=["",""],extra_data=None)

    # # sys.exit()


    df_dvpip_exp = makeDVpi0(df_exp_epgg)
    df_dvpip_rec = makeDVpi0(df_rec_epgg)

    print(df_dvpip_exp.columns)
    print(df_dvpip_rec.columns)

    print("There are {} exp dvpip events".format(df_dvpip_exp.shape[0]))
    print("There are {} rec dvpip events".format(df_dvpip_rec.shape[0]))


    title_dir = "plots/{}/".format(exp_data_name)


# try: 
#     os.mkdir(title_dir) 
# except OSError as error: 
#     print(error)  
#histo_plotting.make_all_histos(df_dvpip_exp,datatype="Recon",hists_2d=False,hists_1d=True,hists_overlap=False,saveplots=True,output_dir = title_dir)

# # title_dir = "plots/{}/".format(rec_data_name)

# # #histo_plotting.make_all_histos(df_dvpip_rec,datatype="Recon",hists_2d=True,hists_1d=True,hists_overlap=False,saveplots=True,output_dir = title_dir)

# # title_dir = "plots/{}_{}_extracuts/".format(exp_data_name,rec_data_name)

# # print("Making histograms... {}".format(title_dir))
# # histo_plotting.make_all_histos(df_dvpip_exp,datatype="Recon",hists_2d=False,hists_1d=False,hists_overlap=True,saveplots=True,output_dir = title_dir,
# #                                 df_2=df_dvpip_rec,first_label=exp_common_name,second_label=rec_common_name)











# # df_dvpip_exp = df_dvpip_exp.query("Q2>2 and Q2<2.5 and xB>0.3 and xB<0.38 and t1>0.2 and t1<0.3")
# # df_dvpip_rec = df_dvpip_rec.query("Q2>2 and Q2<2.5 and xB>0.3 and xB<0.38 and t1>0.2 and t1<0.3")

# #df_exp_epgg = df_exp_epgg.query("Q2>2 and Q2<2.5 and xB>0.3 and xB<0.38 and t1>0.2 and t1<0.3 and ME_epgg<0.7")# and W>2")
# #df_rec_epgg = df_rec_epgg.query("Q2>2 and Q2<2.5 and xB>0.3 and xB<0.38 and t1>0.2 and t1<0.3 and ME_epgg<0.7")# and W>2")

# #print(df_dvpip_exp.columns)
# #y_data = df["Ptheta"]
# # x_data = df_exp_epgg["Mpi0"]
# # x_data2 = df_rec_epgg["Mpi0"]
# x_data = df_exp_epgg["W"]
# x_data2 = df_rec_epgg["W"]

# #x_data2 = df_dvpip_rec["Gp"]
# #y_data = df_dvpip_exp["Ptheta"]
# y_data = df_rec_epgg["Gp"]

# var_names = ["Gp2","Gp1"]
# vars = ["Gp"]
# ranges = [[0,0.3,100],[0,0.3,100]]
# # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
# #            saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
# #            filename="ExamplePlot",units=["",""],extra_data=None)

# ranges = [0,12,100]

# #ranges = [0,0.3,100]


# make_histos.plot_1dhist(x_data,vars,ranges=ranges,second_x=True,second_x_data=x_data2,logger=False,first_label="exp Gp",second_label="rec Gp",
#              saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False)

# sys.exit()


# df_dvpip_exp = pd.read_pickle(dvpip_data_dir+exp_data_name+run_identifiyer+"_dvpip"+".pkl")
# df_dvpip_rec = pd.read_pickle(dvpip_data_dir+exp_data_name+run_identifiyer+"_dvpip"+".pkl")


#### BIN EVENTS
if bin_all_events:
    print(
        "Binning events..."
    )

    if bin_gen:
        df_gen = pd.read_pickle(raw_data_dir + gen_data_name+".pkl")
        print("There are {} gen dvpip events".format(df_gen.shape[0]))
        print(df_gen.columns)
        df_gen = df_gen.query("GenQ2>1 and GenW>2")

        print("There are {} gen dvpip events".format(df_gen.shape[0]))
        

        df_gen_binned = bin_df(df_gen, "Gen")
        df_gen_binned.to_pickle(binned_data_dir + gen_data_name+"_binned"+".pkl")

    df_exp_binned = bin_df(df_dvpip_exp, "exp")
    df_rec_binned = bin_df(df_dvpip_rec, "rec")





    df_exp_binned.to_pickle(binned_data_dir+exp_data_name+run_identifiyer+"_binned"+".pkl")
    df_rec_binned.to_pickle(binned_data_dir+rec_data_name+run_identifiyer+"_binned"+".pkl")

    df_exp = df_exp_binned
    df_rec = df_rec_binned
    df_gen = df_gen_binned



    """
    Need to add logic case for when need to bin over gen also
    """


    # COMBINE INTO ONE DATAFRAME

    # Load relevant dataframes:

else:
    df_gen = pd.read_pickle(binned_data_dir + gen_data_name+"_binned"+".pkl")
    df_exp = pd.read_pickle(binned_data_dir + exp_data_name+run_identifiyer+"_binned"+".pkl")
    df_rec = pd.read_pickle(binned_data_dir + rec_data_name+run_identifiyer+"_binned"+".pkl")
                            


df_energy = pd.read_pickle(binned_data_dir + "EnergyDependenceRatio.pkl")

space_clas6 = False
if space_clas6:
    df_clas6 = pd.read_pickle(binned_data_dir + "xs_clas6_binned.pkl")
    df_clas6 = expand_clas6(df_clas6)
    df_clas6.to_pickle(base_dir + "xs_clas6_binned_expanded.pkl")
else:
    df_clas6 = pd.read_pickle(binned_data_dir + "xs_clas6_binned_expanded.pkl")



df_exp = df_exp.rename(columns={"qave": "qave_exp", "xave": "xave_exp","tave": "tave_exp", "pave": "pave_exp","counts":"counts_exp"})
df_rec = df_rec.rename(columns={"qave": "qave_rec", "xave": "xave_rec","tave": "tave_rec", "pave": "pave_rec","counts":"counts_rec"})
df_gen = df_gen.rename(columns={'qave': 'qave_gen', 'xave': 'xave_gen', 'tave': 'tave_gen', 'pave': 'pave_gen', 'Gencounts': 'counts_gen'})
df_energy = df_energy.rename(columns={"counts_high": "counts_10600GeV", "counts_low": "counts_5776GeV"})



for df in [df_exp, df_rec, df_gen, df_clas6, df_energy]:
    print(df.shape)
    print(df.columns)



df_merged_1 = pd.merge(df_exp,df_rec,how='inner', on=['qmin','xmin','tmin','pmin'])
df_merged_2 = pd.merge(df_gen,df_energy,how='inner', on=['qmin','xmin','tmin','pmin'])
df_merged_3 = pd.merge(df_merged_1,df_merged_2,how='inner', on=['qmin','xmin','tmin','pmin'])
df_merged_total = pd.merge(df_merged_3,df_clas6,how='inner', on=['qmin','xmin','tmin','pmin'])


print(df_merged_total)
df_merged_total.to_pickle(binned_data_dir + merged_data_name+run_identifiyer+".pkl")


# Calc x-section:


base_dir = binned_data_dir
# df = pd.read_pickle(base_dir + "merged_total_noseccut.pkl")
df = df_merged_total


#print(df.columns)
# Columns:
#Index(['qmin', 'xmin', 'tmin', 'pmin', 'qave_exp', 'xave_exp', 'tave_exp',
#       'pave_exp', 'counts_exp', 'qave_rec', 'xave_rec', 'tave_rec',
#       'pave_rec', 'counts_rec', 'qave_gen', 'xave_gen', 'tave_gen',
#       'pave_gen', 'counts_gen', 'counts_10600GeV', 'counts_5776GeV',
#       'Energy_ratio', 'q', 'x', 't', 'p', 'dsdtdp', 'stat', 'sys'],

df.loc[:,"gamma_exp"] = get_gamma(df["xave_exp"],df["qave_exp"],10.604)[0] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[0]
df.loc[:,"epsi_exp"] =  get_gamma(df["xave_exp"],df["qave_exp"],10.604)[1] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]

df.loc[:,"gamma6_sim"] = get_gamma((df["xmin"]+df["xmax"])/2,(df["qmin"]+df["qmax"])/2,5.776)[0]
df.loc[:,"gamma12_sim"] = get_gamma((df["xmin"]+df["xmax"])/2,(df["qmin"]+df["qmax"])/2,10.604)[0]


df.loc[:,"xsec_sim_12"] = df["counts_10600GeV"]/Clas12_Sim_BeamTime/df["gamma12_sim"]
df.loc[:,"xsec_sim_6"] = df["counts_5776GeV"]/Clas6_Sim_BeamTime/df["gamma6_sim"]


df.loc[:,"xsec_ratio_sim"] = df["xsec_sim_12"]/df["xsec_sim_6"]


df.loc[:,"binvol"] = (df["qmax"]-df["qmin"])*(df["xmax"]-df["xmin"])*(df["tmax"]-df["tmin"])*(df["pmax"]-df["pmin"])*3.14159/180


df.loc[:,"acc_corr"] = df["counts_rec"]/df["counts_gen"]


df.loc[:,"xsec"] = df["counts_exp"]/Clas12_exp_luminosity/df["binvol"]
df.loc[:,"xsec_corr"] = df["xsec"]/df["acc_corr"]
df.loc[:,"xsec_corr_red"] = df["xsec_corr"]/df["gamma_exp"]
df.loc[:,"xsec_corr_red_nb"] = df["xsec_corr_red"]*1E33

df.loc[:,"xsec_ratio_exp"] = df["xsec_corr_red_nb"]/df["dsdtdp"]

df.loc[:,"xsec_ratio_exp_corr"] = df["xsec_ratio_exp"]/df["xsec_ratio_sim"]





df.loc[:,"uncert_counts_exp"] = np.sqrt(df["counts_exp"])
df.loc[:,"uncert_counts_rec"] = np.sqrt(df["counts_rec"])
df.loc[:,"uncert_counts_gen"] = np.sqrt(df["counts_gen"])
df.loc[:,"uncert_counts_10600GeV"] = np.sqrt(df["counts_10600GeV"])
df.loc[:,"uncert_counts_5776GeV"] = np.sqrt(df["counts_5776GeV"])


df.loc[:,"uncert_xsec"] = df["uncert_counts_exp"]/df["counts_exp"]*df["xsec"]
df.loc[:,"uncert_acc_corr"] = np.sqrt(  np.square(df["uncert_counts_rec"]/df["counts_rec"]) + np.square(df["uncert_counts_gen"]/df["counts_gen"]))*df["acc_corr"]
df.loc[:,"uncert_xsec_corr_red_nb"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_red_nb"]

df.loc[:,"uncert_xsec_ratio_exp"] = np.sqrt(  np.square(df["uncert_xsec_corr_red_nb"]/df["xsec_corr_red_nb"]) + np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]) )*df["xsec_ratio_exp"]

df.loc[:,"uncert_xsec_ratio_exp_corr"] =  np.sqrt(  np.square(df["uncert_xsec_ratio_exp"]/df["xsec_ratio_exp"]) + np.square(df["uncert_counts_10600GeV"]/df["counts_10600GeV"]) + np.square(df["uncert_counts_5776GeV"]/df["counts_5776GeV"]) )*df["xsec_ratio_exp_corr"]


#df.loc[:,"xsec_ratio_exp_corr"] = df["xsec_ratio_exp_corr"].replace([np.inf, -np.inf], np.nan)

#df = df[df['xsec_ratio_exp_corr'].notna()]
#print(df)

#df.to_csv("out.csv")


df.to_pickle(final_xsec_dir+final_output_name+run_identifiyer+".pkl")
print("Output pickle file save to {}".format(final_xsec_dir+final_output_name+run_identifiyer+".pkl"))


