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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

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

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
fs = filestruct.fs()
E = 10.604
cmap = plt.cm.jet  # define the colormap


def fit_function(x,A,B):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    #rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A*np.exp(x*B)


df = pd.read_pickle("interactive/dataArrays/full_xsection_outbending_rad_All_All_All_compare_c12_c6_bin_averages.pkl")
#df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/struct_funcsoutbending_rad_All_All_All_for_t_q_deps.pkl")

for col in df.columns:
    print(col)

#df.loc[:,"ratio"]= df['xsec_corr_red_nb']/df['dsdtdp']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

q2bins,xBbins, tbins, phibins = fs.q2bins[0:8], fs.xBbins[0:12], np.array(fs.tbins[2:11]), fs.phibins
#q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:11]), fs.xBbins[0:12], fs.q2bins[0:8], fs.phibins

#q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:9]), np.array(fs.xBbins[0:12]) ,np.array(fs.q2bins[0:8]), fs.phibins

compare_all_bins = False
int_across_phi = True

if compare_all_bins:


    reduced_plot_dir = "Comparison_plots/"

    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    #q2bins = [2,2.5]
    #xBbins = np.array([0.2,0.25])
    #tbins = np.array([0.09,0.15,0.2])
    phibins = np.array(phibins)

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            zs = []
            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                #query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                print(query)
                df_small = df.query(query)
                #print(df_small['xsec_corr_red_nb'])
                #print(df_small['dsdtdp'])


                cmap.set_bad(color='black')

                zs.append(df_small['ratio'].values)
                #zs.append(df_small['xsec_corr_red_nb'].values)

                # z = df_small['ratio'].values
                # z = np.expand_dims(z, axis=0)  # or axis=1
                # print(z)

                # x = phibins
                # y = tbins
                # fig, ax = plt.subplots(figsize =(36, 17)) 

                # print(x.size)
                # print(y.size)
                # print(z.size)

                # plt.rcParams["font.family"] = "Times New Roman"
                # plt.rcParams["font.size"] = "20"


                # plt.pcolormesh(x,y,z)#,cmap=cmap)#norm=mpl.colors.LogNorm())
                # #plt.imshow(z,interpolation='none')

                # plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
                # ax.set_xlabel('Lepton-Hadron Angle')
                # ax.set_ylabel('-t (GeV$^2)$')

                # plt.colorbar()

                # plt.show()

            #z = np.expand_dims(zs, axis=0)  # or axis=1
            z = zs
            print(z)

            x = phibins
            y = tbins
            fig, ax = plt.subplots(figsize =(36, 17)) 

            print(x.size)
            print(y.size)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"

            vmin,vmax = 0.5,1.5
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            plt.pcolormesh(x,y,z,norm=norm)#,cmap=cmap)#norm=mpl.colors.LogNorm())
            #plt.imshow(z,interpolation='none')

            plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
            ax.set_xlabel('Lepton-Hadron Angle')
            ax.set_ylabel('-t (GeV$^2)$')

            plt.colorbar()

            plt.savefig(reduced_plot_dir+"ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            plt.close()



if int_across_phi:

    reduced_plot_dir = "Comparison_plots_phi_int/"

    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    #q2bins = [2,2.5]
    #xBbins = np.array([0.2,0.25])
    #tbins = np.array([0.09,0.15,0.2])
    phibins = np.array(phibins)

    base_x_q = []
    base_x_q_6 = []

    xb_dep = []
    b_values_6 = []
    b_values_12 = []
    b_errs_6 = []
    b_errs_12 = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        #print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        base_x=[]
        base_x_6=[]

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            means_on_t_base = []
            means_on_t_base_6 = []
            t_errs = []
            t_errs_6 = []

            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                #query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                #print(query)
                df_small = df.query(query)
                #print(df_small['ratio'])
                #print(df_small['ratio'].mean())
                #means_on_t_base.append(df_small['ratio'].mean())

                means_on_t_base.append(df_small['xsec_corr_red_nb'].mean())
                means_on_t_base_6.append(df_small['dsdtdp'].mean())
                t_errs.append(df_small['uncert_xsec_corr_red_nb'].mean())
                t_errs_6.append(df_small['stat'].mean())
                
                # # # means_on_t_base.append(df_small['tel_c12'].mean())
                # # # means_on_t_base_6.append(df_small['telC6'].mean())
                # # # t_errs.append(df_small['mean_uncert_c12'].mean())
                # # # t_errs_6.append(df_small['tel-statC6'].mean())


            x = tbins[1:]#.tolist()
            y = np.array(means_on_t_base)#.tolist()
            y_6 = np.array(means_on_t_base_6)#.tolist()
            y_err_6 = np.array(t_errs_6)#.tolist()
            y_err = np.array(t_errs)#.tolist()



            valid = ~(np.isnan(x) | np.isnan(means_on_t_base) | np.isnan(t_errs))
            valid2 = ~(np.isnan(x) | np.isnan(y_6) | np.isnan(y_err_6))



            #print(x[valid])
            fit1, fit2 = False,False
            try:
                popt, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid],
                    sigma=y_err[valid], absolute_sigma=True)
                fit1 = True
            except:
                pass
            
            try:
                popt2, pcov2 = curve_fit(fit_function, xdata=x[valid2], ydata=y_6[valid2],
                    sigma=y_err_6[valid2], absolute_sigma=True)
                fit2 = True
            except:
                pass
            
            if fit1 and fit2:
                a,b = popt[0],popt[1]
                print(b)
                if b>-10:
                    a2,b2 = popt2[0],popt2[1]

                    a_err = np.sqrt(pcov[0][0])#*qmod
                    b_err = np.sqrt(pcov[1][1])#*qmod
                    a_err2 = np.sqrt(pcov2[0][0])#*qmod
                    b_err2 = np.sqrt(pcov2[1][1])#*qmod

                    #print(a/a2,b/b2)
                    #print("\n Q2 bin: {} to {}".format(qmin,qmax))
                    #print("xB bin: {} to {}".format(xmin,xmax))
                    #print(b/b2,np.sqrt(b_err*b_err+b_err2*b_err2))
                    #print(b,b_err)
                    #print(b2,b_err2)
                    xb_dep.append((xmin+xmax)/2)
                    b_values_6.append(-1*b2)
                    b_values_12.append(-1*b)
                    b_errs_6.append(b_err2)
                    b_errs_12.append(b_err)


                    xspace = np.linspace(0, 2, 1000)

                    fit_y_data_weighted_12 = fit_function(xspace,a,b)
                    fit_y_data_weighted_6 = fit_function(xspace,a2,b2)



                    fig, ax = plt.subplots(figsize =(36, 17)) 
                    plt.errorbar(x,means_on_t_base,yerr=t_errs,linestyle="None",marker="x",ms=12,color="red",label="CLAS12")
                    plt.errorbar(x,means_on_t_base_6,yerr=t_errs_6,linestyle="None",marker="x",ms=12,color="blue",label="CLAS6")  
                    fit2, = ax.plot(xspace, fit_y_data_weighted_6, color='blue', linewidth=2.5, label='CLAS6 Fit:')
                    #fit3, = ax.plot(xspace, fit_y_data_weighted_new, color='black', linewidth=2.5, label='New CLAS6 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel,tt,lt))
                    fit4, = ax.plot(xspace, fit_y_data_weighted_12, color='red', linewidth=2.5, label='CLAS12 Fit: ')
                    plt.legend()
                    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
                    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
                    plt.title("t Dep of xsection, Q2 = {}, xB = {}".format(qmin,xmin))
                    ax.set_yscale("log")
                    plt.show()

            

            base_x.append(means_on_t_base)
            base_x_6.append(means_on_t_base_6)
        base_x_q.append(base_x)
        base_x_q_6.append(base_x_6)


    fig, ax = plt.subplots(figsize =(36, 17)) 
    plt.errorbar(xb_dep,b_values_6,yerr=b_errs_6,linestyle="None",marker="x",ms=12,color="blue",label="CLAS6")
    plt.errorbar(xb_dep,b_values_12,yerr=b_errs_12,linestyle="None",marker="x",ms=12,color="red",label="CLAS12")    
    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
    plt.title("t Dep of xsection, Q2 = {}, xB = {}".format(qmin,xmin))
    #ax.set_yscale("log")
    plt.legend()
    plt.show()


    sys.exit()

    q_colors = ['red','orange','yellow','green','blue','purple','black','cyan','magenta','brown','pink','gray','olive','salmon','gold','teal','navy','indigo','maroon','lime','tan','aqua','darkgreen','darkblue','darkcyan','darkmagenta','darkred','darkorange','darkyellow','darkgreen','darkblue','darkpurple','darkcyan','darkmagenta','darkbrown','darkpink','darkgray','darkolive','darksalmon','darkgold','darkteal','darknavy','darkindigo','darkmaroon','darklime','darktan','darkaqua']
    #q_colors = ['black','purple','blue','green','yellow','orange','red']

    fig, ax = plt.subplots(figsize =(36, 17)) 

    q_labels = q2bins[0:-1]
    for q_count, bigarr in enumerate(base_x_q):
        color = q_colors[q_count]
        label = q_labels[q_count]
        legend_counter = 0
        for arr in bigarr:
            print("here")
            print(arr)
            if legend_counter == 0:
                plt.plot(tbins[0:-1],arr,color=color,label="-t: {}".format(label))
                legend_counter += 1
            else:
                plt.plot(tbins[0:-1],arr,color=color)

    # #FOR integraton over xB
    # q_labels = q2bins[0:-1]


    # for q_count, bigarr in enumerate(base_x_q):
    #     bigarr_6 = base_x_q_6[q_count]
    #     color = q_colors[q_count]
    #     label = q_labels[q_count]
    #     legend_counter = 0
    #     print('HEREE')
    #     print(bigarr)
    #     arr = np.nanmean(bigarr,axis=0)
    #     arr_6 = np.nanmean(bigarr_6,axis=0)

    #     print(arr)
    #     if legend_counter == 0:
    #         plt.plot(tbins[0:-1],arr,color=color,label="Q$^2$: {}".format(label))
    #         plt.plot(tbins[0:-1],arr_6,color=color,linestyle='--',)#,label="Q$^2_6$: {}".format(label))

    #         legend_counter += 1
    #     else:
    #         plt.plot(tbins[0:-1],arr,color=color)
    #         plt.plot(tbins[0:-1],arr_6,color=color,linestyle='--',)


    #FOR integraton over everything but 1
    # q_labels = q2bins[0:-1]

    # #for q_count, bigarr in enumerate(base_x_q):

    # arr1 = np.nanmean(base_x_q,axis=0)

    # arr = np.nanmean(arr1,axis=0)
    # print(arr)

    # q_count = 0
    # color = q_colors[q_count]
    # label = q_labels[q_count]

    # legend_counter = 1
    # print('HEREE')
   
    # print(arr)
    # if legend_counter == 0:
    #     plt.plot(tbins[0:-1],arr,color=color,label="-q: {}".format(label))
    #     legend_counter += 1
    # else:
    #     plt.plot(tbins[0:-1],arr,color=color)



    ax.set_yscale("log")
    plt.title("T dependence of Cross Section CLAS12 to CLAS6 Reduced Cross Sections")
    plt.xlabel('-t (GeV$^2)$')
    plt.ylabel('Ratio of CLAS12 to CLAS6 Reduced Cross Sections')
    plt.legend()
    plt.show()
                #print(df_small['dsdtdp'])


                #cmap.set_bad(color='black')

               # zs.append(df_small['ratio'].values)