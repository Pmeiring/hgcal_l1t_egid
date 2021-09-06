# Script to summarise performance of egid(s)
#  > Can be used to directly compare performance of trained and tpg egids
#  > Input: *_eval.root files, output from egid_evaluate.py
#  > Output: screen + .txt file defining working points

#usual imports
import ROOT
import numpy as np
import pandas as pd
import xgboost as xg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
import os
import sys
from array import array

# HARDCODED: working points would like to output
working_points = [0.995,0.975,0.95,0.9,0.8,0.7,0.6]

def leave():
  print "~~~~~~~~~~~~~~~~~~~~~ egid SUMMARY (END) ~~~~~~~~~~~~~~~~~~~~~"
  sys.exit(1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTION TO EXTACT PATH TO FILES
# def get_path( _i, _proc ): return "%s/training/results/%s/%s_%s_%s_eval_%s.root"%(os.environ['HGCAL_L1T_BASE'],_i[_proc],_i[_proc],_i['cl3d_algo'],_i['dataset'],_i['ptBin'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def summary_egid(opt, egid_vars, eta_regions, out):

  treeMap = {"signal":"sig_%s"%opt.dataset,"background":"bkg_%s"%opt.dataset}


  print "~~~~~~~~~~~~~~~~~~~~~~~~ egid SUMMARY ~~~~~~~~~~~~~~~~~~~~~~~~"

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # CONFIGURE INPUTS

  # Extract input info
  info = {}
  _i = opt.inputMap.split(",")
  if len(_i)!=4:
    print " --> [ERROR] Incorrect number of input elements. Please use format: <signalType>,<backgroundType>,<clustering Algo.>,<dataset [test,train,all]>"
    leave()
  info['signal'] = _i[0]
  info['background'] = _i[1]
  info['cl3d_algo'] = _i[2]
  info['dataset'] = _i[3]
  info['ptBin'] = opt.ptBin

  # Extract bdts names from input list and save plotting colour in map
  bdt_list = []
  bdt_colours = {}
  colors = ["red","blue","green","black","pink","gray"]
  i=0
  for bdt in opt.bdts.split(","): 
    bdt_name = bdt.split(":")[0]
    bdt_list.append( bdt_name )
    # bdt_colours[ bdt_name ] = bdt.split(":")[1] 
    bdt_colours[ bdt_name ] = colors[i]
    i+=1
  #Check there is atleast one input bdt
  if len(bdt_list) == 0:
    print " --> [ERROR] No input BDT. Leaving..."
    leave()

  # Define variables to store in dataFrame
  stored_vars = ["cl3d_eta","cl3d_pt"]
  for b in bdt_list: 
    if "tpg" in b: stored_vars.append( "cl3d_bdt_tpg" )
    else: stored_vars.append( "cl3d_bdt_%s"%b )

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # EXTRACT DATAFRAME FROM INPUT SIG AND BKG FILES
  frames = {}
  for proc in ['signal','background']:
    # Extract signal and background files
    # fpath = "%s/%s_%s_%s_eval_%s.root"%(out,proc,opt.clusteringAlgo,opt.dataset,opt.ptBin)
    fpath = "%s/%s_%s_%s_eval_%seta_%spt.root"%(out,proc,opt.clusteringAlgo,opt.dataset,opt.etaBin,opt.ptBin)
    # if not os.path.exists( fpath ):
    #   print " --> [ERROR] Input %s ntuple does not exists: %s. Please run egid_evaluate first! Leaving..."%(proc,get_path(info,proc))
    #   leave()
    iFile = ROOT.TFile( fpath )
    iTree = iFile.Get( treeMap[ proc ] )
    # Initialise new tree with frame variables
    _file = ROOT.TFile("tmp%s%s.root"%(opt.etaBin,opt.ptBin),"RECREATE")
    _tree = ROOT.TTree("tmp%s%s"%(opt.etaBin,opt.ptBin),"tmp%s%s"%(opt.etaBin,opt.ptBin))
    _vars = {}
    for var in stored_vars:
      _vars[var] = array('f',[-1.])
      _tree.Branch( '%s'%var, _vars[var], '%s/F'%var )
    # Loop over clusters in input tree and fill new tree
    for cl3d in iTree:
      for var in stored_vars: _vars[ var ][0] = getattr( cl3d, '%s'%var )
      _tree.Fill()
    # Convert tree to dataFrame
    dataTree, columnsTree = _tree.AsMatrix(return_labels=True)
    frames[proc] = pd.DataFrame( data=dataTree, columns=columnsTree )
    #frames[proc] = pd.DataFrame( tree2array(_tree) )
    del _file
    del _tree
    system('rm tmp%s%s.root'%(opt.etaBin,opt.ptBin))      

    # Add columns to dataFrame to label clusters
    frames[proc]['proc'] = proc
    # frames[proc]['type'] = info[proc]

  print " --> Extracted dataframes signal and background input ntuples"
  # Make one combined dataFrame
  frames_list = [frames["signal"].append(frames["background"])]
  frameTotal = pd.concat( frames_list, sort=False )

  # Split into eta regions
  frames_splitByEta = {}
  for reg in eta_regions: 
    frames_splitByEta[reg] = frameTotal[ abs(frameTotal['cl3d_eta']) > eta_regions[reg][0] ]
    frames_splitByEta[reg] = frames_splitByEta[reg][ abs(frames_splitByEta[reg]['cl3d_eta']) <= eta_regions[reg][1] ]

  print " --> Dataframes split into eta regions"
  print frames_splitByEta

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # DEFINE DICTS TO STORE EFFS FOR EACH BDT
  eff_signal = {}
  eff_background = {}
  bdt_points = {}
  wp_idx = {}

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # LOOP OVER BDTS CONFIGS
  for b in bdt_list:

    print " --> Calculating efficiencies for BDT: %s"%b

    # sort frame according to BDT score
    if "tpg" in b: bdt_var = "cl3d_bdt_tpg"
    else: bdt_var = "cl3d_bdt_%s"%b

    # Loop over eta regions
    for reg, fr in frames_splitByEta.iteritems():

      # Sort frame
      fr = fr.sort_values(bdt_var)

      # Create key name
      key = "%s_%s"%(b,reg) 

      # Initiate lists to store efficiencies
      eff_signal[key] = [1.]
      eff_background[key] = [1.]
      bdt_points[key] = [-9999.]
      wp_idx[key] = []

      # Total number of signal and bkg events in eta region
      N_sig_total, N_bkg_total = float(len(fr[fr['proc']=='signal'])), float(len(fr[fr['proc']=='background'])) 

      # Iterate over rows in frame and calc eff for given bdt_points
      N_sig_running, N_bkg_running = 0., 0.
      for index, row in fr.iterrows():
        # Add one to running counters depending on proc
        # print row['proc']

        if row['proc'] == 'signal': N_sig_running += 1.
        elif row['proc'] == 'background': N_bkg_running += 1.
        eff_s, eff_b = 1.-(N_sig_running/N_sig_total), 1.-(N_bkg_running/N_bkg_total)
        # Only add one entry for each bdt output value: i.e. if same as previous then remove last entry
        if row[bdt_var] == bdt_points[key][-1]:
          bdt_points[key] = bdt_points[key][:-1]
          eff_signal[key] = eff_signal[key][:-1] 
          eff_background[key] = eff_background[key][:-1] 
        # Add entry
        bdt_points[key].append( row[bdt_var] )
        eff_signal[key].append( eff_s )
        eff_background[key].append( eff_b )

      # print bdt_points
      # print eff_signal
      # print eff_background

      # Convert lists into numpy arrays
      bdt_points[key] = np.asarray(bdt_points[key])
      eff_signal[key] = np.asarray(eff_signal[key])
      eff_background[key] = np.asarray(eff_background[key])

      # Extract indices of working points
      for wp in working_points: wp_idx[key].append( abs((eff_signal[key]-wp)).argmin() )
      print " --> Extracted working points for BDT: %s, eta_region = %s"%(b,reg)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PRINT INFO TO USER
  # print " ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"  
  # print " --> INPUT: * signal          = %s"%info['signal']
  # print "            * background      = %s"%info['background']
  # print "            * cl3d_algo       = %s"%info['cl3d_algo']
  # print "            * dataset         = %s"%info['dataset']
  print " ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"  
  print ""
  print "   ~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~>,~.,~.,~.,~"
  for b in bdt_list:
    print "   --> BDT:   * discriminator = %s"%("_".join(b.split("_")[:-1]))
    print "              * config        = %s"%b.split("_")[-1]
    for reg in eta_regions:
      key = "%s_%s"%(b,reg)
      print ""
      print "   --> Eta region: %s --> %.2f < |eta| < %.2f"%(reg,eta_regions[reg][0],eta_regions[reg][1])
      print "      --> Working points:"
      for wp_itr in range(len(working_points)):
        wp = working_points[wp_itr]
        print "                  * At epsilon_s = %4.3f ::: BDT cut = %8.7f, epsilon_b = %5.4f"%(wp,bdt_points[key][wp_idx[key][wp_itr]],eff_background[key][wp_idx[key][wp_itr]])
    
    print "   ~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~.,~>,~.,~.,~.,~"
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # SAVE WORKING POINTS TO TXT FILE
  # only save if signal and background match what was used to train BDT
  # if not os.path.isdir("%s/_Summary"%out): os.system("%s/_Summary"%out)
  for b in bdt_list:

    # if( info['signal'] in b )&( info['background'] in b ):
    print " --> Saving working points to .txt files: %s"%b
    for reg in eta_regions:
      f_out = open("%s/_Summary/%s_%seta_%s_wp.txt"%(out,b,reg,opt.ptBin),"w")
      f_out.write("Working Points: %s\n"%b)
      key = "%s_%s"%(b,reg)
      f_out.write(" --> Eta region: %s\n"%reg)
      for wp_itr in range(len(working_points)):
        wp = working_points[wp_itr]
        f_out.write("          * At epsilon_s = %4.3f ::: BDT cut = %8.7f, epsilon_b = %5.4f\n"%(wp,bdt_points[key][wp_idx[key][wp_itr]],eff_background[key][wp_idx[key][wp_itr]]))
      f_out.close()
    # else: continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PLOT ROC CURVES
  if opt.outputROC:
    print " --> Plotting ROC curves"
    # Plot high and low eta regions separately
    plt_itr = 1
    for reg in eta_regions:
      plt.figure(plt_itr)
      for b in bdt_list:
        key = "%s_%s"%(b,reg)
        _label = b
        plt.plot( eff_signal[key], 1-eff_background[key], label=_label, color=bdt_colours[b] )
      plt.xlabel('Signal Eff. ($\epsilon_s$)')
      plt.ylabel('1 - Background Eff. ($1-\epsilon_b$)')
      plt.title('%.2f$ < |\eta| < $%.2f'%(eta_regions[reg][0],eta_regions[reg][1]))
      plt.grid(True)
      axes = plt.gca()
      axes.set_xlim([0.0,1.1])
      axes.set_ylim([0.0,1.1])
      plt.legend(bbox_to_anchor=(0.05,0.1), loc='lower left')
      plt.savefig( "%s/_Summary/ROC_%seta_%s.png"%(out,reg,opt.ptBin) )
      plt.savefig( "%s/_Summary/ROC_%seta_%s.pdf"%(out,reg,opt.ptBin) )
      axes.set_ylim([0.5,1.1])
      axes.set_xlim([0.5,1.1])
      plt.legend(bbox_to_anchor=(0.05,0.1), loc='lower left')
      plt.savefig( "%s/_Summary/ROC_%seta_%s_zoom.png"%(out,reg,opt.ptBin) )
      plt.savefig( "%s/_Summary/ROC_%seta_%s_zoom.pdf"%(out,reg,opt.ptBin) )

      plt_itr += 1
      # print " --> Saved plot: %s/plotting/plots/ROC_%seta_%s.(png/pdf)"%(os.environ['HGCAL_L1T_BASE'],reg,opt.ptBin)
      
  # leave()
# END OF SUMMARY FUNCTION

# Main function for running program
if __name__ == "__main__": summary_egid()
