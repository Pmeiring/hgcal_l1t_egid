# Script to evaluate newly trained egid(s)
#  > Input: selected clusters in cl3d_selection directory
#  > Output: copy of ntuples + new BDT scores
#  > Can evaluate multiple BDTs, just need xml input

#usual imports
import ROOT
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
import os
import sys
from array import array
# import utils.correlator_common as cc


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS TO INITIATE AND EVALUATE BDTs

# Initialisation: returns BDT and dict of input variables. Takes xml file name as input
def initialise_egid_BDT( in_xml, in_var_names ):
  # book mva reader with input variables
  in_var = {}
  for var in in_var_names: in_var[var] = array( 'f', [0.] )
  # initialise TMVA reader and add variables
  bdt_ = ROOT.TMVA.Reader()
  for var in in_var_names: bdt_.AddVariable( var, in_var[var] )
  # book mva with xml file
  bdt_.BookMVA( "BDT", in_xml )
  # return initialised BDT and input variables
  return bdt_, in_var

# # Evaluation: calculates BDT score for 3D cluster taking bdt as input
# def evaluate_egid_BDT( _bdt, _bdt_var, in_cl3d, in_var_names ):
#   # Loop over input vars and extract values from tree
#   for var in in_var_names: _bdt_var[var][0]=getattr( in_cl3d, "%s"%var ) 
#   #return BDT score
#   return _bdt.EvaluateMVA("BDT")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

norms_highpt = {
  'hoe': [-1.0, 1566.547607421875],
  'tkpt': [1.9501149654388428, 11102.0048828125],
  'srrtot': [0.0, 0.01274710614234209],
  'deta': [-0.24224889278411865, 0.23079538345336914],
  'dpt': [0.010325592942535877, 184.92538452148438],
  'meanz': [325.0653991699219, 499.6089782714844],
  'dphi': [-6.281332015991211, 6.280326843261719],
  'tkchi2': [0.024048099294304848, 1258.37158203125],
  'tkz0': [-14.94140625, 14.94140625],
  'tknstubs': [4.0, 6.0],
}

norms_lowpt = {
  'hoe': [-1.0, 909.2194213867188],
  'coreshowerlength': [1.0, 26.0],
  'tkpt': [1.9501149654388428, 11102.0048828125],
  'srrtot': [0.0, 0.013055052608251572],
  'deta': [-0.24158358573913574, 0.2402184009552002],
  'dpt': [0.07829763740301132, 1558.2740478515625],
  'meanz': [322.10272216796875, 504.7358093261719],
  'dphi': [-6.280882835388184, 6.28144645690918],
  'tkchi2': [0.00784884113818407, 1252.6024169921875],
  'tkz0': [-14.94140625, 14.94140625],
  'tknstubs': [4.0, 6.0],
}

# Evaluation: calculates BDT score for 3D cluster taking bdt as input
def evaluate_egid_BDT(b, _bdt, _bdt_var, in_cl3d, in_var_names ):

  # Loop over input vars and extract values from tree  
  norm = norms_lowpt if "_lowpt" in b else norms_highpt

  for var in in_var_names: 
    if b=="allAvailVars_best3cl_alltrk":
      if var=="tkchi2":
        var_beforenorm = getattr( in_cl3d, "tkChi2" ) 
      elif var=="tknstubs": 
        var_beforenorm = getattr( in_cl3d, "nstubs")
      else:
        var_beforenorm = getattr( in_cl3d, "%s"%var ) 
    else:
      var_beforenorm = getattr( in_cl3d, "%s"%var ) 
    # min_ = norm[var][0]
    # max_ = norm[var][1]
    # var_afternorm  = (var_beforenorm-min_)/(max_-min_)
    # var_scaled = var_afternorm * (max_-min_) + min_
    # print var, var_beforenorm, var_afternorm, var_scaled

    _bdt_var[var][0]=var_beforenorm
    # _bdt_var[var][0]=var_afternorm
    # if var=="hoe":_bdt_var[var][0]=cc.makeHoe(var_beforenorm)
    # if var=="tkpt":_bdt_var[var][0]=cc.makePtFromFloat(var_beforenorm)
    # if var=="srrtot":_bdt_var[var][0]=cc.makeSrrTot(var_beforenorm)
    # if var=="deta":_bdt_var[var][0]=cc.makeEta(var_beforenorm)
    # if var=="dpt":_bdt_var[var][0]=cc.makePtFromFloat(var_beforenorm)
    # if var=="meanz":_bdt_var[var][0]=cc.makeMeanZ(var_beforenorm)
    # if var=="dphi":_bdt_var[var][0]=cc.makePhi(var_beforenorm)
    # if var=="tkchi2":_bdt_var[var][0]=cc.makeChi2(var_beforenorm)
    # if var=="tkz0":_bdt_var[var][0]=cc.makeZ0(var_beforenorm)
    # if var=="tknstubs": _bdt_var[var][0]=var_beforenorm

  #return BDT score
  # print (_bdt[0].EvaluateMVA("BDT"))
  return _bdt.EvaluateMVA("BDT")

def evaluate_egid(opt, egid_vars, eta_regions, f_sig, f_bkg, out):

  # subsample="test"
  # subsample="train"
  # subsample="full"

  procFileMap = {"signal":f_sig,"background":f_bkg}
  treeMap = {"signal":"sig_%s"%opt.dataset,"background":"bkg_%s"%opt.dataset}

  print ( "~~~~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE ~~~~~~~~~~~~~~~~~~~~~~~~")

  # Extract bdt names from input list
  bdt_list = []
  for bdt in opt.bdts.split(","): bdt_list.append( bdt )

  # Check there is at least one input BDT
  if len(bdt_list) == 0: 
    print ( " --> [ERROR] No input BDT. Leaving...")
    print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
    sys.exit(1)

  # Check bdts exist (as xml files), if so then add to dict
  model_xmls = {}
  for bdt_name in bdt_list:
    #check if input variables for this bdt are defined
    if not bdt_name in egid_vars:
      print ( " --> [ERROR] Input variables for BDT %s are not defined. Add key to egid_vars in training. Leaving..."%bdt_name)
      print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
      sys.exit(1)
    # for reg in ['low','high']:
    for reg in eta_regions:
      if not os.path.exists("%s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)):
        print ( " --> [ERROR] no xml file for BDT: %s. Leaving..."%bdt_name)
        print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
        sys.exit(1)
      else:
        # passed checks: add xml to dict
        model_xmls[ "%s_%seta_%s"%(bdt_name,reg,opt.ptBin) ] = "%s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)


  for proc in ["signal","background"]:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE INPUT NTUPLE
    # Define input ntuple
    f_in_name = procFileMap[ proc ] #"/afs/cern.ch/work/j/jheikkil/tools/ntuple-tools/ele23_copy.root"#"%s/cl3d_selection/%s/%s_%s_%s.root"%(os.environ['HGCAL_L1T_BASE'],opt.sampleType,opt.sampleType,opt.clusteringAlgo,opt.dataset)
    if not os.path.exists( f_in_name ):
      print ( " --> [ERROR] Input ntuple %s does not exist. Leaving..."%f_in_name)
      print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
      sys.exit(1)

    # Extract trees
    f_in = ROOT.TFile.Open( f_in_name )
    t_in = ROOT.TTree() 
    if opt.dataset=="all": # add the test and train samples
      treelist = ROOT.TList()
      t_tmp1 = f_in.Get( treeMap[proc].replace("all","test") )
      t_tmp2 = f_in.Get( treeMap[proc].replace("all","train") )
      treelist.Add( t_tmp1 )
      treelist.Add( t_tmp2 )
      f_tmp = ROOT.TFile.Open("tmpfile.root","RECREATE")
      t_in = ROOT.TTree.MergeTrees(treelist)
    else:
      t_in = f_in.Get( treeMap[ proc ] )
    print ( " --> Input ntuple %s read successfully"%f_in_name)

    # sys.exit()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE OUTPUT NTUPL
    f_out_name = "%s/%s_%s_%s_eval_%seta_%spt.root"%(out,proc,opt.clusteringAlgo,opt.dataset,opt.etaBin,opt.ptBin)

    # Variables to store in output ntuple #removed clusters_n, replace with nclu
    # out_var_names = ['pt','eta','phi','nclu','coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'seemax', 'sppmax', 'srrmax', 'meanz', 'emaxe', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90', 'hoe']
    out_var_names = ['pt','eta']
    # Add bdt score from TPG: i.e. one that was calculated in ntuple production
    # out_var_names.append( "bdt_tpg" )
    # Add new bdt scores
    for bdt_name in bdt_list: out_var_names.append( "bdt_%s"%bdt_name )

    # Define dict to store output var
    out_var = {}
    for var in out_var_names: out_var[var] = array('f',[0.])
    
    #Open file: check if already exists (if so ask user if they want to rewrite)
    if os.path.exists( f_out_name ):
      recreate = 1 #input("Output file %s already exists. Do you want to write over file [yes=1,no=0]:"%f_out_name)
      if not recreate:
        print ( " --> Move %s to a new folder then run again. Leaving..."%f_out_name)
        print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
        sys.exit(1)
    
    f_out = ROOT.TFile.Open( f_out_name, "RECREATE" )
    t_out = ROOT.TTree( treeMap[proc], treeMap[proc] )
      
    #Add branches to tree
    for var_name, var in out_var.items(): t_out.Branch("%s"%var_name, var, "cl3d_%s/F"%var_name) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # INITIALISE BDTS
    bdts = {}
    bdt_input_variables = {} #dict of dicts to store input var for each bdt
    # Loop over bdts
    for b in bdt_list:
      # Loop over eta regions
      # for reg in ['low','high']:
      # for reg in eta_regions:
      #   bdts["%s_%seta"%(b,reg)], bdt_input_variables["%s_%seta"%(b,reg)] = initialise_egid_BDT( model_xmls["%s_%seta_%s"%(b,reg,opt.ptBin)], egid_vars[b] )
      #   print ( " --> Initialised BDT (%s) in %s eta region"%(b,reg))
      bdts["%s_%seta"%(b,opt.etaBin)], bdt_input_variables["%s_%seta"%(b,opt.etaBin)] = initialise_egid_BDT( model_xmls["%s_%seta_%s"%(b,opt.etaBin,opt.ptBin)], egid_vars[b] )
      print ( " --> Initialised BDT (%s) in %s eta region"%(b,opt.etaBin))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EVALUATE BDTS: + store output variables in tree
    #Loop over clusters in input tree
    for cl3d in t_in:
    
      #evaluate bdts
      for b in bdt_list:
   
        if "low" in eta_regions:
          #Low eta region: use low eta bdt
          if(abs(cl3d.eta) > eta_regions['low'][0])&(abs(cl3d.eta) <= eta_regions['low'][1]):
            out_var["bdt_%s"%b][0] = evaluate_egid_BDT(b, bdts["%s_loweta"%b], bdt_input_variables["%s_loweta"%b], cl3d, egid_vars[b] )
        elif "high" in eta_regions:
          #High eta region: use high eta bdt
          if(abs(cl3d.eta) > eta_regions['high'][0])&(abs(cl3d.eta) <= eta_regions['high'][1]):
            out_var["bdt_%s"%b][0] = evaluate_egid_BDT(b, bdts["%s_higheta"%b], bdt_input_variables["%s_higheta"%b], cl3d, egid_vars[b] )
        # Else: outside allowed eta range, give value of -999
        else: out_var["bdt_%s"%b][0] = -999.

      # Add all other variables to output ntuple
      for var in out_var_names[:-1*len(bdt_list)]:
        if "bdt_tpg" in var: out_var[var][0] = getattr(cl3d,"bdteg")
        else: out_var[var][0] = getattr(cl3d,"%s"%var)

      # Write cluster with new BDT scores to tree
      t_out.Fill()

    #END of loop over clusters
    print ( " --> Evaluated BDT scores and saved in output: %s"%f_out_name)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CLOSE FILES
    f_in.Close()
    f_tmp.Close()
    os.system("rm tmpfile.root")
    f_out.Write()
    f_out.Close()

  print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
# END OF EVALUATION FUNCTION

# Main function for running program
if __name__ == "__main__": evaluate_egid()


