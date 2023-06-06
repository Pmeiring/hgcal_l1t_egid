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
from scipy.special import expit
import math


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS TO INITIATE AND EVALUATE BDTs

xgb_red = xg.Booster()

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

  xgb_red.load_model(in_xml.replace(".xml",".model"))

  # return initialised BDT and input variables
  return bdt_, in_var

norms = {
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

# cluster = {
#   'hoe': 0.698152,
#   'tkpt': 16.75,
#   'srrtot': 0.00421461,
#   'deta': 0.0174533,
#   'dpt': 0.301802,
#   'meanz': 339.497,
#   'dphi': 0.0436332,
#   'tkchi2': 496.659,
#   'tkz0': -0.75,
#   'tknstubs': 6,
# }


# cluster = {
#   'hoe': 0.360064,
#   'tkpt': 12.,
#   'srrtot': 0.00340373,
#   'deta': 0.0654498,
#   'dpt': 1.02128,
#   'meanz': 332.055,
#   'dphi': -0.0872665,
#   'tkchi2': 4.10412,
#   'tkz0': -10.15,
#   'tknstubs': 6,
# }

cluster = {
  'hoe': 0.00968323,
  'tkpt': 37.75,
  'srrtot': 0.00348815,
  'deta': 0.,
  'dpt': 0.920732,
  'meanz': 332.152,
  'dphi': 0.,
  'tkchi2': 8.56384,
  'tkz0': 0.7,
  'tknstubs': 6,
}

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

data_X = []
data_Y = []

# Evaluation: calculates BDT score for 3D cluster taking bdt as input
def evaluate_egid_BDT( _bdt, _bdt_var, in_cl3d, in_var_names ):
  # Loop over input vars and extract values from tree
  clustervars = []

  for var in in_var_names: 
    # var_beforenorm = getattr( in_cl3d, "%s"%var ) 
    var_beforenorm = cluster[var]
    min_ = norms[var][0]
    max_ = norms[var][1]
    var_afternorm  = (var_beforenorm-min_)/(max_-min_)
    var_scaled = var_afternorm * (max_-min_) + min_
    # print (var, var_beforenorm, var_afternorm, var_scaled)
    clustervars.append(var_afternorm)
    _bdt_var[var][0]=var_afternorm #getattr( in_cl3d, "%s"%var ) 
    # _bdt_var[var][0]=getattr( in_cl3d, "%s"%var ) 
  mvaScore = _bdt.EvaluateMVA("BDT")
  print ("SCORE MVA = ",mvaScore, expit(mvaScore))

  data_X.append((1 / (1 + math.sqrt((1-mvaScore)/(1+mvaScore)))))
  print (1 / (1 + math.sqrt((1-mvaScore)/(1+mvaScore))))

  egid_train_X_red = np.array([clustervars])
  # egid_train_X_red = np.array([[0.00108331765617, 0.00160359, 0.330632690505, 0.502882, 0.00186859, 0.0826819348174, 0.498303, 0.383274, 0.53681, 0]])

  # egid_train_X_red = np.array([[0.00108331765617, 0.00133331639745, 0.330632690505, 0.549001871135, 0.00157627187719, 0.0826819348174, 0.503513531681, 0.394672328994, 0.474901960784, 1.0]])
  label_ = np.array([1]*len(egid_train_X_red))
  test = xg.DMatrix( egid_train_X_red, label=label_, feature_names=in_var_names)
  test_pred = xgb_red.predict(test)
  print ("SCORE XGB = ",test_pred)
  data_Y.append(test_pred)

  WP_MVA = 0.5406244
  WP_XGB = (1 / (1 + math.sqrt((1-WP_MVA)/(1+WP_MVA))))

  print (WP_MVA, WP_XGB)

  # For few clusters that fail/pass, run evaluation here
  # Check if score is correct. 
  # Check if they pass the WP (MVA/XGB)


  #return BDT score
  return _bdt.EvaluateMVA("BDT")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def evaluate_egid_single(opt, egid_vars, eta_regions, f_sig, f_bkg, out):

  # subsample="test"
  # subsample="train"
  # subsample="full"

  procFileMap = {"signal":f_sig,"background":f_bkg}
  treeMap = {"signal":"sig_%s"%opt.dataset,"background":"bkg_%s"%opt.dataset}

  print ("~~~~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE ~~~~~~~~~~~~~~~~~~~~~~~~")

  # Extract bdt names from input list
  bdt_list = []
  for bdt in opt.bdts.split(","): bdt_list.append( bdt )

  # Check there is at least one input BDT
  if len(bdt_list) == 0: 
    print (" --> [ERROR] No input BDT. Leaving...")
    print ("~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
    sys.exit(1)

  # Check bdts exist (as xml files), if so then add to dict
  model_xmls = {}
  for bdt_name in bdt_list:
    #check if input variables for this bdt are defined
    if not bdt_name in egid_vars:
      print (" --> [ERROR] Input variables for BDT %s are not defined. Add key to egid_vars in training. Leaving..."%bdt_name)
      print ("~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
      sys.exit(1)
    # for reg in ['low','high']:
    for reg in eta_regions:
      if not os.path.exists("%s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)):
        print (" --> [ERROR] no xml file for BDT: %s. Leaving..."%bdt_name)
        print ("~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
        sys.exit(1)
      else:
        # passed checks: add xml to dict
        model_xmls[ "%s_%seta_%s"%(bdt_name,reg,opt.ptBin) ] = "%s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)


  for proc in ["background"]:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE INPUT NTUPLE
    # Define input ntuple
    f_in_name = procFileMap[ proc ] #"/afs/cern.ch/work/j/jheikkil/tools/ntuple-tools/ele23_copy.root"#"%s/cl3d_selection/%s/%s_%s_%s.root"%(os.environ['HGCAL_L1T_BASE'],opt.sampleType,opt.sampleType,opt.clusteringAlgo,opt.dataset)
    if not os.path.exists( f_in_name ):
      print (" --> [ERROR] Input ntuple %s does not exist. Leaving..."%f_in_name)
      print ("~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
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
    print (" --> Input ntuple %s read successfully"%f_in_name)

    # sys.exit()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE OUTPUT NTUPL
    f_out_name = "%s/%s_%s_%s_eval_%seta_%spt.root"%(out,proc,opt.clusteringAlgo,opt.dataset,opt.etaBin,opt.ptBin)

    # Variables to store in output ntuple #removed clusters_n, replace with nclu
    # out_var_names = ['pt','eta','phi','nclu','coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'seemax', 'sppmax', 'srrmax', 'meanz', 'emaxe', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90', 'hoe']
    out_var_names = ['pt','eta']
    # Add bdt score from TPG: i.e. one that was calculated in ntuple production
    out_var_names.append( "bdt_tpg" )
    # Add new bdt scores
    for bdt_name in bdt_list: out_var_names.append( "bdt_%s"%bdt_name )

    # Define dict to store output var
    out_var = {}
    for var in out_var_names: out_var[var] = array('f',[0.])
      
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
      #   print " --> Initialised BDT (%s) in %s eta region"%(b,reg)
      bdts["%s_%seta"%(b,opt.etaBin)], bdt_input_variables["%s_%seta"%(b,opt.etaBin)] = initialise_egid_BDT( model_xmls["%s_%seta_%s"%(b,opt.etaBin,opt.ptBin)], egid_vars[b] )
      print (" --> Initialised BDT (%s) in %s eta region"%(b,opt.etaBin))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EVALUATE BDTS: + store output variables in tree
    #Loop over clusters in input tree
    for ic,cl3d in enumerate(t_in):
      if ic==1: break
    
      #evaluate bdts
      for b in bdt_list:
   
        if "low" in eta_regions:
          #Low eta region: use low eta bdt
          if(abs(cl3d.eta) > eta_regions['low'][0])&(abs(cl3d.eta) <= eta_regions['low'][1]):
            out_var["bdt_%s"%b][0] = evaluate_egid_BDT( bdts["%s_loweta"%b], bdt_input_variables["%s_loweta"%b], cl3d, egid_vars[b] )
        elif "high" in eta_regions:
          #High eta region: use high eta bdt
          if(abs(cl3d.eta) > eta_regions['high'][0])&(abs(cl3d.eta) <= eta_regions['high'][1]):
            out_var["bdt_%s"%b][0] = evaluate_egid_BDT( bdts["%s_higheta"%b], bdt_input_variables["%s_higheta"%b], cl3d, egid_vars[b] )
        # Else: outside allowed eta range, give value of -999
        else: out_var["bdt_%s"%b][0] = -999.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CLOSE FILES
    f_in.Close()
    f_tmp.Close()
    os.system("rm tmpfile.root")

  print ("~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
# END OF EVALUATION FUNCTION

  # Scatter plot
  plt.scatter(data_X, data_Y, cmap='hot')

  x = np.array([0,1])
  y = np.array([0,1])
  plt.plot(x, y, color ="green")

  # Display the plot
  # plt.show()
  # plt.savefig( "%s/_Summary/test2.png"%(out) )

# Main function for running program
if __name__ == "__main__": evaluate_egid_single()


