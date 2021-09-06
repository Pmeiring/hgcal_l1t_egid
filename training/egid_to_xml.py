# Script for converting eid xgboost model to xml file (to be used directly in TPG software)

#usual imports
import numpy as np
import xgboost as xg
import pickle
import pandas as pd
import ROOT as r
#from root_numpy import tree2array, testdata, list_branches, fill_hist
from os import system, path
import os
import sys
from optparse import OptionParser

# Extract input variables to BDT from egid_training.py: if BDT config not defined there then will fail
# from egid_fullProcedureBDT import egid_vars

# Configure options
# def get_options():
#   parser = OptionParser()
#   parser.add_option('--clusteringAlgo', dest='clusteringAlgo', default='Histomaxvardr', help="Clustering algorithm with which to optimise BDT" )
#   # parser.add_option('--signalType', dest='signalType', default='electron_200PU', help="Input signal type" )
#   # parser.add_option('--backgroundType', dest='backgroundType', default='neutrino_200PU', help="Input background type" )
#   parser.add_option('--bdtConfig', dest='bdtConfig', default='full', help="BDT config (accepted values: baseline/full)" )
#   parser.add_option('--ptBin', dest='ptBin', default='default', help="Used pT bin (accepted values: default, low)" )

#   return parser.parse_args()

# (opt,args) = get_options()

# Function to convert model into xml
def egid_to_xml(opt, egid_vars, eta_regions, out):

  print "~~~~~~~~~~~~~~~~~~~~~~~~ egid TO XML ~~~~~~~~~~~~~~~~~~~~~~~~"

  #Define BDT name
  bdt_name = opt.bdtConfig

  # Check if input vars for BDT name are defined
  if not bdt_name in egid_vars: 
    print " --> [ERROR] Input variables for BDT %s are not defined. Add key to egid_vars dict. Leaving..."%bdt_name
    print "~~~~~~~~~~~~~~~~~~~~~ egid TO XML (END) ~~~~~~~~~~~~~~~~~~~~~"
    sys.exit(1)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # LOOP OVER ETA REGIONS
  for reg in eta_regions:
  # for reg in ['low','high']:

    # Check if model exists
    if not os.path.exists("%s/BDT_%s/egid_%s_%s_%seta_%s.model"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)):
      print " --> [ERROR] No model exists for this BDT: %s/BDT_%s/egid_%s_%s_%seta_%s.model. Train first! Leaving..."%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)
      print "~~~~~~~~~~~~~~~~~~~~~ egid TRAINING (END) ~~~~~~~~~~~~~~~~~~~~~"
      sys.exit(1)
  
    print " --> Loading model for %s eta region: %s/BDT_%s/egid_%s_%s_%seta_%s.model"%(reg,out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)    
    egid = xg.Booster()
    egid.load_model( "%s/BDT_%s/egid_%s_%s_%seta_%s.model"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)   )
 
    #Define name of xml file to save
    f_xml = "%s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)

    # Convert to xml: using mlglue.tree functions
    from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
    target_names = ['background','signal']
    # FIXME: add options for saving BDT with user specified hyperparams
    bdt = BDTxgboost( egid, egid_vars[bdt_name], target_names, kind='binary', max_depth=6, learning_rate=0.3 )
    bdt.to_tmva( f_xml )

    print " --> Converted to xml: %s/BDT_%s/egid_%s_%s_%seta_%s.xml"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # END OF LOOP OVER ETA REGIONS
  print "~~~~~~~~~~~~~~~~~~~~~ egid TO XML (END) ~~~~~~~~~~~~~~~~~~~~~"
# END OF TO_XML FILE

# Main function for running program
if __name__ == "__main__": egid_to_xml()
