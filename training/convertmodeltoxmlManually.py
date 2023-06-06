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



egid = xg.Booster()
egid.load_model( "egid_allAvailVars_best3cl_alltrk_Histomaxvardr_loweta_high.model")

#Define name of xml file to save
f_xml = "egid_allAvailVars_best3cl_alltrk_Histomaxvardr_loweta_high.xml"

# Convert to xml: using mlglue.tree functions
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
target_names = ['background','signal']
# FIXME: add options for saving BDT with user specified hyperparams
bdt = BDTxgboost( egid, ['hoe','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'], target_names, kind='binary', max_depth=3, learning_rate=0.3 )
bdt.to_tmva( f_xml )
