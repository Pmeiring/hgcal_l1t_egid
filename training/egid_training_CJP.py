# Train egid (BDT: xgboost) for hgcal l1t: using shower shape variables
# > Takes as input clusters which pass selection 
# > Trains separately in different eta regions (1.5-2.7 and 2.7-3.0)

#usual imports
import ROOT
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
import os
import sys
from array import array
from optparse import OptionParser
import seaborn as sns
import joblib
from itertools import compress
import utils.correlator_common as cc
# from egid_fullProcedureBDT import egid_vars
import shap

print (cc.makePtFromFloat(20.333))


#Function to train xgboost model for HGCal L1T egid
def train_egid(opt, egid_vars, eta_regions, f_sig, f_bkg, out):

  # (opt,args) = get_options()
  print ("~~~~~~~~~~~~~~~~~~~~~~~~ egid TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~")

  #Set numpy random seed 123456
  np.random.seed(1651231)

  # Training and validation fractions
  trainFrac = 0.9
  validFrac = 0.1

  bdt_name = opt.bdtConfig

  procFileMap = {"signal":f_sig,"background":f_bkg}
  treeMap = {"signal":"sig_train","background":"bkg_train"}

  # Check if models and frames directories exist
  if not os.path.isdir("./models"):
    print (" --> Making ./models directory to store trained egid models")
    os.system("mkdir models")
  if not os.path.isdir("./frames"):
    print (" --> Making ./frames directory to store pandas dataFrames")
    os.system("mkdir frames")

  print ("did I come here")
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # EXTRACT DATAFRAMES FROM INPUT SELECTED CLUSTERS
  trainTotal = None
  trainFrames = {}
  
  #extract the trees: turn them into arrays
  for proc,fileName in procFileMap.items():
    print (proc, fileName)
    trainFile = ROOT.TFile("%s"%fileName)
    # print treeMap[proc]
    trainTree = trainFile.Get( treeMap[proc] )

    #initialise new tree with only relevant variables
    _file = ROOT.TFile("tmp%s%s.root"%(opt.ptBin,opt.etaBin),"RECREATE")
    _tree = ROOT.TTree("tmp","tmp")
    _vars = {}
    for var in egid_vars[bdt_name]:
      _vars[ var ] = array( 'f', [-1.] )
      _tree.Branch( '%s'%var, _vars[ var ], '%s/F'%var )
    #Also add cluster eta to do eta splitting
    _vars['eta'] = array( 'f', [-999.] )
    _tree.Branch( 'eta', _vars['eta'], 'eta/F' )  

    #loop over events in tree and add to tmp tree
    for ev in trainTree:
      for var in egid_vars[bdt_name]: _vars[ var ][0] = getattr( ev, '%s'%var )
      _vars['eta'][0] = getattr( ev, 'eta' )
      _tree.Fill()
  
    #Convert tmp tree to pandas dataFrame and delete tmp files
    print ("Okay let us do the conversion")
    dataTree, columnsTree = _tree.AsMatrix(return_labels=True)
    trainFrames[proc] = pd.DataFrame( data=dataTree, columns=columnsTree )
    del _file
    del _tree
    os.system('rm tmp%s%s.root'%(opt.ptBin,opt.etaBin))

    #Add columns to dataframe to label clusters
    # trainFrames[proc]['proc'] = procMap[ proc ]
    trainFrames[proc]['proc'] = proc
    print (trainFrames[proc])
    print (" --> Extracted %s dataFrame from file: %s"%(proc,fileName))

  #Create one total frame: i.e. concatenate signal and bkg
  # trainList = []
  # for proc in procs: trainList.append( trainFrames[proc] )
  trainList = [trainFrames["signal"].append(trainFrames["background"])]
  trainTotal = pd.concat( trainList, sort=False )
  del trainFrames
  print (" --> Created total dataFrame")
  print ("trainTotal: \n{}".format(trainTotal))
  # Save dataFrames as pkl file
  # joblib.dump(trainTotal, "./frames/%s.pkl"%bdt_name)
  # pd.to_picle( trainTotal, "./frames/%s.pkl"%bdt_name )

    #### QUANTIZE THE DATA
    # ['hoe','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs']
  # trainTotal['hoe'] =   trainTotal['hoe'].map(lambda element: cc.makeHoe(element))
  # trainTotal['tkpt'] =  trainTotal['tkpt'].map(lambda element: cc.makePtFromFloat(element))
  # trainTotal['srrtot']= trainTotal['srrtot'].map(lambda element: cc.makeSrrTot(element))
  # trainTotal['deta'] =  trainTotal['deta'].map(lambda element: cc.makeEta(element))
  # trainTotal['dpt'] =   trainTotal['dpt'].map(lambda element: cc.makeDPtFromFloat(element))
  # trainTotal['meanz'] = trainTotal['meanz'].map(lambda element: cc.makeMeanZ(element))
  # trainTotal['dphi'] =  trainTotal['dphi'].map(lambda element: cc.makePhi(element))
  # trainTotal['tkchi2'] =trainTotal['tkchi2'].map(lambda element: cc.makeChi2(element))
  # trainTotal['tkz0'] =  trainTotal['tkz0'].map(lambda element: cc.makeZ0(element))
  # print ("trainTotal: \n{}".format(trainTotal))


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # TRAIN MODEL: loop over different eta regions
  print ("")
  for reg in eta_regions:

    print (" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print (" --> Training for %s eta region: %2.1f < |eta| < %2.1f"%(reg,eta_regions[reg][0],eta_regions[reg][1]))

    #Impose eta cuts
    train_reg = trainTotal[ abs(trainTotal['eta'])>eta_regions[reg][0] ]
    train_reg = train_reg[ abs(train_reg['eta'])<=eta_regions[reg][1] ]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # REWEIGHTING: 
    if opt.reweighting:
      print (" --> Reweighting: equalise signal and background samples (same sum of weights)")
      sum_sig = len( train_reg[ train_reg['proc'] == "signal" ].index )
      sum_bkg = len( train_reg[ train_reg['proc'] == "background" ].index )
      weights = list( map( lambda a: (sum_sig+sum_bkg)/sum_sig if a == "signal" else (sum_sig+sum_bkg)/sum_bkg, train_reg['proc'] ) )
      train_reg['weight'] = weights 
    else:
      print (" --> No reweighting: assuming same S/B as in input ntuples")
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE DATASETS: shuffle to get train and validation
    print (" --> Configuring training and validation datasets")
    label_encoder = LabelEncoder()
    theShape = train_reg.shape[0]  
    theShuffle = np.random.permutation( theShape )
    egid_trainLimit = int(theShape*trainFrac)
    egid_validLimit = int(theShape*validFrac)
  
    #Set up dataFrames for training BDT
    egid_X = train_reg[ egid_vars[bdt_name] ].values
    egid_y = label_encoder.fit_transform( train_reg['proc'].values )
    if opt.reweighting: egid_w = train_reg['weight'].values

    #Peform shuffle
    egid_X = egid_X[theShuffle]
    egid_y = egid_y[theShuffle]
    if opt.reweighting: egid_w = egid_w[theShuffle]

    #Define training and validation sets
    egid_train_X, egid_valid_X, dummy_X = np.split(egid_X, [egid_trainLimit, egid_validLimit+egid_trainLimit] )
    egid_train_y, egid_valid_y, dummy_y = np.split(egid_y, [egid_trainLimit, egid_validLimit+egid_trainLimit] )

    if opt.reweighting: egid_train_w, egid_valid_w, dummy_w = np.split(egid_w, [egid_trainLimit, egid_validLimit+egid_trainLimit] )

    #### SCALE THE DATA TO 0-1
    scaler = MinMaxScaler()
    print(scaler.fit(egid_train_X))
    egid_train_X_scaled=scaler.transform(egid_train_X)

    # print(scaler.get_params())
    print(egid_train_X)
    # print(egid_train_X_scaled)
    for f in range(len(egid_vars[bdt_name])):
        feature_name=egid_vars[bdt_name][f]
        min_=0
        max_=1
        feature_vals= egid_train_X[:,f]
        feature_std = (feature_vals-feature_vals.min()) / (feature_vals.max()-feature_vals.min())
        feature_scld= feature_std * (max_-min_) + min_
        # Checked that feature_scld=egid_train_X_scaled
        print(feature_name,feature_vals.min(),feature_vals.max())

    # egid_train_X=egid_train_X_scaled
    # print(scaler.fit(egid_valid_X))
    # egid_valid_X_scaled=scaler.transform(egid_valid_X)
    # egid_valid_X=egid_valid_X_scaled

    # Save plots
    for f in range(len(egid_vars[bdt_name])):
        feature=egid_vars[bdt_name][f]
        print(feature)
        feature_values=[obj[f] for obj in egid_train_X]
        # feature_values=[obj[f] for obj in egid_train_X_scaled]
        feature_values_sig=list(compress(feature_values, egid_train_y))
        egid_train_y_flip=[not elem for elem in egid_train_y]
        feature_values_bkg=list(compress(feature_values, egid_train_y_flip))

        plt.hist(feature_values_bkg,alpha=0.5,bins=50,range=(0,1),label="background")
        plt.hist(feature_values_sig,alpha=0.5,bins=50,range=(0,1),label="signal")
        plt.yscale('log')
        plt.legend(loc='upper right')
        # plt.savefig('%s/BDT_%s/%s.png'%(out,bdt_name,feature))
        plt.clf()

    np.save( '%s/BDT_%s/egid_train_X.npy'%(out,bdt_name), egid_train_X)
    np.save( '%s/BDT_%s/egid_valid_X.npy'%(out,bdt_name), egid_valid_X)
    np.save( '%s/BDT_%s/egid_train_y.npy'%(out,bdt_name), egid_train_y)
    np.save( '%s/BDT_%s/egid_valid_y.npy'%(out,bdt_name), egid_valid_y)
    np.save( '%s/BDT_%s/egid_train_w.npy'%(out,bdt_name), egid_train_w)
    np.save( '%s/BDT_%s/egid_valid_w.npy'%(out,bdt_name), egid_valid_w)
  

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BUILDING THE MODEL
    if opt.reweighting:
      training_egid = xg.DMatrix( egid_train_X, label=egid_train_y, weight=egid_train_w, feature_names=egid_vars[bdt_name] )
      validation_egid = xg.DMatrix( egid_valid_X, label=egid_valid_y, weight=egid_valid_w, feature_names=egid_vars[bdt_name] )
    else:
      training_egid = xg.DMatrix( egid_train_X, label=egid_train_y, feature_names=egid_vars[bdt_name] )
      validation_egid = xg.DMatrix( egid_valid_X, label=egid_valid_y, feature_names=egid_vars[bdt_name] )

    # extract training hyper-parameters for model from input option
    trainParams = {}
    trainParams['objective'] = 'binary:logistic'
    trainParams['nthread'] = 1
    trainParams['random_state'] = 2360
    # trainParams['max_depth'] = 3


    # paramExt = ''
    if opt.trainParams:
      paramExt = '__'
      paramPairs = opt.trainParams.split(",")
      for paramPair in paramPairs:
        # paramPair=opt.trainParams
        print (paramPair)
        param = paramPair.split(":")[0]
        value = paramPair.split(":")[1]
        trainParams[param] = value
        # paramExt += '%s)%s__'%(param_value)
        # paramExt = paramExt[:-2]

    # Train the model
    print (" --> Training the model: %s"%trainParams)
    egid = xg.train( trainParams, training_egid )
    print (" --> Done.")

    # Save the model
    egid.save_model( '%s/BDT_%s/egid_%s_%s_%seta_%s.model'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin) )
    myconfig = egid.save_config()
    with open('%s/BDT_%s/egid_%s_%s_%seta_%s_config.json'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin) , "w") as text_file:
      text_file.write(myconfig)

    print (myconfig)
    print (" --> Model saved: %s/BDT_%s/egid_%s_%s_%seta_%s.model"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin))
 
    # Feature importance: number of splittings
    xg.plot_importance( egid )
    plt.gcf().subplots_adjust( left = 0.3 )
    plt.grid(True)
    plt.xlabel( 'Number of splittings', fontsize = 22 )
    plt.ylabel( 'Feature', fontsize = 22 )
    plt.savefig( '%s/BDT_%s/feature_importance_egid_split_%s_%s_%s_%s.pdf'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.savefig( '%s/BDT_%s/feature_importance_egid_split_%s_%s_%s_%s.png'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.clf()

    # Feature importance: gains
    xg.plot_importance( egid , importance_type="gain")
    plt.gcf().subplots_adjust( left = 0.3 )
    plt.grid(True)
    plt.xlabel( 'Number of splittings', fontsize = 22 )
    plt.ylabel( 'Feature', fontsize = 22 )
    plt.savefig( '%s/BDT_%s/feature_importance_egid_gain_%s_%s_%s_%s.pdf'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.savefig( '%s/BDT_%s/feature_importance_egid_gain_%s_%s_%s_%s.png'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.clf()

    # Feature importance: Shapley values
    XD=training_egid
    XD.shape=[theShape,len(egid_vars[bdt_name])]
    explainer = shap.TreeExplainer(egid)
    shap_values = explainer(XD)
    shap_values.data=egid_train_X
    shap_values.feature_names=egid_vars[bdt_name]

    shap.plots.beeswarm(shap_values, max_display=27)
    plt.gcf().subplots_adjust( left = 0.3 )
    plt.grid(True)
    plt.savefig( '%s/BDT_%s/feature_importance_egid_shap_%s_%s_%s_%s.pdf'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.savefig( '%s/BDT_%s/feature_importance_egid_shap_%s_%s_%s_%s.png'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.clf()
    
    shap.plots.bar(shap_values.abs.mean(0), max_display=27)
    plt.gcf().subplots_adjust( left = 0.3 )
    plt.grid(True)
    plt.savefig( '%s/BDT_%s/feature_importance_egid_shapbar_%s_%s_%s_%s.pdf'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.savefig( '%s/BDT_%s/feature_importance_egid_shapbar_%s_%s_%s_%s.png'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.clf()

    # Correlation matrix
    correlations = train_reg[egid_vars[bdt_name]].corr()
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(correlations, vmax=1.0, center=0, cmap="coolwarm",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    for t in ax.texts: t.set_text(str(int(100.* float(t.get_text()))))
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['-100%', '-75%', '-50%', '-25%', '0%', '25%', '50%', '75%', '100%'])
    plt.savefig( '%s/BDT_%s/correlations_%s_%s_%s_%s.pdf'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))
    plt.savefig( '%s/BDT_%s/correlations_%s_%s_%s_%s.png'%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin ))


    # Save in raw format
    # if not os.path.isdir("out/raw"): os.system("mkdir models/raw")
    egid.dump_model("%s/BDT_%s/egid_%s_%s_%seta_%s.raw.txt"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin))
    print (" --> Model saved (RAW): %s/BDT_%s/egid_%s_%s_%seta_%s.raw.txt"%(out,bdt_name,bdt_name,opt.clusteringAlgo,reg,opt.ptBin))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHECKING PERFORMANCE OF MODEL: using trainig and validation sets
    egid_train_predy = egid.predict( training_egid )
    egid_valid_predy = egid.predict( validation_egid )

    print ("    *************************************************")
    print ("    --> Performance: in %s eta region (%2.1f < |eta| < %2.1f)"%(reg,eta_regions[reg][0],eta_regions[reg][1]))
    print ("      * Training set   ::: AUC = %5.4f"%roc_auc_score( egid_train_y, egid_train_predy ))
    print ("      * Validation set ::: AUC = %5.4f"%roc_auc_score( egid_valid_y, egid_valid_predy ))
    print ("    *************************************************")
    print ("")

  #END OF LOOP OVER ETA REGIONS
  print ("~~~~~~~~~~~~~~~~~~~~~ egid TRAINING (END) ~~~~~~~~~~~~~~~~~~~~~")
# END OF TRAINING FUNCTION

# Main function for running program
if __name__ == "__main__": train_egid()
