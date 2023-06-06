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



# load both bdts
# load the sample for inclusive pt
#   (should have )

# evaluate low pt BDT  --> Plot ROC
# evaluate high pt BDT --> Plot ROC
# evaluate BDT combinations --> Plot ROC for different pT cuts

BDT_high = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220918_depth4/BDT_allAvailVars_best3cl_alltrk/egid_allAvailVars_best3cl_alltrk_Histomaxvardr_loweta_high.xml"
BDT_low  = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/BDT_allAvailVars_best3cl_alltrk_lowpt/egid_allAvailVars_best3cl_alltrk_lowpt_Histomaxvardr_loweta_low.xml"

bdtvars={
'allAvailVars_best3cl_alltrk':      ['hoe',             'tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'],
'allAvailVars_best3cl_alltrk_lowpt':['coreshowerlength','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'],
}

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
    var_beforenorm = getattr( in_cl3d, "%s"%var ) 
    min_ = norm[var][0]
    max_ = norm[var][1]
    var_afternorm  = (var_beforenorm-min_)/(max_-min_)
    var_scaled = var_afternorm * (max_-min_) + min_
    # print var, var_beforenorm, var_afternorm, var_scaled
    _bdt_var[var][0]=var_afternorm #getattr( in_cl3d, "%s"%var ) 
    # _bdt_var[var][0]=getattr( in_cl3d, "%s"%var ) 
  #return BDT score
  # print (_bdt[0].EvaluateMVA("BDT"))
  return _bdt.EvaluateMVA("BDT")

bdt_low = initialise_egid_BDT( BDT_low,  bdtvars["allAvailVars_best3cl_alltrk_lowpt"] )
bdt_high= initialise_egid_BDT( BDT_high, bdtvars["allAvailVars_best3cl_alltrk"] )

bdt_list={
    "allAvailVars_best3cl_alltrk":       bdt_high,
    "allAvailVars_best3cl_alltrk_lowpt": bdt_low,
}
# (<cppyy.gbl.TMVA.Reader object at 0xa333f60>, {'hoe': array('f', [0.0]), 'tkpt': array('f', [0.0]), 'srrtot': array('f', [0.0]), 'deta': array('f', [0.0]), 'dpt': array('f', [0.0]), 'meanz': array('f', [0.0]), 'dphi': array('f', [0.0]), 'tkchi2': array('f', [0.0]), 'tkz0': array('f', [0.0]), 'tknstubs': array('f', [0.0])})


# procFileMap={
#     "signal":"/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT_pt5.root",
#     "background": "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT_pt5.root",
# }
# procFileMap={
#     "signal":"/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT_pt15to25.root",
#     "background": "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT_pt15to25.root",
# }
# procFileMap={
#     "signal":"/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT.root",
#     "background": "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v123low_1p5eta2p7_BDT.root",
# }
procFileMap={
    "signal":"/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v123high_1p5eta2p7_BDT_pt25.root",
    "background": "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v123high_1p5eta2p7_BDT_pt25.root",
}
# procFileMap={
#     "signal":"/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v123high_1p5eta2p7_BDT.root",
#     "background": "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v123high_1p5eta2p7_BDT.root",
# }
treeMap = {"signal":"sig_all","background":"bkg_all"}

# BDT cuts in low pt model for given signal efficiency
wp_low={
    0.550 :  0.9980687,
    0.600 :  0.9976677,
    0.650 :  0.9964133,
    0.975 : -0.9260299,
    0.950 : -0.6348263,
    0.900 :  0.6027616,
}

wp_high={
    0.975 :  0.7927004,
    0.950 :  0.9826955,
    0.900 :  0.9948407,
}


# Number of events: [total signal , signal passing cut, bkg passing cut]
nEv_lowptmodel={
    0.550 : [0,0,0,0],
    0.600 : [0,0,0,0],
    0.650 : [0,0,0,0],
    0.975 : [0,0,0,0],
    0.950 : [0,0,0,0],
    0.900 : [0,0,0,0],
}
nEv_highptmodel={
    0.975 : [0,0,0,0],
    0.950 : [0,0,0,0],
    0.900 : [0,0,0,0],
}


truelabels = []
truelabels_ = {
    "pt5to10":[],
    "pt10to15":[],
    "pt15to20":[],
    "pt20to25":[],
    "pt25to30":[],
    "pt30":[],
}  

scores = {
    "bdtlow":[],
    "bdthigh":[],
    "bdthigh_pt5to10":[],
    "bdthigh_pt10to15":[],
    "bdthigh_pt15to20":[],
    "bdthigh_pt20to25":[],
    "bdthigh_pt25to30":[],
    "bdthigh_pt30":[],
}  

nEv_highptmodel_wp97p5={
    "pt5to10":  [0,0,0,0],
    "pt10to15": [0,0,0,0],
    "pt15to20": [0,0,0,0],
    "pt20to25": [0,0,0,0],
    "pt25to30": [0,0,0,0],
    "pt30":     [0,0,0,0],
}

for proc in ["signal","background"]:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONFIGURE INPUT NTUPLE
    # Define input ntuple
    f_in_name = procFileMap[ proc ]
    if not os.path.exists( f_in_name ):
        print ( " --> [ERROR] Input ntuple %s does not exist. Leaving..."%f_in_name)
        print ( "~~~~~~~~~~~~~~~~~~~~~ egid EVALUATE (END) ~~~~~~~~~~~~~~~~~~~~~")
        sys.exit(1)

    # Extract trees
    f_in = ROOT.TFile.Open( f_in_name )
    t_in = ROOT.TTree()


    treelist = ROOT.TList()
    t_tmp1 = f_in.Get( treeMap[proc].replace("all","test") )
    t_tmp2 = f_in.Get( treeMap[proc].replace("all","train") )
    treelist.Add( t_tmp1 )
    treelist.Add( t_tmp2 )
    f_tmp = ROOT.TFile.Open("tmpfile.root","RECREATE")
    t_in = ROOT.TTree.MergeTrees(treelist)

    # for s in scores:
    #     t_in.Branch(s, scores[s], "cl3d_%s/F"%s) 

    for ic,cl3d in enumerate(t_in):
        # print (cl3d.pt)
        # if proc=="background" and ic==100000:
        #     break

        if not ((abs(cl3d.eta) > 1.52)&(abs(cl3d.eta) <= 2.4)):
            continue

        truelabel = 1 if proc=="signal" else 0
        truelabels.append(truelabel)

        b = "allAvailVars_best3cl_alltrk_lowpt"
        scores["bdtlow"].append(evaluate_egid_BDT(b, bdt_list[b][0], bdt_list[b][1], cl3d, bdtvars[b] ))

        b = "allAvailVars_best3cl_alltrk"
        scores["bdthigh"].append(evaluate_egid_BDT(b, bdt_list[b][0], bdt_list[b][1], cl3d, bdtvars[b] ))


        if cl3d.pt>5. and cl3d.pt<10.:
            truelabels_["pt5to10"].append(truelabel)
            scores["bdthigh_pt5to10"].append(scores["bdtlow"][-1])
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt5to10"][0]+=1
                nEv_highptmodel_wp97p5["pt5to10"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt5to10"][2]+=1
                nEv_highptmodel_wp97p5["pt5to10"][3]+= scores["bdtlow"][-1]>wp_low[0.975]      

        if cl3d.pt>10. and cl3d.pt<15.:
            scores["bdthigh_pt10to15"].append(scores["bdtlow"][-1])
            truelabels_["pt10to15"].append(truelabel)
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt10to15"][0]+=1
                nEv_highptmodel_wp97p5["pt10to15"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt10to15"][2]+=1
                nEv_highptmodel_wp97p5["pt10to15"][3]+= scores["bdtlow"][-1]>wp_low[0.975]  

        if cl3d.pt>15. and cl3d.pt<20.:
            scores["bdthigh_pt15to20"].append(scores["bdtlow"][-1])
            truelabels_["pt15to20"].append(truelabel)
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt15to20"][0]+=1
                nEv_highptmodel_wp97p5["pt15to20"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt15to20"][2]+=1
                nEv_highptmodel_wp97p5["pt15to20"][3]+= scores["bdtlow"][-1]>wp_low[0.975]    

        if cl3d.pt>20. and cl3d.pt<25.:
            scores["bdthigh_pt20to25"].append(scores["bdtlow"][-1])
            truelabels_["pt20to25"].append(truelabel)
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt20to25"][0]+=1
                nEv_highptmodel_wp97p5["pt20to25"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt20to25"][2]+=1
                nEv_highptmodel_wp97p5["pt20to25"][3]+= scores["bdtlow"][-1]>wp_low[0.975]      

        if cl3d.pt>25. and cl3d.pt<30.:
            scores["bdthigh_pt25to30"].append(scores["bdtlow"][-1])
            truelabels_["pt25to30"].append(truelabel)
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt25to30"][0]+=1
                nEv_highptmodel_wp97p5["pt25to30"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt25to30"][2]+=1
                nEv_highptmodel_wp97p5["pt25to30"][3]+= scores["bdtlow"][-1]>wp_low[0.975]     

        if cl3d.pt>30.:
            scores["bdthigh_pt30"].append(scores["bdtlow"][-1])
            truelabels_["pt30"].append(truelabel)
            if proc=="signal":
                nEv_highptmodel_wp97p5["pt30"][0]+=1
                nEv_highptmodel_wp97p5["pt30"][1]+= scores["bdtlow"][-1]>wp_low[0.975]
            if proc=="background":
                nEv_highptmodel_wp97p5["pt30"][2]+=1
                nEv_highptmodel_wp97p5["pt30"][3]+= scores["bdtlow"][-1]>wp_low[0.975]      


        # if cl3d.pt>10:
        #     scores["pt10"].append(scores["bdthigh"][-1])
        # else:
        #     scores["pt10"].append(scores["bdtlow"][-1])

        # if cl3d.pt>15:
        #     scores["pt15"].append(scores["bdthigh"][-1])
        # else:
        #     scores["pt15"].append(scores["bdtlow"][-1])

        # if cl3d.pt>20:
        #     scores["pt20"].append(scores["bdthigh"][-1])
        # else:
        #     scores["pt20"].append(scores["bdtlow"][-1])


        for wp in wp_low:
            if proc=="signal":
                nEv_lowptmodel[wp][0]+=1
                nEv_lowptmodel[wp][1]+= scores["bdtlow"][-1]>wp_low[wp]
            if proc=="background":
                nEv_lowptmodel[wp][2]+=1
                nEv_lowptmodel[wp][3]+= scores["bdtlow"][-1]>wp_low[wp]                

        for wp in wp_high:
            if proc=="signal":
                nEv_highptmodel[wp][0]+=1
                nEv_highptmodel[wp][1]+= scores["bdthigh"][-1]>wp_high[wp]
            if proc=="background":
                nEv_highptmodel[wp][2]+=1
                nEv_highptmodel[wp][3]+= scores["bdthigh"][-1]>wp_high[wp]                



    # CLOSE FILES
    f_in.Close()
    f_tmp.Close()
    os.system("rm tmpfile.root")

#
efficiencies_lowptmodel={
    0.550 : [nEv_lowptmodel[0.550][1]/nEv_lowptmodel[0.550][0], 1-nEv_lowptmodel[0.550][3]/nEv_lowptmodel[0.550][2]],
    0.600 : [nEv_lowptmodel[0.600][1]/nEv_lowptmodel[0.600][0], 1-nEv_lowptmodel[0.600][3]/nEv_lowptmodel[0.600][2]],
    0.650 : [nEv_lowptmodel[0.650][1]/nEv_lowptmodel[0.650][0], 1-nEv_lowptmodel[0.650][3]/nEv_lowptmodel[0.650][2]],
    0.900 : [nEv_lowptmodel[0.900][1]/nEv_lowptmodel[0.900][0], 1-nEv_lowptmodel[0.900][3]/nEv_lowptmodel[0.900][2]],
    0.950 : [nEv_lowptmodel[0.950][1]/nEv_lowptmodel[0.950][0], 1-nEv_lowptmodel[0.950][3]/nEv_lowptmodel[0.950][2]],
    0.975 : [nEv_lowptmodel[0.975][1]/nEv_lowptmodel[0.975][0], 1-nEv_lowptmodel[0.975][3]/nEv_lowptmodel[0.975][2]],
}

efficiencies_highptmodel={
    0.900 : [nEv_highptmodel[0.900][1]/nEv_highptmodel[0.900][0], 1-nEv_highptmodel[0.900][3]/nEv_highptmodel[0.900][2]],
    0.950 : [nEv_highptmodel[0.950][1]/nEv_highptmodel[0.950][0], 1-nEv_highptmodel[0.950][3]/nEv_highptmodel[0.950][2]],
    0.975 : [nEv_highptmodel[0.975][1]/nEv_highptmodel[0.975][0], 1-nEv_highptmodel[0.975][3]/nEv_highptmodel[0.975][2]],
}
# print(nEv)
print(efficiencies_lowptmodel)
print(efficiencies_highptmodel)

fpr, tpr, _ = roc_curve(truelabels, scores["bdtlow"])
AUC_xgb = roc_auc_score(truelabels, scores["bdtlow"])
plt.plot(tpr, 1-fpr, color="darkorange", lw=2, label="ROC XGBoost model (low pt). AUC = %0.5f" %AUC_xgb)

# ROC for Conifer model (hls backend)
fpr, tpr, _ = roc_curve(truelabels, scores["bdthigh"])
AUC_hls = roc_auc_score(truelabels, scores["bdthigh"])
plt.plot(tpr, 1-fpr, color="blue", lw=2, label="ROC XGBoost model (high pt). AUC = %0.5f" %AUC_hls)

plt.plot(efficiencies_lowptmodel[0.550][0], efficiencies_lowptmodel[0.550][1], 'gx', label='WP 55.0% for low pT model')
plt.plot(efficiencies_lowptmodel[0.600][0], efficiencies_lowptmodel[0.600][1], 'rx', label='WP 60.0% for low pT model')
plt.plot(efficiencies_lowptmodel[0.650][0], efficiencies_lowptmodel[0.650][1], 'bx', label='WP 65.0% for low pT model')
plt.plot(efficiencies_lowptmodel[0.900][0], efficiencies_lowptmodel[0.900][1], 'go', label='WP 90.0% for low pT model')
plt.plot(efficiencies_lowptmodel[0.950][0], efficiencies_lowptmodel[0.950][1], 'ro', label='WP 95.0% for low pT model')
plt.plot(efficiencies_lowptmodel[0.975][0], efficiencies_lowptmodel[0.975][1], 'bo', label='WP 97.5% for low pT model')

plt.plot(efficiencies_highptmodel[0.900][0], efficiencies_highptmodel[0.900][1], 'co', label='WP 90.0% for high pT model')
plt.plot(efficiencies_highptmodel[0.950][0], efficiencies_highptmodel[0.950][1], 'mo', label='WP 95.0% for high pT model')
plt.plot(efficiencies_highptmodel[0.975][0], efficiencies_highptmodel[0.975][1], 'yo', label='WP 97.5% for high pT model')

plt.xlabel('Signal Eff. ($\epsilon_s$)')
plt.ylabel('1 - Background Eff. ($1-\epsilon_b$)')
# plt.title('%.2f$ < |\eta| < $%.2f'%(1.52,2.4))
# plt.title("Candidates with 15<pt<25")
# plt.title("Candidates with 5<pt<25")
plt.title("Candidates with pT>25 GeV")
plt.grid(True)
axes = plt.gca()
axes.set_xlim([0.0,1.1])
axes.set_ylim([0.0,1.1])
plt.legend(bbox_to_anchor=(0.05,0.1), loc='lower left')
plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROC_pt25.png")
plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROC_pt25.pdf")

axes.set_xlim([0.5,1.05])
axes.set_ylim([0.5,1.05])
plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROCzoom_pt25.png")
plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROCzoom_pt25.pdf")




# efficiencies_highptmodel={
#     "pt5to10":  [nEv_highptmodel_wp97p5["pt5to10"][1]/nEv_highptmodel_wp97p5["pt5to10"][0], 1-nEv_highptmodel_wp97p5["pt5to10"][3]/nEv_highptmodel_wp97p5["pt5to10"][2]],
#     "pt10to15": [nEv_highptmodel_wp97p5["pt10to15"][1]/nEv_highptmodel_wp97p5["pt10to15"][0], 1-nEv_highptmodel_wp97p5["pt10to15"][3]/nEv_highptmodel_wp97p5["pt10to15"][2]],
#     "pt15to20": [nEv_highptmodel_wp97p5["pt15to20"][1]/nEv_highptmodel_wp97p5["pt15to20"][0], 1-nEv_highptmodel_wp97p5["pt15to20"][3]/nEv_highptmodel_wp97p5["pt15to20"][2]],
#     "pt20to25": [nEv_highptmodel_wp97p5["pt20to25"][1]/nEv_highptmodel_wp97p5["pt20to25"][0], 1-nEv_highptmodel_wp97p5["pt20to25"][3]/nEv_highptmodel_wp97p5["pt20to25"][2]],
#     "pt25to30": [nEv_highptmodel_wp97p5["pt25to30"][1]/nEv_highptmodel_wp97p5["pt25to30"][0], 1-nEv_highptmodel_wp97p5["pt25to30"][3]/nEv_highptmodel_wp97p5["pt25to30"][2]],
#     "pt30":     [nEv_highptmodel_wp97p5["pt30"][1]/nEv_highptmodel_wp97p5["pt30"][0], 1-nEv_highptmodel_wp97p5["pt30"][3]/nEv_highptmodel_wp97p5["pt30"][2]],
# }
# print (efficiencies_highptmodel)
# # plt.plot(efficiencies_highptmodel["pt5to10"][0], efficiencies_highptmodel["pt5to10"][1], 'bo',markersize=12)
# # plt.plot(efficiencies_highptmodel["pt10to15"][0], efficiencies_highptmodel["pt10to15"][1], 'go',markersize=12)
# # plt.plot(efficiencies_highptmodel["pt15to20"][0], efficiencies_highptmodel["pt15to20"][1], 'ro',markersize=12)
# # plt.plot(efficiencies_highptmodel["pt20to25"][0], efficiencies_highptmodel["pt20to25"][1], 'yo',markersize=12)
# # plt.plot(efficiencies_highptmodel["pt25to30"][0], efficiencies_highptmodel["pt25to30"][1], 'co',markersize=12)
# # plt.plot(efficiencies_highptmodel["pt30"][0], efficiencies_highptmodel["pt30"][1], 'mo',markersize=12)


# # ROC for Conifer model (hls backend)
# fpr, tpr, _ = roc_curve(truelabels_["pt5to10"], scores["bdthigh_pt5to10"])
# AUC_hls = roc_auc_score(truelabels_["pt5to10"], scores["bdthigh_pt5to10"])
# plt.plot(tpr, 1-fpr, color="blue", lw=2, label="Low pT model, 5<pt<10. AUC = %0.5f" %AUC_hls)
# fpr, tpr, _ = roc_curve(truelabels_["pt10to15"], scores["bdthigh_pt10to15"])
# AUC_hls = roc_auc_score(truelabels_["pt10to15"], scores["bdthigh_pt10to15"])
# plt.plot(tpr, 1-fpr, color="green", lw=2, label="Low pT model, 10<pt<15. AUC = %0.5f" %AUC_hls)
# fpr, tpr, _ = roc_curve(truelabels_["pt15to20"], scores["bdthigh_pt15to20"])
# AUC_hls = roc_auc_score(truelabels_["pt15to20"], scores["bdthigh_pt15to20"])
# plt.plot(tpr, 1-fpr, color="red", lw=2, label="Low pT model, 15<pt<20. AUC = %0.5f" %AUC_hls)
# fpr, tpr, _ = roc_curve(truelabels_["pt20to25"], scores["bdthigh_pt20to25"])
# AUC_hls = roc_auc_score(truelabels_["pt20to25"], scores["bdthigh_pt20to25"])
# plt.plot(tpr, 1-fpr, color="yellow", lw=2, label="Low pT model, 20<pt<25. AUC = %0.5f" %AUC_hls)
# fpr, tpr, _ = roc_curve(truelabels_["pt25to30"], scores["bdthigh_pt25to30"])
# AUC_hls = roc_auc_score(truelabels_["pt25to30"], scores["bdthigh_pt25to30"])
# plt.plot(tpr, 1-fpr, color="cyan", lw=2, label="Low pT model, 25<pt<30. AUC = %0.5f" %AUC_hls)
# fpr, tpr, _ = roc_curve(truelabels_["pt30"], scores["bdthigh_pt30"])
# AUC_hls = roc_auc_score(truelabels_["pt30"], scores["bdthigh_pt30"])
# plt.plot(tpr, 1-fpr, color="magenta", lw=2, label="High pT model, pt>30. AUC = %0.5f" %AUC_hls)
# plt.grid(True)
# plt.xlabel('Signal Eff. ($\epsilon_s$)')
# plt.ylabel('1 - Background Eff. ($1-\epsilon_b$)')
# axes = plt.gca()
# axes.set_xlim([0.0,1.1])
# axes.set_ylim([0.0,1.1])
# plt.legend(bbox_to_anchor=(0.05,0.1), loc='lower left')
# plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROClowpTmodel_pt5_binned.png")
# plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROClowpTmodel_pt5_binned.pdf")
# axes.set_xlim([0.0,1.05])
# axes.set_ylim([0.75,1.05])
# plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROClowpTmodel_pt5_binned_zoom.png")
# plt.savefig( "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/MyBDT_123_20220920_depth4_lowpt/_Summary/summaryROClowpTmodel_pt5_binned_zoom.pdf")




















