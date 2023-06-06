from egid_training_CJP import train_egid
from egid_to_xml import egid_to_xml
from egid_evaluate_CJP import evaluate_egid
from egid_summary_CJP import summary_egid
from egid_evaluate_singleCandidate import evaluate_egid_single
from optparse import OptionParser
import os
from shutil import copyfile
from ROOT import *


# Counter for the canvasses, gives annoying warnings otherwise
counter =0
def makePlots(sigTTree, bkgTTree,outputdir,subdir=""):
  global counter

  # Create output eos directory
  outputdir_=outputdir+subdir
  if not os.path.exists(outputdir_):
      os.makedirs(outputdir_)
      copyfile("/eos/user/p/pmeiring/www/L1Trigger/00_index.php",outputdir_+"index.php")    

  # Loop over the variables in the trees and plot them
  for key in sigTTree.GetListOfBranches():
    c = TCanvas("c%s"%counter)
    c.cd()
    gStyle.SetOptStat(0)
    leg = TLegend(0.65,0.65,0.88,0.88)
    var=key.GetName()

    # if "tkpt" in var:
    #   sigTTree.Draw("%s>>dummy%s(100,0,100)"%(var,counter),"","histnorm")
    # else:
    #   sigTTree.Draw("%s>>dummy%s"%(var,counter),"","histnorm")
    # if "dpt" in var:
    #   sigTTree.Draw("%s>>dummy%s(100,0,2)"%(var,counter),"","histnorm")
    # else:
    #   sigTTree.Draw("%s>>dummy%s"%(var,counter),"","histnorm")


    sigTTree.Draw("%s>>dummy%s"%(var,counter),"","histnorm")
    dummy=gDirectory.Get("dummy%s"%counter)
    dummy.SetLineColor(2)
    dummy.SetLineWidth(2)
    leg.AddEntry(dummy,"Signal")

    # sigTTree.Draw(var,"","histnorm")
    # gPad.GetListOfPrimitives().At(0).SetLineColor(2)
    # gPad.GetListOfPrimitives().At(0).SetLineWidth(2)
    # leg.AddEntry(gPad.GetListOfPrimitives().At(0),"Signal")
    # htemp.SetLineColor(2)
    # if var in bkgTTree.GetListOfBranches():
    # if "tkpt" in var:
    #   bkgTTree.Draw("%s>>dummybkg%s(100,0,100)"%(var,counter),"","samehistnorm")      
    # else:
    #   bkgTTree.Draw("%s>>dummybkg%s"%(var,counter),"","samehistnorm")
    # if "dpt" in var:
    #   bkgTTree.Draw("%s>>dummybkg%s(100,0,2)"%(var,counter),"","samehistnorm")      
    # else:
    #   bkgTTree.Draw("%s>>dummybkg%s"%(var,counter),"","samehistnorm")

    bkgTTree.Draw("%s>>dummybkg%s"%(var,counter),"","samehistnorm")
    dummy=gDirectory.Get("dummybkg%s"%counter)
    dummy.SetLineWidth(2)
    leg.AddEntry(dummy,"Background")


    # bkgTTree.Draw(var,"","samehistnorm")
    # gPad.GetListOfPrimitives().At(1).SetLineWidth(2)
    # leg.AddEntry(gPad.GetListOfPrimitives().At(1),"Background")    
    leg.Draw("same")
    # if "hoe" in var:
    c.SetLogy()
    c.SaveAs(outputdir_+var+".png")
    c.SaveAs(outputdir_+var+".pdf")
    counter+=1



# Configure options
def get_options():
  parser = OptionParser()
  parser.add_option('--clusteringAlgo', dest='clusteringAlgo', default='Histomaxvardr', help="Clustering algorithm with which to optimise BDT" )
  # parser.add_option('--signalType', dest='signalType', default='electron_200PU', help="Input signal type" )
  # parser.add_option('--backgroundType', dest='backgroundType', default='neutrino_200PU', help="Input background type" )
  parser.add_option('--bdtConfig', dest='bdtConfig', default='full', help="BDT config (accepted values: baseline/full/extended)" )
  parser.add_option('--reweighting', dest='reweighting', default=1, type='int', help="Boolean to perform re-weighting of clusters to equalise signal and background [yes=1 (default), no=0]" )
  parser.add_option('--trainParams',dest='trainParams', default=None, help='Comma-separated list of colon-separated pairs corresponding to (hyper)parameters for the training')
  parser.add_option('--ptBin', dest='ptBin', default='default', help="Used pT bin (accepted values: default, low)" )
  parser.add_option('--bdts', dest='bdts', default='full', help="Comma separated list of BDTs to evaluate. Format is <discrimnator>:<config>,... e.g. electron_200PU_vs_neutrino_200PU:baseline,electron_200PU_vs_neutrino_200PU:full" )
  parser.add_option('--sampleType', dest='sampleType', default='electron', help="Input sample type" )  
  parser.add_option('--dataset', dest='dataset', default='all', help="Ntuple to evaluate on [test,train,all]" )
  parser.add_option('--inputMap', dest='inputMap', default='electron,minbias,Histomaxvardr,test', help='Comma separated list of input info. Format is <signalType>,<backgroundType>,<clustering Algo.>,<dataset [test,train,all]>')
  parser.add_option('--outputROC', dest='outputROC', default=1, type='int', help="Display output ROC curves for egids [1=yes,0=no]" )  
  parser.add_option('--step', dest='step', default='01234', help='Parts of the full BDT procedure to run')
  parser.add_option('--etaBin', dest='etaBin', default='low')
  return parser.parse_args()


# Input variables to BDT for different configs. Specify config in options. To try new BDT with different inputs variables, then add another key to dict
egid_vars = {"basic":['coreshowerlength','firstlayer','maxlayer','srrmean'],
             'baseline':['coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot'],
             'allvars':    ['coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'seemax', 'sppmax', 'srrmax', 'meanz', 'emaxe', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90', 'hoe'],
             'allvars_trk':['coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'seemax', 'sppmax', 'srrmax', 'meanz', 'emaxe', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90', 'hoe','tkchi2','tkz0','tknstubs','tkpt','dphi','deta'],
             'allvars_trk2':['coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'seemax', 'sppmax', 'srrmax', 'meanz', 'emaxe', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90', 'hoe','tkchi2','tkz0','tknstubs','tkpt','dphi','deta','dpt'],
             'best9_loweta': ['hoe', 'srrtot', 'firstlayer', 'ntc67', 'ntc90', 'layer50','seetot', 'layer10', 'emaxe'],
             'best9_higheta':['hoe', 'ntc67', 'srrtot', 'spptot', 'ntc90', 'emaxe', 'layer90', 'szz', 'layer50'],
             'best9_loweta_lowpt': ['layer90', 'hoe', 'srrtot', 'ntc67', 'ntc90', 'coreshowerlength','seetot', 'layer50', 'spptot'],
             'best9_higheta_lowpt':['seetot', 'layer90', 'meanz', 'hoe', 'ntc90', 'ntc67','spptot', 'layer10', 'emaxe'], 
             'allvars_red':['coreshowerlength','showerlength','firstlayer','maxlayer','szz','srrmean','srrtot','seetot','spptot', 'meanz', 'layer10', 'layer50', 'layer90', 'ntc67', 'ntc90'],
             'allvars_trk2_best9': ['tkpt','srrtot','dpt','hoe','ntc67','deta','tkchi2','dphi','layer50'],
             # 'best9_loweta_red': ['srrtot', 'ntc67', 'ntc90', 'hoe', 'seetot', 'coreshowerlength','srrmean', 'srrmax', 'emaxe'],
             # 'best9_higheta_red':['ntc67', 'ntc90', 'srrtot', 'hoe', 'spptot', 'sppmax', 'seetot', 'emaxe', 'layer10'],
             'allAvailVars':               ['hoe','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','spptot','tkz0','seetot','showerlength','coreshowerlength','firstlayer','szz','tknstubs'],
             'allAvailVars_best3cl_alltrk':['hoe','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'],
             'allAvailVars_best3cl_alltrk2':['hoe','tkpt','srrtot','deta'],
             'allAvailVars_best3cl_alltrk_lowpt':['coreshowerlength','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'],
             'allAvailVars_best3cl_alltrk_lowpt2':['coreshowerlength','hoe','tkpt','srrtot','deta','dpt','meanz','dphi','tkchi2','tkz0','tknstubs'],
             'emulator':['tkChi2','tkpt','tkz0','hoe','srrtot','deta','dphi','dpt','meanz','nstubs','chi2rphi','chi2rz','ch2bend'],
             'emulator2':['tkpt','hoe','srrtot','deta','dphi','dpt','meanz','nstubs','chi2rphi','chi2rz','ch2bend'],
             'emulator_allpt':['tkpt','hoe','srrtot','deta','dphi','dpt','meanz','nstubs','chi2rphi','chi2rz','ch2bend'],
             'emulator_12p5':['tkpt','hoe','srrtot','deta','dphi','dpt','meanz','nstubs','chi2rphi','chi2rz','ch2bend'],
            }

eta_range={
  'low':'1p5eta2p7',
  'high':'2p7eta3p0',
}

def main():
  (opt,args) = get_options()
  eos = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/BDTs/"
  php = "/eos/user/p/pmeiring/www/L1Trigger/00_index.php"
  v = "1"
  # subdir = "MyBDT_%s_20211210/"%v
  # subdir = "MyBDT_%s_20230315_depth4_nonorm_shap/"%v
  # subdir = "MyBDT_emulated_20230331_v12p5samples/"
  subdir = "MyBDT_emulated_20230331_638pm/"
  outputdir = eos+subdir

  # Create output eos directory
  for bdt in opt.bdts.split(","):
    bdt="BDT_%s/"%bdt
    if not os.path.exists(outputdir+bdt):
      os.makedirs(outputdir+bdt)
    copyfile(php,outputdir+bdt+"index.php")    
  if not os.path.exists(outputdir+"_Summary"):
    os.makedirs(outputdir+"_Summary")
  copyfile(php,outputdir+"_Summary/index.php")

  eta_regions = {"low":[1.5,2.7]} if opt.etaBin=='low' else {"high":[2.7,3.0]}

  # Give paths to input files (separate for low pT bin, and separate for eta range)
  # file_sig = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_tkE_eg_v230105_1.3high_1p5eta2p7_BDT.root"
  # file_bkg = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_tkE_eg_v230105_2.3high_1p5eta2p7_BDT.root"

  file_bkg = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_tkE_eg_v230331_floattohw.3high_1p5eta2p7_BDT.root"
  file_sig = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_tkE_eg_v230331_floattohw.3high_1p5eta2p7_BDT.root"

  # file_sig = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v%s%s_%s_BDT.root"%(v,opt.ptBin,eta_range[opt.etaBin])
  # file_bkg = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v%s%s_%s_BDT.root"%(v,opt.ptBin,eta_range[opt.etaBin])
  # file_sig = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_eg_v%s%s_%s_BDT_pt10.root"%(v,opt.ptBin,eta_range[opt.etaBin])
  # file_bkg = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_eg_v%s%s_%s_BDT_pt10.root"%(v,opt.ptBin,eta_range[opt.etaBin])


  # Open the files and mount the trees
  f_sig=TFile.Open(file_sig)
  f_bkg=TFile.Open(file_bkg)
  signal = f_sig.Get("sig_train")
  backgr = f_bkg.Get("bkg_train")  

  if '0' in opt.step: makePlots(signal,backgr,outputdir,subdir="_features_%spT_%seta/"%(opt.ptBin,opt.etaBin))
  f_sig.Close()
  f_bkg.Close()

  # Loop over BDT input variable sets
  for varset in opt.bdts.split(","):
    opt.bdtConfig=varset

    # Train the BDT
    if '1' in opt.step: train_egid(opt, egid_vars, eta_regions, f_sig=file_sig, f_bkg=file_bkg, out=outputdir)

    # Convert the .model to .xml
    if '2' in opt.step: egid_to_xml(opt, egid_vars, eta_regions, out=outputdir)

  # Evaluate the clusters with the trained BDTs
  if '3' in opt.step: evaluate_egid(opt, egid_vars, eta_regions, f_sig=file_sig, f_bkg=file_bkg, out=outputdir)
  if 'single' in opt.step: evaluate_egid_single(opt, egid_vars, eta_regions, f_sig=file_sig, f_bkg=file_bkg, out=outputdir)

  # opt.bdts=opt.bdts+",tpg"
  # Make a summary of BDT performance
  if '4' in opt.step: summary_egid(opt, egid_vars, eta_regions, out=outputdir)

  print ("\nView results at: https://pmeiring.web.cern.ch/pmeiring/L1Trigger/l1eg/BDTs/"+subdir)

if __name__ == "__main__": main()


