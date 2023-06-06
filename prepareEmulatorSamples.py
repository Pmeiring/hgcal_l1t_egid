import ROOT as r

#___________________________________________________________

gencut = {
  # "low"  : "Pt30", #<30
  # "high" : "Pt15", #>15
  "1p5eta2p7": "EtaBC",
  "2p7eta3p0": "EtaDE",  
}

tpcut = {
  # "low" :   "Pt5to25",
  # "high":   "Pt15",
  "1p5eta2p7": "EtaBC",
  "2p7eta3p0": "EtaDE",
}

treecut_clusterpt = {
  # "low": "pt>15 && pt<20",
  "low": "pt>5 && pt<25",
  "high":"pt>15",
  # "high":"pt>25",
}

def getTreeName(file, pt, eta):
  sigTree="NaN"
  bkgTree="NaN"

  # tree="h_cl3d_HMvDR_%s_"%(tpcut[eta])
  tree="h_cl3dtrk_TkEleEE_%s_"%(tpcut[eta])
  if "histos_ele" in file:
    sigTree = tree+"GEN"+gencut[eta]
  if "histos_minbias" in file:
    bkgTree = tree+"noMatch"
  return sigTree,bkgTree

bkgFile = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_minbias_PU200_HLTTDR_tkE_eg_v230331_floattohw.3.root"
sigFile = "/eos/user/p/pmeiring/www/L1Trigger/l1eg/histos_matching/histos_ele_flat2to100_PU200_HLTTDR_tkE_eg_v230331_floattohw.3.root"


if __name__ == '__main__':

  for file in [sigFile,bkgFile]:
    # for pTrange in ["low","high"]:
    for pTrange in ["high"]:
      # for etarange in ["1p5eta2p7","2p7eta3p0"]:
      for etarange in ["1p5eta2p7"]:

        inputFile = r.TFile.Open(file)
        outName = file.replace(".root","%s_%s_BDT.root"%(pTrange,etarange))
        outputFile = r.TFile.Open(outName, "recreate")
        outputFile.cd()

        dir = inputFile.Get("L1Trees")
        sigTree,bkgTree=getTreeName(file, pTrange, etarange)

        print ("Searching for Trees: \n",sigTree,"\n",bkgTree)

        for j in dir.GetListOfKeys():
          print ("dealing with: ", j.GetName(), file)
          if sigTree==j.GetName():
             nameTrain = "sig_train"
             nameTest = "sig_test"
          elif bkgTree==j.GetName():
             nameTrain = "bkg_train"
             nameTest = "bkg_test"
          else: continue
          origTree = inputFile.Get(dir.GetName()+"/"+j.GetName())
          print ("Total number of events to be processed: ", origTree.GetEntries())

          cut = treecut_clusterpt[pTrange] #+ " && tttrack_chi2>-500"

          dummyTree = origTree.CopyTree(cut)
          print ("I have produced the dummy tree, moving on to the actual trees")

          newTrain=dummyTree.CloneTree(0)
          newTrain.SetName(nameTrain)
          newTest=dummyTree.CloneTree(0)
          newTest.SetName(nameTest)

          nEv = dummyTree.GetEntries()
          if "minbias" in file and "low" in pTrange:
            nEv=int(nEv/5)

          fracTrain=0.95
          fracTest=0.05

          nTrain = int(fracTrain*nEv)
          nTest  = int(fracTest*nEv)

          print ("here we go: ", nEv, nTrain, nTest)

          for i in range(nEv):
             dummyTree.GetEntry(i)
             if i % 1000 == 1000: print ("Processed event ", i)
             if i<nTrain:
                 newTrain.Fill(dummyTree.GetEntry(i))
             elif i>=nTrain and i<=nTrain+nTest: 
                 newTest.Fill(dummyTree.GetEntry(i))
          newTrain.Write()
          newTest.Write()
                       
        outputFile.Close()
        inputFile.Close()
