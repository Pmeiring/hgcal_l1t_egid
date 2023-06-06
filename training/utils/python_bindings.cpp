#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"

// We need primitive types on the interface, so cast the l1ct::Scales conversions with float return
float makePt(int pt){ return (float) l1ct::Scales::makePt(pt); }
float makeDPt(int dpt){ return (float) l1ct::Scales::makeDPt(dpt); }
float makePtFromFloat(float pt){ return (float) l1ct::Scales::makePtFromFloat(pt); }
float makeDPtFromFloat(float dpt){ return (float) l1ct::Scales::makeDPtFromFloat(dpt); }
float makeZ0(float z0){ return (float) l1ct::Scales::makeZ0(z0); }

float makePhi(float phi){ return (float) l1ct::Scales::makePhi(phi); }
float makeEta(float eta){ return (float) l1ct::Scales::makeEta(eta); }
float makeGlbEta(float eta){ return (float) l1ct::Scales::makeGlbEta(eta); }
float makeGlbPhi(float phi){ return (float) l1ct::Scales::makeGlbPhi(phi); }
float makeIso(float iso){ return (float) l1ct::Scales::makeIso(iso); }
float makeChi2(float chi2){ return (float) l1ct::Scales::makeChi2(chi2); }
float makeSrrTot(float var){ return (float) l1ct::Scales::makeSrrTot(var); }
float makeMeanZ(float var){ return (float) l1ct::Scales::makeMeanZ(var); }
float makeHoe(float var){ return (float) l1ct::Scales::makeHoe(var); }

namespace py = pybind11;
PYBIND11_MODULE(correlator_common, m){
  m.doc() = "Python bindings for correlator common scalings";
  m.def("makePt", &makePt, "convert pt from integer to physical float");
  m.def("makeDPt", &makeDPt, "convert dpt from integer to physical float");
  m.def("makePtFromFloat", &makePtFromFloat, "convert pt from physical float to hardware units");
  m.def("makeDPtFromFloat", &makeDPtFromFloat, "convert dpt from physical float to hardware units");
  m.def("makeZ0", &makeZ0, "convert z0 from physical float to hardware units");
  m.def("makePhi", &makePhi, "convert phi from physical float to hardware units");
  m.def("makeEta", &makeEta, "convert eta from physical float to hardware units");
  m.def("makeGlbPhi", &makeGlbPhi, "convert global phi from physical float to hardware units");
  m.def("makeGlbEta", &makeGlbEta, "convert global eta from physical float to hardware units");
  m.def("makeIso", &makeIso, "convert isolation from physical float to hardware units");
  m.def("makeChi2", &makeChi2, "convert track chi2 from physical float to hardware units");
  m.def("makeDR2FromFloatDR", &l1ct::Scales::makeDR2FromFloatDR, "convert dr from physical units to dr2 in hardware units");
  m.def("makeSrrTot", &makeSrrTot, "convert srrTot from physical float to hardware units");
  m.def("makeMeanZ", &makeMeanZ, "convert meanz from physical float to hardware units");
  m.def("makeHoe", &makeHoe, "convert HoE from physical float to hardware units");
}