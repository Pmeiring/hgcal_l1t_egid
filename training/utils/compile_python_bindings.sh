#!/bin/bash

if [ -z ${XILINX_HLS} ]; then echo "XILINX_HLS is not set, did you source the HLS toolchain? Exiting"; exit 2; else echo "Including ap_fixed headers from \$XILINX_HLS set to '$XILINX_HLS'"; fi

PYBINDINCL=$(python -m pybind11 --includes)
if [ $? -eq 1 ]; then echo "Command 'python -m pybind11 --includes' returned an error. Is pybind11 installed?"; exit 2; else echo "Including pybind11 headers '$PYBINDINCL'"; fi

[[ "$CMSSW_VERSION" == "" ]] && echo "\$CMSSW_VERSION not defined, set \$CMSSW_VERSION. Exiting" && exit 2
[[ "$CMSSW_BASE" == "" ]] && CMSSW_BASE=../$CMSSW_VERSION && echo "Including CMSSW from $CMSSW_BASE/src"


g++ -O3 -shared -std=c++11 -fPIC $PYBINDINCL -I$XILINX_HLS/include -I$CMSSW_BASE/src -I. python_bindings.cpp -o correlator_common.so 
