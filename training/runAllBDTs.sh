# Standalone
# python egid_fullProcedureBDT.py --bdts baseline,allvars --ptBin low --etaBin low --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin low --etaBin high --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin high --etaBin low --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin high --etaBin high --step 34

# Tracks
# python egid_fullProcedureBDT.py --bdts baseline,allvars,allvars_trk,allvars_trk2 --ptBin high --etaBin low --step 01234
# python egid_fullProcedureBDT.py --bdts baseline,allvars,allvars_trk,allvars_trk2 --ptBin low --etaBin low --step 01234

# python egid_fullProcedureBDT.py --bdts allvars_trk2,allvars_trk2_best9 --ptBin high --etaBin low --step 1

# python egid_fullProcedureBDT.py --bdts allAvailVars --trainParams "max_depth:6,gamma:0.0" --ptBin high --etaBin low --step 1
python egid_fullProcedureBDT.py --bdts allAvailVars_best3cl_alltrk --trainParams "max_depth:6,gamma:0.0" --ptBin high --etaBin low --step 1
