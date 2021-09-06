# Standalone
# python egid_fullProcedureBDT.py --bdts baseline,allvars --ptBin low --etaBin low --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin low --etaBin high --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin high --etaBin low --step 34
# python egid_fullProcedureBDT.py --bdts allvars --ptBin high --etaBin high --step 34

# Tracks
python egid_fullProcedureBDT.py --bdts baseline,allvars,allvars_trk,allvars_trk2 --ptBin high --etaBin low --step 01234
python egid_fullProcedureBDT.py --bdts baseline,allvars,allvars_trk,allvars_trk2 --ptBin low --etaBin low --step 01234

