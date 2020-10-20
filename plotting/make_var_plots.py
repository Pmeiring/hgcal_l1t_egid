import pandas as pd
import numpy as np
import uproot as upr
#import matplotlib
import matplotlib.pyplot as plt
plt.style.use("cms11_nominal")
import argparse

vars_to_plot   = ['cl3d_showerlength','cl3d_coreshowerlength','cl3d_firstlayer',
                  'cl3d_maxlayer','cl3d_seetot','cl3d_spptot','cl3d_szz',
                  'cl3d_srrtot','cl3d_srrmean', 
                  'cl3d_meanz','cl3d_layer90','cl3d_layer50','cl3d_layer10','cl3d_ntc_67','cl3d_ntc_90']

vars_bin_map   = {'cl3d_showerlength':(0,30,30),'cl3d_coreshowerlength':(0,35,100),
                  'cl3d_firstlayer':(0,15,15), 'cl3d_maxlayer':(0,50,80),
                  'cl3d_seetot':(0,0.1,80),'cl3d_spptot':(0,0.1,60),
                  'cl3d_szz':(0,60,80), 'cl3d_srrtot':(0,0.012, 80),
                  'cl3d_srrmean':(0,0.006,80),  'cl3d_meanz':(320,420,80),
                  'cl3d_layer90':(0,30,80),'cl3d_layer50':(0,25,80),
                  'cl3d_layer10':(0,25,80),'cl3d_ntc_67':(0,45,120),
                  'cl3d_ntc_90':(0,110, 120)}

eta_regions = {"low":'cl3d_eta>1.5 and cl3d_eta<2.7', "high":'cl3d_eta>2.7 and cl3d_eta<3.0'}

labels_to_plot = [var.replace('cl3d_','') for var in vars_to_plot]
labels_to_plot = [var.replace('_',' ') for var in labels_to_plot]

def main(options):
    #get root files, convert into dataframes
    sig_file = upr.open(options.sigPath)
    sig_tree = sig_file[options.sigTree]
    sig_df   = sig_tree.pandas.df(vars_to_plot+['cl3d_eta'])

    bkg_file = upr.open(options.bkgPath)
    bkg_tree = bkg_file[options.bkgTree]
    bkg_df   = bkg_tree.pandas.df(vars_to_plot+['cl3d_eta'])
    
    for var in vars_to_plot:
        bins = np.linspace(vars_bin_map[var][0], vars_bin_map[var][1], vars_bin_map[var][2])
        for eta_reg in ['low', 'high']:
            fig = plt.figure()
            axes= plt.gca()
            temp_sig = sig_df.query(eta_regions[eta_reg]) 
            temp_bkg = bkg_df.query(eta_regions[eta_reg])
            var_sig = temp_sig[var]
            var_bkg = temp_bkg[var]

            sig_binned, _ = np.histogram(var_sig, bins=bins)
            sig_sum = np.sum(sig_binned)
            sig_norm_w = np.ones_like(var_sig)/sig_sum

            bkg_binned, _ = np.histogram(var_bkg, bins=bins)
            bkg_sum = np.sum(bkg_binned)
            bkg_norm_w = np.ones_like(var_bkg)/bkg_sum

            axes.tick_params(which='both', top=True, right=True)
            axes.hist(var_sig, bins=bins, weights=sig_norm_w, label='Signal 200PU', histtype='step', color='#91bfdb', zorder=11)
            axes.hist(var_bkg, bins=bins, weights=bkg_norm_w, label='Background 200PU', histtype='step', color='#fc8d59', zorder=11)
            axes.legend(bbox_to_anchor=(0.95,0.95), prop={'size':10})

            plt.text(0, 1.005, r'\textbf{CMS Phase-2} \textit{Simulation Preliminary}', ha='left', va='bottom', transform=axes.transAxes)
            plt.text(1, 1.005, r'$14$ TeV, $200$ PU', ha='right', va='bottom', transform=axes.transAxes)
            plt.text(0.05, 0.95, r'\textbf{EG endcap}, %s $\eta$'%(eta_reg), ha='left', va='top', transform=axes.transAxes)
            c_bottom, c_top =axes.get_ylim()
            axes.set_ylim(top=c_top*1.2)

            label_to_plot = var.replace('cl3d_','')
            label_to_plot = label_to_plot.replace('_',' ') 
            axes.set_xlabel(label_to_plot , size=11, ha='right', x=1)
            axes.set_ylabel('Arbitrary units', size=11, ha='right', y=1)

            fig.savefig('plots/sig_vs_bkg_var_{}_{}_eta.pdf'.format(var,eta_reg))

            plt.close()

if __name__ == '__main__':
    parser        = argparse.ArgumentParser()
    required_args =  parser.add_argument_group('Required Arguments')
    required_args.add_argument('-S', '--sigPath', action='store', type=str, required=True)
    required_args.add_argument('-s', '--sigTree', action='store', type=str, required=True)
    required_args.add_argument('-B', '--bkgPath', action='store', type=str, required=True)
    required_args.add_argument('-b', '--bkgTree', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)
