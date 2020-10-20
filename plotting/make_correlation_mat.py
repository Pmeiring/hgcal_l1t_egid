import pandas as pd
import numpy as np
import uproot as upr
#import matplotlib
import matplotlib.pyplot as plt
plt.style.use("cms11_LessTicks")
import argparse

vars_to_plot   = ['cl3d_showerlength','cl3d_coreshowerlength','cl3d_firstlayer',
                  'cl3d_maxlayer','cl3d_seetot','cl3d_seemax','cl3d_spptot','cl3d_sppmax','cl3d_szz',
                  'cl3d_srrtot','cl3d_srrmax','cl3d_srrmean', 
                  'cl3d_meanz','cl3d_layer90','cl3d_layer50','cl3d_layer10','cl3d_ntc_67','cl3d_ntc_90']

labels_to_plot = [var.replace('cl3d_','') for var in vars_to_plot]
labels_to_plot = [var.replace('_',' ') for var in labels_to_plot]

def corr_matrix(arr1, arr2):
    '''
    Calculate pearson correlation coefficient for two vectors (features)
    '''

    m1 = np.average(arr1)*np.ones_like(arr1)
    m2 = np.average(arr2)*np.ones_like(arr2)
    cov_11 = float(((arr1-m1)**2).sum()) #denom1
    cov_22 = float(((arr2-m2)**2).sum()) #denom2
    cov_12 = float(((arr1-m1)*(arr2-m2)).sum()) #numerator
    return cov_12/np.sqrt(cov_11*cov_22)

def plot_numbers(ax,mat):
    '''
    Plot correlation coefficient as text labels
    '''
    for i in xrange(mat.shape[0]):
        for j in xrange(mat.shape[1]):
            c = mat[j,i]
            if np.abs(c)>=1:
                ax.text(i,j,'{:.0f}'.format(c),fontdict={'size': 8},va='center',ha='center')

def main(options):
    #get root files, convert into dataframes
    sig_file = upr.open(options.sigPath)
    sig_tree = sig_file[options.sigTree]
    sig_df   = sig_tree.pandas.df(vars_to_plot)

    bkg_file = upr.open(options.bkgPath)
    bkg_tree = bkg_file[options.bkgTree]
    bkg_df   = bkg_tree.pandas.df(vars_to_plot)
    
    #get correlations
    sig_corrs = np.array([ [100*corr_matrix(sig_df[var1].values, sig_df[var2].values) for var2 in vars_to_plot] for var1 in vars_to_plot])
    bkg_corrs = np.array([ [100*corr_matrix(bkg_df[var1].values, bkg_df[var2].values) for var2 in vars_to_plot] for var1 in vars_to_plot])
    
    #plot sig correlations
    plt.set_cmap('bwr')
    fig = plt.figure()
    axes= plt.gca()
    mat = axes.matshow(sig_corrs, vmin=-100, vmax=100)
    plot_numbers(axes, sig_corrs)
    axes.set_yticks(np.arange(len(vars_to_plot)))
    axes.set_xticks(np.arange(len(vars_to_plot)))
    axes.set_xticklabels(labels_to_plot,rotation='vertical')
    axes.set_yticklabels(labels_to_plot)
    axes.xaxis.tick_top()
    cbar = fig.colorbar(mat)
    cbar.set_label(r'Correlation (\%)')
    fig.savefig('plots/sig_crl_matrix.pdf')
    plt.close()
    
    #plot bkg correlations
    plt.set_cmap('bwr')
    fig = plt.figure()
    axes= plt.gca()
    mat = axes.matshow(bkg_corrs, vmin=-100, vmax=100)
    plot_numbers(axes, bkg_corrs)
    axes.set_yticks(np.arange(len(vars_to_plot)))
    axes.set_xticks(np.arange(len(vars_to_plot)))
    axes.set_xticklabels(labels_to_plot,rotation='vertical')
    axes.set_yticklabels(labels_to_plot)
    axes.xaxis.tick_top()
    cbar = fig.colorbar(mat)
    cbar.set_label(r'Correlation (\%)')
    fig.savefig('plots/bkg_crl_matrix.pdf')

if __name__ == '__main__':
    parser        = argparse.ArgumentParser()
    required_args =  parser.add_argument_group('Required Arguments')
    required_args.add_argument('-S', '--sigPath', action='store', type=str, required=True)
    required_args.add_argument('-s', '--sigTree', action='store', type=str, required=True)
    required_args.add_argument('-B', '--bkgPath', action='store', type=str, required=True)
    required_args.add_argument('-b', '--bkgTree', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)
