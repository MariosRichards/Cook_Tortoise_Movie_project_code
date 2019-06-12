import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab, cm
import pickle, os
import re      
from IPython.display import display, display_html, HTML                   
                         
from scipy.stats import pearsonr, spearmanr


def match(df, pattern, case_sensitive=True, mask=None):
    if mask is None:
           mask = pd.Series(np.ones( (df.shape[0]) )).astype('bool')
    if case_sensitive:
        return df[[x for x in df.columns if re.match(pattern,x)]][mask].notnull().sum()
    else:
        return df[[x for x in df.columns if re.match(pattern, x, re.IGNORECASE)]][mask].notnull().sum()

def search(df, pattern, case_sensitive=False, mask=None):
    if mask is None:
           mask = pd.Series(np.ones( (df.shape[0]) )).astype('bool')
    if case_sensitive:
        return df[[x for x in df.columns if re.search(pattern,x)]][mask].notnull().sum()
    else:
        return df[[x for x in df.columns if re.search(pattern, x, re.IGNORECASE)]][mask].notnull().sum()

def remove_wave(x):
    return re.sub("(W\d+)+","",x)   
    
def trim_strings(x):
    if len( x.split("\n") )>1:
        return x.split("\n")[0] + "[...]"
    else:
        return x      
        
def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in set(lst1) if value in temp] 
    return lst3         

def corr_simple_pearsonr(df1,df2, mask=1, round_places=2):
    mask = df1.notnull()&df2.notnull()&mask
    (r,p) = pearsonr(df1[mask],df2[mask])
    return [round(r,round_places), round(p,round_places), mask.sum()]

def corr_simple_spearmanr(df1,df2, mask=1, round_places=2):
    mask = df1.notnull()&df2.notnull()&mask
    (r,p) = spearmanr(df1[mask],df2[mask])
    return [round(r,round_places), round(p,round_places), mask.sum()]            
                         
def display_corr(df, name, corr_type, top_num = 20, round_places = 2,
                 correlation_text = "r", p_value_text = "p", sample_size_text = "N",
                 text_wrap_length=50):
#     df.index = [x[0:60] for x in df.index]
    df.index =  [trim_strings(x) for x in df.index.str.wrap(width = text_wrap_length)]
    
    df[correlation_text] = df[correlation_text].round(round_places)
    
    df1 = df.sort_values(by=correlation_text, ascending=False)[0:top_num][[correlation_text,p_value_text,sample_size_text]]
    df2 = df.sort_values(by=correlation_text)[0:top_num][[correlation_text,p_value_text,sample_size_text]]
    
    df1[p_value_text]     = df1[p_value_text].apply(lambda x: "{0:0.2f}".format(x))
    df2[p_value_text]     = df2[p_value_text].apply(lambda x: "{0:0.2f}".format(x))

    df1_caption = "Top "+str(top_num)+ " positive "+"("+corr_type+")"+" correlations for "+name
    df2_caption = "Top "+str(top_num)+ " negative "+"("+corr_type+")"+" correlations for "+name

    df1_styler = df1.style.set_table_attributes("style='display:inline'").set_caption(df1_caption)
    df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption(df2_caption)

    display_html(df1_styler._repr_html_().replace("\\n","<br />")+df2_styler._repr_html_().replace("\\n","<br />"), raw=True)


def make_corr_summary(input_df, name,  corr_type = "spearman", pattern=None, sample_size_text = "N", correlation_text = "r",
                      abs_correlation_text = "abs_r", p_value_text = "p",
                      min_p_value = 0.01, min_variance = 0.0, min_sample_size = 500):

    if pattern is None:
        pattern=name
    df1 = input_df.copy()
    focal_var = df1[name]
    focal_mask = focal_var.notnull()


    pattern_list = [x for x in df1.columns if re.search(pattern,x)]

    variances = df1[focal_mask].var()
    low_var_list = list(variances[variances<min_variance].index)
    sample_sizes = df1[focal_mask].notnull().sum()
    low_sample_size_list = list(sample_sizes[sample_sizes<min_sample_size].index)

    drop_list = pattern_list+low_var_list+low_sample_size_list
    df1.drop(drop_list,axis=1,inplace=True)

    if corr_type == "pearson":
        df = df1.apply(lambda x: corr_simple_pearsonr(x,focal_var)).apply(pd.Series)
    elif corr_type == "spearman":
        df = df1.apply(lambda x: corr_simple_spearmanr(x,focal_var)).apply(pd.Series)

    if len(df.columns)!=3:
        df=df.T
    df.columns = [correlation_text,p_value_text,sample_size_text]
 
    df[sample_size_text] = df[sample_size_text].astype('int')
    df[abs_correlation_text] = df[correlation_text].abs()

    zero_var_other_way_around_list = list(df[df[correlation_text].isnull()].index)
    df.dropna(inplace=True)

    insignificant_list = df[df[p_value_text]>min_p_value].index
    df.drop(insignificant_list,inplace=True)

    df.sort_values(by=abs_correlation_text,ascending=False,inplace=True)


    stub_dict = {}
    drop_list = []
    # drop repeated references to same variable in different waves???
    # so, what about different categories??? eg. blahWX_subcat
    # how about, just replace wave match as "X"
    # create a dictionary keyed on the top corr variable with all the drops inside
    for ind in df.index:
        waveless = remove_wave(ind)
        if waveless in stub_dict.keys():
            drop_list.append(ind)
            stub_dict[waveless].append(ind)
        else:
            stub_dict[waveless] = [ind]
    df.drop(drop_list,inplace=True)
    return df, corr_type    

import unicodedata
import string

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)


def clean_filename(filename, whitelist=valid_filename_chars, replace=' ', char_limit = 30):
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]   

    
def create_subdir(base_dir, subdir, char_limit=50):
    output_subfolder = base_dir + os.sep + clean_filename(subdir, char_limit=char_limit) + os.sep
    if not os.path.exists( output_subfolder ):
        os.makedirs( output_subfolder )
    return output_subfolder
    

def display_components(n_components, decomp, cols, BES_decomp, manifest, 
                       save_folder = False, show_first_x_comps=4,
                       show_histogram=True, flip_axes=True):
    
    if hasattr(decomp, 'coef_'):
        decomp_components = decomp.coef_
    elif hasattr(decomp, 'components_'):
        decomp_components = decomp.components_
    else:
        raise ValueError('no component attribute in decomp')    

    # hardcoded at 20?    
    n_comps = min(n_components,20)
    comp_labels = {}
    comp_dict = {}

    for comp_no in range(0,n_comps):

        fig, axes = plt.subplots(ncols=1+show_histogram)
        
        comp = pd.DataFrame( decomp_components[comp_no], index = cols, columns = ["components_"] )
        comp["comp_absmag"] = comp["components_"].abs()
        comp = comp.sort_values(by="comp_absmag",ascending=True)        
        
        if show_histogram:
            comp_ax = axes[0]
            
            hist_ax = axes[1]
            hist_ax.set_xlabel("abs. variable coeffs")
            hist_ax.set_title("Histogram of abs. variable coeffs")
            comp["comp_absmag"].hist( bins=30, ax=hist_ax, figsize=(10,6) )
            
        else:
            comp_ax = axes
            
        # set top abs_mag variable to label
        comp_labels[comp_no] = comp.index[-1:][0] # last label (= highest magnitude)
        # if top abs_mag variable is negative
     
        if flip_axes & (comp[-1:]["components_"].values[0] < 0):

            comp["components_"]         = -comp["components_"]
            decomp_components[comp_no]  = -decomp_components[comp_no]
            BES_decomp[comp_no]         = -BES_decomp[comp_no]

        if manifest is not None:
            dataset_description = manifest["Friendlier_Description"].values[0]
            title = "Comp. "+str(comp_no)+" (" + comp.index[-1:][0] + ")"
            comp_labels[comp_no] = title
            comp_ax.set_title( dataset_description + "\n" + title )
            comp_ax.set_xlabel("variable coeffs")
            xlim = (min(comp["components_"].min(),-1) , max(comp["components_"].max(),1) )
            comp["components_"].tail(30).plot( kind='barh', ax=comp_ax, figsize=(10,6), xlim=xlim )
            dataset_citation = "Source: " + manifest["Citation"].values[0]

            if (save_folder != False):
                comp_ax.annotate(dataset_citation, (0,0), (0, -40),
                                 xycoords='axes fraction', textcoords='offset points', va='top', fontsize = 7)            
                fname = save_folder + clean_filename(title) + ".png"
                fig.savefig( fname, bbox_inches='tight' )
        else:
            title = "Comp. "+str(comp_no)+" (" + comp.index[-1:][0] + ")"
            comp_labels[comp_no] = title
            comp_ax.set_title( title )
            comp_ax.set_xlabel("variable coeffs")    
            xlim = (min(comp["components_"].min(),-1) , max(comp["components_"].max(),1) )
            comp["components_"].tail(30).plot( kind='barh', ax=comp_ax, figsize=(10,6), xlim=xlim )
            
        comp_dict[comp_no] = comp
        # show first x components
        if (comp_no >= min(show_first_x_comps,n_components)):
            plt.close()

        
    return (BES_decomp, comp_labels, comp_dict)
    

def display_pca_data(n_components, decomp, BES_std, y=[]):    
    
    figsz = (16,3)
    
    f, axs = plt.subplots( 1, 4, figsize=figsz )

    axno = 0
    
    if hasattr(decomp, 'explained_variance_ratio_'):
        print('explained variance ratio (first 30): %s'
              % str(decomp.explained_variance_ratio_[0:30]) )
        
    if hasattr(decomp, 'explained_variance_'):
        print('explained variance (first 30): %s'
              % str(decomp.explained_variance_[0:30]) )

        axs[axno].plot( range(1,n_components+1), decomp.explained_variance_, linewidth=2)
        # ,figsize = figsz)
        axs[axno].set_xlabel('n_components')
        axs[axno].set_ylabel('explained_variance_')
        axs[axno].set_title('explained variance by n_components')
        axno = axno + 1
        
    if hasattr(decomp, 'noise_variance_'): 
        if isinstance(decomp.noise_variance_, float):
            print('noise variance: %s'
                  % str(decomp.noise_variance_) )
        
    if hasattr(decomp, 'score'):
        if len(y)==0:
            print('average log-likelihood of all samples: %s'
                  % str(decomp.score(BES_std)) )
        else:
            print('mean classification accuracy (harsh if many cats.): %s'
                  % str(decomp.score(BES_std, y)) )
        
    if hasattr(decomp, 'score_samples') and not np.isinf( decomp.score(BES_std) ):
        pd.DataFrame( decomp.score_samples(BES_std) ).hist(bins=100,figsize = figsz, ax=axs[axno])
        axs[axno].set_xlabel('log likelihood')
        axs[axno].set_ylabel('frequency')
        axs[axno].set_title('LL of samples')
        axno = axno + 1

    if hasattr(decomp, 'n_iter_'):
        print('number of iterations: %s'
              % str(decomp.n_iter_) )
        
    if hasattr(decomp, 'loglike_'):
        axs[axno].plot( decomp.loglike_, linewidth=2) # ,figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('log likelihood')
        axs[axno].set_title('LL by iter')
        axno = axno + 1

    if hasattr(decomp, 'error_'):

        axs[axno].plot( decomp.error_, linewidth=2, figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('error')
        axs[axno].set_title('LL by iter')
        axno = axno + 1
    
    

def display_pca_data(n_components, decomp, BES_std, y=[]):    
    
    figsz = (16,3)
    
    f, axs = plt.subplots( 1, 4, figsize=figsz )

    axno = 0
    
    if hasattr(decomp, 'explained_variance_ratio_'):
        print('explained variance ratio (first 30): %s'
              % str(decomp.explained_variance_ratio_[0:30]) )
        
    if hasattr(decomp, 'explained_variance_'):
        print('explained variance (first 30): %s'
              % str(decomp.explained_variance_[0:30]) )

        axs[axno].plot( range(1,n_components+1), decomp.explained_variance_, linewidth=2)
        # ,figsize = figsz)
        axs[axno].set_xlabel('n_components')
        axs[axno].set_ylabel('explained_variance_')
        axs[axno].set_title('explained variance by n_components')
        axno = axno + 1
        
    if hasattr(decomp, 'noise_variance_'): 
        if isinstance(decomp.noise_variance_, float):
            print('noise variance: %s'
                  % str(decomp.noise_variance_) )
        
    if hasattr(decomp, 'score'):
        if len(y)==0:
            print('average log-likelihood of all samples: %s'
                  % str(decomp.score(BES_std)) )
        else:
            print('mean classification accuracy (harsh if many cats.): %s'
                  % str(decomp.score(BES_std, y)) )
        
    if hasattr(decomp, 'score_samples') and not np.isinf( decomp.score(BES_std) ):
        pd.DataFrame( decomp.score_samples(BES_std) ).hist(bins=100,figsize = figsz, ax=axs[axno])
        axs[axno].set_xlabel('log likelihood')
        axs[axno].set_ylabel('frequency')
        axs[axno].set_title('LL of samples')
        axno = axno + 1

    if hasattr(decomp, 'n_iter_'):
        print('number of iterations: %s'
              % str(decomp.n_iter_) )
        
    if hasattr(decomp, 'loglike_'):
        axs[axno].plot( decomp.loglike_, linewidth=2) # ,figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('log likelihood')
        axs[axno].set_title('LL by iter')
        axno = axno + 1

    if hasattr(decomp, 'error_'):

        axs[axno].plot( decomp.error_, linewidth=2, figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('error')
        axs[axno].set_title('LL by iter')
        axno = axno + 1
        
        
# transform a column of data until it's as approximately normally distributed as can be
# because most Machine Learning/Statistical methods assume data is ~normally distributed
# basically, what people normally do randomly logging/square-rooting data, only automatically

from scipy import stats
def box_cox_normalise(ser, offset = 3, bw='scott'):
    
    
    # box cox lr_scale
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    x = ser.values +ser.values.min()+offset
    prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    ax2 = fig.add_subplot(312)
    xt, _ = stats.boxcox(x)
    prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    ax2.set_title('Probplot after Box-Cox transformation')
    ax3 = fig.add_subplot(313)
    xt_std = (xt-xt.mean())/xt.std()
    sns.kdeplot(xt_std, ax=ax3, bw=bw, cut=0);
    sns.kdeplot(np.random.normal(size=len(xt_std)), ax=ax3, cut=0);
    plt.suptitle(ser.name)
    return xt_std
    
    