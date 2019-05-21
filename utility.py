import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab, cm
import pickle, os
import re      
from IPython.display import display, display_html, HTML                   
                         
from scipy.stats import pearsonr, spearmanr

def remove_wave(x):
    return re.sub("(W\d+)+","",x)   
    
def trim_strings(x):
    if len( x.split("\n") )>1:
        return x.split("\n")[0] + "[...]"
    else:
        return x      

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