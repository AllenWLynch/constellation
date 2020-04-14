#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn
import plotly.express as px 
#%%
def import_data(filename):

    lit_data = pd.read_csv(filename)
    lit_data['docvec'] = lit_data['docvec'].str.strip('[]').str.strip().str.split(r'\s+')
    lit_data['docvec'] = lit_data['docvec'].apply(lambda x : np.array(x).astype(np.float32))

    lit_data['disease_tags'] = lit_data['disease_tags'].str.strip('[]').str.strip().str.lower().str.split(r',\s+')\
        .apply(lambda x : [s.strip('\'') for s in x])

    lit_data['authors_list'] = lit_data['authors'].str.split(';').apply(lambda x : [s.strip() for s in x if len(x) > 0])

    lit_data = lit_data.set_index('id')

    doc_matrix = np.stack([*lit_data['docvec'].values]).astype(np.float32)
    sim_matrix = cosine_similarity(doc_matrix)

    sim_df = pd.DataFrame(sim_matrix, index = lit_data.index, columns = lit_data.index)

    #1. normalize pagerank
    norm_pagerank = (lit_data.pagerank - lit_data.pagerank.min())/(lit_data.pagerank.max() - lit_data.pagerank.min())
    lit_data['norm_pagerank'] = 1 - np.exp(-(1/np.mean(norm_pagerank)) * norm_pagerank)

    #2. make colorscale
    cateogories = list(lit_data['category_description'].value_counts().sort_values(ascending = False).index)
    color_scale = px.colors.qualitative.Light24[10:20] #seaborn.color_palette("bright")[:10]
    #scaled_scale = (np.array(color_scale) * 255).astype(np.int32).astype('str')
    #rgb_codes = ['rgb({},{},{})'.format(r, g, b) for (r,g,b) in scaled_scale]
    #color_pairs = list(zip(cateogories, rgb_codes))
    color_pairs = list(zip(cateogories, color_scale))
    color_pairs_df = pd.DataFrame(color_pairs, columns = ['category_description', 'color_code'])

    lit_data = lit_data.merge(color_pairs_df, how = 'left')
    lit_data = lit_data.set_index(sim_df.index)

    return lit_data, sim_df


# %%
