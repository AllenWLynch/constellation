
#%%
from azure.storage.file import FileService
import pandas as pd
from io import StringIO
import numpy as np
import json
import networkx as nx

ACCOUNT_NAME = 'constellationdata'
ACCOUNT_KEY = "GUUwP+7rb3Px4EONo8vcJrc3JTySEhzqVvupgMBnPebiTDTcMSzTolQHkG0rQtBqImIjZSKqJkKHv8qm/FPXaQ=="
SHARE_NAME = 'constellationdata'

def load_data(ACCOUNT_NAME, ACCOUNT_KEY, SHARE_NAME):

    print('Loading data...')

    file_service = FileService(account_name = ACCOUNT_NAME, account_key=ACCOUNT_KEY)

    def read_csv_from_azure(sharename, dirname, filename):
        return pd.read_csv(
            StringIO(file_service.get_file_to_text(sharename, dirname, filename).content)
        )
    #%%
    data = read_csv_from_azure(SHARE_NAME, None, 'dashboard_data.csv').set_index('index')
    data.index = data.index.astype(np.str)
    data['tags'] = data.tags.fillna('')

    def get_important_authors(authors_list):
        all_authors = authors_list.split(';')
        if len(all_authors) < 6:
            return authors_list
        else:
            return all_authors[0] + ', et al.'

    data['pretty_authors'] = data.authors.apply(lambda x : get_important_authors(x))

    groups = data.groupby('label')
    num_groups = len(groups)
    #%%__category_labels____
    category_labels = file_service.get_file_to_text(SHARE_NAME, None, 'cluster_categories.txt')
    category_labels = [label.strip() for label in category_labels.content.split('\n')]

    #%%__Similarities_______
    similarities = read_csv_from_azure(SHARE_NAME, None, 'similarities.csv').astype(np.str)

    #%%__Abstracts________
    abstracts = read_csv_from_azure(SHARE_NAME, None, 'abstracts.csv').set_index('index')
    abstracts.index = abstracts.index.astype(np.str)
    #%%__Load in taxonomy_____
    graph_data = read_csv_from_azure(SHARE_NAME,'taxonomy','graph_data.csv').drop(columns = ['Unnamed: 0'])

    edge_data = json.loads(file_service.get_file_to_text(SHARE_NAME, 'taxonomy', 'edge_data.json').content)

    taxonomy = nx.DiGraph()
    taxonomy.add_edges_from(edge_data)

    pos = {
        node : xy
        for node, xy in list(zip(graph_data.node.values, graph_data[['x','y']].values))
    }

    edge_x = []
    edge_y = []
    for (src, dest) in taxonomy.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dest]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

#%%#__Index_____________

     

    print('Data loaded!')
#%%
    return (data, groups, num_groups, category_labels, similarities, abstracts, taxonomy, edge_x, edge_y)
# %%
