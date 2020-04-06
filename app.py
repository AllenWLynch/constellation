
#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import json
import networkx as nx
import datetime
from dash.dependencies import Input, Output, State
from import_data import import_data

import whoosh
from whoosh_index import get_index
from whoosh.qparser import QueryParser

myindex = get_index('./search_index')
parser = QueryParser('content', schema = myindex.schema)

#%%
app = dash.Dash(__name__)

#%%
DATA_FILE = './dash_sample.csv'
#__Data_________________

print('Loading data ...')
_data, sim_df = import_data(DATA_FILE)

#__Load in filters_______
with open('./topic_graph.json','r') as f:
    edges = json.loads(f.read())

filters_graph = nx.DiGraph()
filters_graph.add_edges_from(edges)
filters = list(filters_graph.nodes)

_constellation = None

print('Data loaded!')

#%%
#___FIGURES_____________
BACKGROUND_COLOR = '#1F2132'
BOX_COLOR = '#262B3D'

#__Layout_______________
app.layout = html.Div(children=[

    #Title Bar
    html.Div([
        html.Img(
                src=app.get_asset_url('logo.png'),
                height = 100,
                style = {'margin-left':'25px', 'margin-top':'0px', 'padding-top':'0px'}
            ),
    ]),
    html.Div([        
        #Left panel
        html.Div([
            html.Div([
                html.H1('Constellation', className = 'infobox_header'),
                dcc.Graph(
                    id = 'constellation',
                    style = {'height':'55vh'},
                ),
                html.Br(),
                html.Div([
                    html.Div([
                        html.Label('Search'),
                        html.Br(),
                        html.Div([
                            dcc.Input(id="searchbar", type="text", placeholder="", debounce=True,
                            style = {'margin-right' : '5px', 'height' : '36px','display' : 'inline-block', 'width':'80%',
                                        'background-color' : BOX_COLOR, 'border-color' : '#fff','border-radius' : '2px', 'border-width' : '1px'}),
                            #html.Button('Submit'),
                        ], style = {'margin-top' : '0px', 'margin-bottom' : '15px'})
                    ], className = 'column'),
                    html.Div([
                        html.Label('Filters'),
                        html.Div([
                            dcc.Dropdown(
                                options=[
                                    {'label' : node_name, 'value' : node_name}
                                    for node_name in filters                                    
                                ],
                                value=[],
                                multi=True,
                                id = 'filtersbar',
                                style = {'background-color' : BOX_COLOR, 'border-color' : '#fff', 'border-width' : '1px',
                                    'font-family' : "Courier New", 'font-color' : '#fff', 'border-radius' : '2px',
                                    'width' : '95%', 
                                    'margin-left' : '0px', 'margin-right' : '0px',
                                }
                            ),
                        ], id="wrapper", style = {'clear' : 'both', 'margin-bottom' : '15px'}),
                    ],className = 'column'),
                ], className = 'row')                
            ], className = 'infobox'),
            html.Div([
                html.H1('Paper Information', className = 'infobox_header'),
                html.Div(children = [], id = 'abstract_box', style = {'margin-left' : '20px', 'margin-right' :'20px'}),
            ], className='infobox', style = {'display': 'block', 'overflow': 'auto','margin-top' : '10px'})
        ], className = 'column'),

        #right panel
        html.Div([
            html.Div([
                html.H1('Literature Suggestions', className = 'infobox_header'),
                html.Div(children = [], id = 'suggestions_list', style = {'margin-left' : '20px', 'margin-right' :'20px', 
                    'overflow-y' : 'scroll', 'height' : '95%'}),
            ],
            className = 'infobox', style = {'height' : '86vh'})
        ], className = 'column',), 
        
    ], className = 'row'),
])

#%%
#Literature suggestion fetching
def row_to_text(row):
    return [
        html.B(row['title'], style = {'font-size' : '1.8rem'}),
        html.Br(),
        html.P('{}, {} et al., {}'.format(row['pretty_date'], row['first_author'], row['source_x']), style = {'margin-bottom' : '5px', 'display' : 'inline'}),
        html.Br(),
        html.P('Category: {}'.format(row['category_description']), style = {'margin-bottom' : '5px', 'display' : 'inline'}),
        html.Br(),
        html.P("doi:", style = {'display' : 'inline','margin-bottom' : '5px'}),
        dcc.Link(row['doi'], href = 'https://doi.org/' + row['doi'], style = {'display' : 'inline','margin-bottom' : '5px'}),
    ]

#%%
#___Filtering_________

def abides_filter(doc_tags, filter_set):
    return len(filter_set.intersection(set(doc_tags))) > 0

def apply_filters(doc_tags, filter_sets):
    return all([
        abides_filter(doc_tags, filter_set) for filter_set in filter_sets
    ])

def get_filtersets(graph, filter_values):

    return [
        set([desc.lower() for desc in nx.descendants(graph, filter_val)])\
            .union(set([filter_val.lower()]))
        for filter_val in filter_values
    ]

def update_filters(filter_values):
    filter_sets = get_filtersets(filters_graph, filter_values)

    _data['visible'] = _data['disease_tags'].apply(lambda x : apply_filters(x, filter_sets))

#________________________________
#%%
#__Searching__________

def update_search_scores(search_value):

    query = parser.parse(search_value)

    with myindex.searcher() as searcher:
        results = searcher.search(query, limit = 50)
        results_df = pd.DataFrame([(r.fields()['id'], r.score) for r in results], columns = ['id','search_score'])

        scores = results_df['search_score']
        results_df['search_score'] = (scores - scores.min())/(scores.max()-scores.min())
        r = results_df.set_index('id')['search_score']

    _data['search_score'] = 0

    if not results_df.empty:
        _data.loc[r.index, 'search_score'] = r.values

    _data['agg_score'] = np.exp(_data.norm_pagerank) + 3 * np.exp(_data.search_score)


#%%
def update_suggestion_list():
    return html.Ol([
        html.Li(children = row_to_text(row[1]), style = {'margin-bottom' : '30px'})
        for row in _data.nlargest(n = 40, columns = ['visible','agg_score'], keep = 'first').iterrows()
    ], style = {'list-style-type':'none'})


@app.callback(
    Output('suggestions_list', 'children'),
    [
        Input('filtersbar', 'value'),
        Input('searchbar', 'value'),
    ]
)
def update_suggestions_callback(filter_values, search_value):

    ctx = dash.callback_context

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'filtersbar'

    if trigger_id == 'searchbar' and not search_value is None:
        update_search_scores(search_value)
    elif trigger_id == 'filtersbar':
        update_filters(filter_values)

    return update_suggestion_list()

#__Constellations______________
#%%

def get_top_sims(click_id):

    top_25 = sim_df[click_id].nlargest(n = 26, keep = 'first')[1:26].rename('similarity')
    top_25_rows = _data.loc[top_25.index]
    top_25_rows['similarity'] = top_25.values
    show = top_25_rows[(top_25_rows.visible == True) & (top_25_rows['similarity'] >= 0.25)]

    return show

def calc_constellation(click_id):

    row = _data.loc[click_id]

    connections = get_top_sims(click_id)

    origin_x, origin_y = row['published_timestamp'], row['norm_pagerank']

    connections_x, connections_y = list(zip(*[
        ([origin_x, x, None], [origin_y, y, None])
        for x, y in connections[['published_timestamp','norm_pagerank']].values
    ]))

    edge_x = [x for connection in connections_x for x in connection]
    edge_y = [y for connection in connections_y for y in connection]

    return (edge_x, edge_y)


def update_graph(constellation_data):

    graph_data = _data[_data['visible'] == True]

    newfig = go.Figure(
        layout = dict(
                        margin = {'l': 5, 'b': 0, 't': 5, 'r': 20},
                        font = {'size':12, 'family':'Courier New', 'color' : '#fff'},
                        plot_bgcolor = BACKGROUND_COLOR,
                        paper_bgcolor = BOX_COLOR,
                        xaxis_title = 'Publish Date',
                        yaxis_title = 'Impact',
                        showlegend = True,
                        uirevision = 'dont change',
                        legend_orientation = 'h',
                        legend = dict(
                            x = 0.5,
                            xanchor = 'center',
                        )
                    ),
    )

    if constellation_data:
        (edge_x, edge_y) = constellation_data
        newfig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.9, color='#fff'),
                hoverinfo='none',
                mode='lines',
                name = 'Constellation',
            )
        )

    for category_name, group in graph_data.groupby('category_description'):
        newfig.add_trace(
            go.Scatter(
                x = group['published_timestamp'], 
                y = group['norm_pagerank'],  
                opacity=1.0,
                marker_color = group['color_code'],
                marker_line_color = '#000',
                mode = 'markers',
                marker_size = group['agg_score'] * 2,
                text = group.index,
                name = category_name,
            ),
        )

    return newfig

@app.callback(
    Output('constellation','figure'),
    [
        Input('suggestions_list','children'),
        Input('constellation','clickData')
    ]
)
def update_graph_callback(suggestions_list, click_data):
    
    try:
        click_id = click_data['points'][0]['text']
    except (KeyError, TypeError):
        return update_graph(None)

    constellation_data = calc_constellation(click_id)

    return update_graph(constellation_data)


@app.callback(
    Output('abstract_box', 'children'),
    [Input('constellation', 'clickData')])
def update_abstract_box(click_data):
    if click_data is None:
        return [html.P('Select a paper from the constellation to learn more.')]
        
    #1 update abstract box
    point = click_data['points'][0]
    row = _data.loc[point['text']]
    new_children = row_to_text(row)
    new_children.extend([
        html.Br(),
        html.Br(),
        html.I('Abstract:'),
        html.P(row['abstract'], style = {'font-family' : 'Arial'}),
    ])
    return new_children


if __name__ == '__main__':
    app.run_server(debug=True)

