
#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly
import pandas as pd
import numpy as np
import os
import json
import networkx as nx
import datetime
from dash.dependencies import Input, Output, State
from import_data import import_data
from datetime import datetime as dt

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

_groups = _data.groupby('label')

num_groups = len(_groups)

#%%
#__Load in taxonomy_____
graph_data = pd.read_csv(os.path.join('.','taxonomy','graph_data.csv'))\
    .drop(columns = ['Unnamed: 0'])

with open(os.path.join('.','taxonomy','edge_data.json'), 'r') as f:
    edge_data = json.loads(f.read())

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

print('Data loaded!')

#%%
#___FIGURES_____________
BACKGROUND_COLOR = '#1F2132'
BOX_COLOR = '#262B3D'

#__TAB STYLE______________

def get_tab_style(color1, color2, side = 'middle', height = 50):

    tab_style = {
        'background-color' : color1,
        'color' : '#fff',
        'font-family' : "Courier New",
        'border' : '3px solid ' + color2,
        'font-size' : '2.15rem',
        'padding-top' : '4px',
        'height' : '{}px'.format(height),
        'border-radius' : '0px 0px 0px 0px',
    }
    if side == 'left':
        tab_style['border-radius'] = '{0}px 0px 0px {0}px'.format(height//2)
    elif side == 'right': 
        tab_style['border-radius'] = '0px {0}px {0}px 0px'.format(height//2)

    return tab_style

#____Components_________________

START_INFO = html.Div(className = 'infobox', id='explanation_infobox', children = [
                            html.H1('Launch Instructions', className = 'infobox_header'),
                            html.P('''Welcome to Constellation! This dashboard is intended to use Natural Language Processing and Machine Learning to help Biomedical researchers 
                            investigate a trove of almost 24,000 COVID-19 and infectious-disease-related articles. Below is your sky plot, showing each paper 
                            as a star according to its importance in the corpus and its publish date. Click on a paper to see other papers with similar topics, explore
                            different categories, and when you've found some documents you like, save them and use the summary extraction feature to get aquainted quickly.''',
                            style = {'margin-bottom' : '8px'}),
                            html.Button('Got It!', id = 'got_it', style = {'margin-bottom' : '8px', 'margin' : 'auto'}),
                        ])

ABSTRACT_BOX = html.Div([
                        html.Div([
                            html.H1('Paper Information', className = 'infobox_header', style = {'float' : 'left', 'width' : '90%'}),
                            dcc.ConfirmDialogProvider(
                                html.Button(id = 'save_paper', children = 'Save Paper', style = {'font-family' : "Courier New", 'float' : 'right', 'font-size' : '1.6rem'}),
                                id = 'confirm_saved_paper',
                                message = 'Save this paper for summarization?',
                            ),
                        ]),
                        html.Div(style = {'clear' : 'both'}),
                        html.Div(children = [], id = 'abstract_box', style = {'margin-left' : '20px', 'margin-right' :'20px'}),
                ], className='infobox', style = {'display': 'block', 'overflow': 'auto', 'height' : '25vh', 'margin-top' : '15px'})

CONSTELLATION_LAYOUT = dict(
                            margin = {'l': 55, 'b': 40, 't': 30, 'r': 35, 'pad' : 0},
                            font = {'size':16, 'family':'Courier New', 'color' : '#fff'},
                            plot_bgcolor = BACKGROUND_COLOR,
                            paper_bgcolor = BOX_COLOR,
                            xaxis_title = 'Publish Date',
                            xaxis_gridcolor = '#858585',
                            yaxis = dict(
                                ticks = "outside", 
                                tickcolor=BOX_COLOR, 
                                ticklen=10, 
                                showticklabels = False,
                                title = 'Impact',
                                gridcolor = '#858585'),
                            showlegend = True,
                            legend_orientation="h",
                            uirevision = 'dont change',
                        )

CONSTELLATION = dcc.Graph(id = 'constellation',
                        style = {'height':'76vh', 'margin' : '0px'},
                        figure = dict(
                            layout = CONSTELLATION_LAYOUT
                        ),
                    )

LIT_LIST = html.Div(children = [], id = 'suggestions_list', style = {'margin-left' : '15px', 'margin-right' :'15px', 
                    'overflow-y' : 'scroll', 'height' : '77vh'})

TAXONOMY_LAYOUT = dict(
        showlegend = False,
        plot_bgcolor = BOX_COLOR,
        paper_bgcolor = BOX_COLOR,
        xaxis = dict(
            showticklabels = False,
            title = '',
            showgrid = False,
            showline = False,
            zeroline = False,
            ),
        yaxis = dict(
            showticklabels = False,
            title = '',
            showgrid = False,
            showline = False,
            zeroline = False,
            ),
        margin=dict(l=0,r=0,b=0,t=0,pad=4)
    )

TAXONOMY = dcc.Graph(id = 'taxonomy',
                        style = {'height':'40vh', 'margin' : '12px'},
                        figure = dict(
                            layout = TAXONOMY_LAYOUT
                        ),
                    )

SEARCH_BAR = html.Div([
                html.Label('Search'),
                html.Br(),
                html.Div([
                    dcc.Input(id="searchbar", type="text", placeholder="", debounce=True,
                    style = {'margin-right' : '5px', 'height' : '36px','display' : 'inline-block', 'width':'85%', 'font-size': '1.6rem',
                                'background-color' : BOX_COLOR, 'border-color' : '#fff','border-radius' : '2px', 'border-width' : '1px'}),
                    #html.Button('Submit'),
                ], style = {'margin-top' : '0px', 'margin-bottom' : '15px'})
            ], className = 'column')
            
FILTERS = html.Div([
            dcc.Dropdown(
                options=[
                    {'label' : node_name, 'value' : node_name}
                    for node_name in taxonomy.nodes                                    
                ],
                value=[],
                multi=True,
                id = 'filtersbar',
                style = {'background-color' : BOX_COLOR, 'border-color' : '#fff', 'border-width' : '1px',
                    'font-family' : "Courier New", 'font-color' : '#fff', 'border-radius' : '2px',
                    'width' : '90%', 
                    'margin-left' : '0px', 'margin-right' : '0px',
                },
                placeholder="Select a Filter",
            ),
        ], id="wrapper", style = {'clear' : 'both', 'margin-bottom' : '15px'})

SUMMARY_SELECT = html.Select(
                    id = 'saved_papers',
                    size = 10,
                    children = [html.Option(value = '3', label = 'Test label for testing purposes only'), html.Option(value = '3', label = 'Another Test label for testing purposes only')],
                    style = {'width' : '94%', 'height' : '88%', 'margin' : '20px'}
                )

DATA_DIVS = html.Div(id='user_cache', style = {'display' : 'none'}, children = [
                html.Div(id = 'filter_terms', style = {'display' : 'none'}),
                html.Div(id = 'visibility', style = {'display' : 'none'}),
                html.Div(id = 'legend_tracker', style = {'display' : 'none'}),
                html.Div(id = 'search_scores', style = {'display' : 'none'}),
                #html.Div(id = 'const_data', style = {'display' : 'none'}),
                html.Div(id = 'active_paper',style = {'display' : 'none'}),
            ])
#__Layout_______________
app.layout = html.Div(children=[
    DATA_DIVS,
    dcc.ConfirmDialog(
        id='save_confirm',
        message='Saved paper!',
    ),
    #Title Bar
    html.Img(
            src=app.get_asset_url('logo.png'),
            height = 100,
            style = {'margin-left':'25px', 'margin-top':'0px', 'padding-top':'0px'}
    ),
    dcc.Tabs(id = 'main_tabs', value = 'exploration', children = [
        #explore tab
        dcc.Tab(label = 'Explore', value = 'exploration', style = get_tab_style(BOX_COLOR, BOX_COLOR,side = 'left'), selected_style = get_tab_style(BACKGROUND_COLOR, BOX_COLOR,side = 'left'), children = [
            html.Div([        
                #Left panel
                html.Div([
                    START_INFO,
                    html.Div([
                        html.H1('Control Panel', className = 'infobox_header'),
                        html.Div(className= 'row', children = [
                            html.Div(className= 'column', children = [
                                html.Label('Concept Taxonomy'),
                                TAXONOMY
                            ]),
                            html.Div(className = 'column', children = [
                                html.Label('Filter by Taxonomy'),
                                FILTERS,
                                html.P('Choose a filter term from the dropdown above, or directly from the taxonomy map. Constellation will find papers that contain the selected filter, along with all its sub-concepts',
                                    style = {'width' : '90%', 'font-size' : '1.6rem'}),
                                html.Br(),
                                SEARCH_BAR,
                                html.Br(),
                                html.Label('See Papers Since'),
                                html.Br(),
                                dcc.DatePickerSingle(
                                    id = 'data_picker',
                                    min_date_allowed=dt(1975, 1, 1),
                                    max_date_allowed=dt(2021, 1, 1),
                                    initial_visible_month=dt(1975, 1, 1),
                                    date=str(dt(1975, 1, 1)),
                                    style = {'width' : '50%'},
                                ),
                            ])
                        ]),
                    ], className = 'infobox'),
                    ABSTRACT_BOX,
                ], className = 'column'),
                #right panel
                html.Div([
                    html.Div([
                        html.H1('Literature', className = 'infobox_header'),
                        dcc.Tabs(id = 'literature_search_tabs', value = 'constellation_tab', children = [
                            dcc.Tab(label = 'Constellation', value = 'constellation_tab', style = get_tab_style(BOX_COLOR, BACKGROUND_COLOR,side = 'left', height = 40), selected_style = get_tab_style(BACKGROUND_COLOR, BACKGROUND_COLOR,side = 'left', height = 40), children =[
                                CONSTELLATION,
                                html.P('If your constellations mysteriously disappear, make sure to reactivate the "Constellation" trace on the legend!', style = {'text-align' : 'center', 'font-size' : '1.6rem', 'margin-top' : '20px'}),
                                html.P('If it still doesn\'t appear, there might be no similar papers in your view.', style = {'text-align' : 'center', 'font-size' : '1.6rem'}),
                            ]),
                            dcc.Tab(label = 'List', value = 'literature_list_tab', style = get_tab_style(BOX_COLOR, BACKGROUND_COLOR,side = 'right', height = 40), selected_style = get_tab_style(BACKGROUND_COLOR, BACKGROUND_COLOR,side = 'right', height = 40), children = [
                                LIT_LIST
                            ])
                        ])
                    ],
                    className = 'infobox', style = {'height' : '86vh'}),
                ], className = 'column',), 
            ], className = 'row', style = {'background-color' : BACKGROUND_COLOR, 'border' : 'none', 'padding' : '0px'}),
        ]),
        dcc.Tab(value = 'extract', label = 'Summarize', style = get_tab_style(BOX_COLOR, BOX_COLOR,side = 'middle'), selected_style = get_tab_style(BACKGROUND_COLOR, BOX_COLOR, side = 'middle'), children = [
            html.Div(children = [
                html.Div(style = {'float' : 'left', 'width' : '33%', 'height' : '90vh'}, children = [
                    html.Div(className = 'infobox', children = [
                        html.H1('Summary Information', className = 'infobox_header'),  
                        html.P('''This summarization module extracts important sentences from a chosen set of documents. Those sentences serve as a summary, and aim to provide 
                            the most holistic review of the most important points shared by that body of documents. For a good summary, add 4-10 papers that share similar 
                            topics or categories. If papers are too disimilar, the summary quality will suffer. If you would like to know more about how summary extraction works, visit the "Models" tab.'''),
                    ], style = {'width' : '100%',}),
                    html.Div(className = 'infobox', children = [
                        html.H1('Saved Papers', className = 'infobox_header'),
                        html.Button(id = 'remove_paper', children = 'Remove', style = {'font-family' : "Courier New", 'font-size' : '1.6rem', 'width' : '35%', 'margin-left' : '20px'}),
                        html.Button(id = 'summarize', children = 'Summarize All ->', style = {'font-family' : "Courier New", 'font-size' : '1.6rem', 'width' : '35%', 'margin-left' : '20px'}),
                        SUMMARY_SELECT,
                    ], style = {'width' : '100%', 'height' : '50vh'}),
                ]),
                html.Div(style = {'float':'right', 'width' : '66%'},children = [
                    html.Div(className = 'infobox', children = [
                        html.H1('Summary', className = 'infobox_header'),
                        html.P('''No summary here yet! Use the "Explore" tab to find interesting papers, then save them to your summary list. When you have found 5-10 papers, hit "Summarize All" to find the most important sentences capturing the ideas of that research.''', 
                            style = {'margin-bottom' : '30px'}),
                    ], style = {'margin-left' : '30px', 'padding' : '20px'}),
                ]),
            ], style = {'clear' : 'both'}),                
        ]),
        dcc.Tab(value = 'models', label = 'Models', style = get_tab_style(BOX_COLOR, BOX_COLOR, side = 'right'), selected_style = get_tab_style(BACKGROUND_COLOR, BOX_COLOR,side = 'right'), children = [
            html.Div([
                html.Div([
                    html.Div([
                        html.H1('Document Similarity and Categorization', className = 'infobox_header'),
                        html.Iframe(src = app.get_asset_url('abstract_tfidf_vectorization.html'), style = {'width' : '97%', 'height' : '40vh', 'border-color' : BOX_COLOR, 'margin' : '20px'})
                    ], className = 'infobox', style = {'width' : '95%'}),
                    html.Div([
                        html.H1('Literature Impact with PageRank', className = 'infobox_header'),
                        html.Iframe(src = app.get_asset_url('network_analysis.html'), style = {'width' : '97%', 'height' : '40vh', 'border-color' : BOX_COLOR, 'margin' : '20px'})
                    ], className = 'infobox', style = {'width' : '95%'}),
                ], className = 'column'),
                html.Div([
                    html.Div([
                        html.H1('Disease Taxonomy Generation', className = 'infobox_header'),
                        html.Iframe(src = app.get_asset_url('entity_recognition.html'), style = {'width' : '97%', 'height' : '40vh', 'border-color' : BOX_COLOR, 'margin' : '20px'})
                    ], className = 'infobox', style = {'width' : '95%'}),
                ], className = 'column')
            ], className = 'row'),
        ])
    ], style = {'margin-bottom' : '10px'}),  
])

#%%
#Literature suggestion fetching
def row_to_text(row):
    return [
        html.H1(row['title'], style = {'font-size' : '2.13rem', 'margin-bottom' : '5px'}),
        html.P('{}, {} et al., {}'.format(row['pretty_date'], row['first_author'], row['source_x']), style = {'margin-bottom' : '5px', 'display' : 'inline'}),
        html.Br(),
        html.P('Category: {}'.format(row['category_description']), style = {'margin-bottom' : '5px', 'display' : 'inline'}),
        html.Br(),
        html.P("doi:", style = {'display' : 'inline','margin-bottom' : '5px'}),
        dcc.Link(row['doi'], href = 'https://doi.org/' + str(row['doi']), style = {'display' : 'inline','margin-bottom' : '5px'}),
    ]
#%%
#___Filtering_________
def draw_taxonomy(active_nodes):
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    #seperate into two traces for active nodes
    node_filter = graph_data.node.isin(active_nodes)

    NODE_SIZE = 7
    active_trace = go.Scatter(
        x = graph_data[node_filter].x,
        y = graph_data[node_filter].y,
        text = graph_data[node_filter].node,
        mode = 'markers',
        marker_size = 1.7 * NODE_SIZE,
        marker_color = '#f5c15f',
    )
    inactive_trace = go.Scatter(
        x = graph_data[~node_filter].x,
        y = graph_data[~node_filter].y,
        text = graph_data[~node_filter].node,
        mode = 'markers',
        marker_size = NODE_SIZE,
        marker_color = '#b9e2ed',
    )
    fig = go.Figure()
    fig.update_layout(TAXONOMY_LAYOUT)
    fig.add_traces([edge_trace, active_trace, inactive_trace])
    return fig

@app.callback(
    Output('filtersbar','value'),
    [Input('taxonomy','clickData')],
    [State('filtersbar','value')],
)
def update_filtersbar(click_data, filter_state):
    if click_data is None:
        return filter_state
    try:
        new_filter = click_data['points'][0]['text']
    except KeyError:
        raise PreventUpdate()

    if new_filter in filter_state:
        raise PreventUpdate() #if already clicked
    else:
        filter_state.append(new_filter)
    return filter_state

@app.callback(
    [Output('taxonomy', 'figure'), Output('filter_terms','children')],
    [Input('filtersbar', 'value')],
)
def update_taxonomy(filter_state):
    active_filters = set([
        descendent
        for filter_val in filter_state for descendent in nx.descendants(taxonomy, filter_val)
    ]).union(set(filter_state))
    return draw_taxonomy(active_filters), json.dumps(list(active_filters))

@app.callback(
    Output('legend_tracker','children'),
    [Input('constellation','restyleData')],
    [State('legend_tracker','children'), State('active_paper','children')]
)
def update_legend_tracker(restyle_event, legend_state, active_paper):

    if legend_state is None:
        legend_state = {str(i) : True for i in range(num_groups + 1)}
    else:
        legend_state = json.loads(legend_state)

    if restyle_event is None:
        return json.dumps(legend_state)

    for new_status, group_num in zip(restyle_event[0]['visible'], restyle_event[1]):
        legend_state[str(group_num)] = new_status == True

    return json.dumps(legend_state)


def abides_filter(tags, filters):
    if len(filters) == 0:
        return True
    return len(filters.intersection(set(tags))) > 0

def is_visible(row, filters, min_date):
    return abides_filter(row.disease_tags, filters) and (min_date is None or row.published_timestamp >= min_date)

@app.callback(
    Output('visibility','children'),
    [Input('filter_terms','children'), Input('data_picker','date')]
)
def update_visibility(filter_values, min_date):
    
    if filter_values is None:
        filters = []
    else:
        filters = set(json.loads(filter_values))

    visible_mask = _data.apply(lambda x : is_visible(x, filters, min_date), axis = 1)

    visible_mask = list(visible_mask[visible_mask].index)

    return json.dumps(visible_mask)

@app.callback(
    Output('search_scores','children'),
    [Input('searchbar','value')]
)
def update_search_scores(search_value):

    if search_value is None or search_value is '':
        return json.dumps({})

    query = parser.parse(search_value)

    with myindex.searcher() as searcher:
        results = searcher.search(query, limit = 100)
        results_dict = {r.fields()['id'] : r.score for r in results}

    results_series = pd.Series(results_dict)
    results_series = (results_series - results_series.min())/(results_series.max() - results_series.min())
    return json.dumps(results_series.to_dict())

def get_top_sims(click_id):
    top_25 = sim_df[click_id].nlargest(n = 26, keep = 'first')[1:26].rename('similarity')
    return top_25.index

def is_legend_visible(legend_state, label):
    return legend_state[str(int(label + 1))]

def update_suggestion_list():
    return 

def get_fig_and_list(visible_mask, search_scores, active_paper_id, legend_state):

    fig = go.Figure()

    if active_paper_id:
        
        connections_made = 0
        row = _data.loc[active_paper_id]
        origin_x, origin_y = row['published_timestamp'], row['norm_pagerank']
        
        connections = get_top_sims(active_paper_id)
        edge_x, edge_y = [], []
        for connection_id in connections:
            if connection_id in visible_mask and is_legend_visible(legend_state, _data.loc[connection_id].label):
                x, y = _data.loc[connection_id][['published_timestamp','norm_pagerank']].values
                edge_x.extend([origin_x, x, None])
                edge_y.extend([origin_y, y, None])
                connections_made += 1

        if connections_made > 0:

            fig.add_trace(
                go.Scattergl(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.9, color='#fff'),
                    hoverinfo='none',
                    mode='lines',
                    name = 'Constellation',
                ))

    if active_paper_id is None or connections_made == 0:

        if len(visible_mask) == 0:
            fake_data = _data.iloc[0]
        else:
            fake_data = _data.loc[visible_mask[0]]

        fig.add_trace((
            go.Scattergl(
                x = [fake_data.published_timestamp], 
                y = [fake_data.norm_pagerank],  
                hoverinfo = 'none',
                mode = 'lines',
                marker_color = '#fff',
                marker_size = 0,
                name = 'Constellation',
                visible = 'legendonly'
            )
        ))

    candidate_papers = pd.DataFrame()

    for i, (category_name, category_group) in enumerate(_groups):

        vis_in_category = category_group[category_group.index.isin(visible_mask)]

        graph_group = vis_in_category.join(search_scores, how = 'left', sort = False)
        graph_group['search_score'].fillna(0, inplace = True)

        if is_legend_visible(legend_state, i):
            if search_scores.empty:
                search_hits = graph_group
            else:
                search_hits = vis_in_category.join(search_scores, how = 'inner', sort = False)

            search_hits['agg_score'] = np.exp(search_hits.norm_pagerank + search_hits.search_score)
            candidate_papers = candidate_papers.append(search_hits)

        fig.add_trace(
            go.Scattergl(
                x = graph_group.published_timestamp, 
                y = graph_group.norm_pagerank,
                marker_color = plotly.express.colors.qualitative.Light24[i],
                marker_line_color = plotly.express.colors.qualitative.Light24[i],
                mode = 'markers',
                marker_size = np.exp(graph_group.search_score) * 8,
                text = graph_group.index,
                name = category_group.iloc[0].category_description,
            ),
        )

    suggestion_list = html.Ol([
        html.Li(children = row_to_text(row[1]), style = {'margin-bottom' : '30px'})
        for row in candidate_papers.nlargest(n = 50, columns = ['agg_score'], keep = 'first').iterrows()
    ], style = {'list-style-type':'none', 'margin-top' : '20px'})

    fig.update_layout(CONSTELLATION_LAYOUT)
    return fig, suggestion_list

@app.callback(
    [Output('constellation','figure'), Output('suggestions_list','children')],
    [Input('visibility','children'), Input('search_scores','children'), Input('active_paper','children'), Input('legend_tracker','children')]
)
def update_paper_representations(vis_data, search_scores, active_paper_id, legend_state):
 
    visible_mask = json.loads(vis_data)

    if search_scores is None:
        search_scores = pd.Series({})
    else:
        search_scores = pd.Series(json.loads(search_scores))
    search_scores.name = 'search_score'

    if legend_state is None:
        raise PreventUpdate()

    legend_state = json.loads(legend_state)

    return get_fig_and_list(visible_mask, search_scores, active_paper_id, legend_state)


#__Constellations______________
@app.callback(
    Output('active_paper', 'children'),
    [Input('constellation','clickData'), Input('visibility','children')]
)
def update_active_paper(click_data, vis_data):
    if vis_data is None:
        raise PreventUpdate()
    elif click_data is None:
        return None
    try:
        click_id = click_data['points'][0]['text']
        visible_nodes = json.loads(vis_data)
        return click_id if click_id in visible_nodes else None
    except (KeyError):
        return None

def update_abstract_box(active_paper_id):
    row = _data.loc[active_paper_id]
    new_children = row_to_text(row)
    new_children.extend([
        html.Br(),
        html.Br(),
        html.I('Abstract:', style = {'font-size' : '1.6rem'}),
        html.P(row['abstract'], style = {'font-family' : 'Arial', 'font-size' : '1.8rem'}),
    ])
    return new_children

@app.callback(
    Output('abstract_box', 'children'),
    [Input('active_paper', 'children')]
)
def update_click_results(active_paper_id):

    if active_paper_id is None:
        return html.P('Select a paper from the constellation to learn more.')
    return update_abstract_box(active_paper_id)


@app.callback(
    Output('saved_papers','children'),
    [Input('remove_paper','n_clicks'), Input('confirm_saved_paper', 'submit_n_clicks')],
    [State('active_paper','children'), State('saved_papers','children')]
)
def update_saved_papers(remove_paper, click_times, active_paper_id, options):

    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate()
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'remove_paper':

        if options is None or len(options) == 0:
            raise PreventUpdate()

        def get_timestamp(option):
            try:
                return option['props']['n_clicks_timestamp']
            except KeyError:
                return 0

        click_timestamps = [get_timestamp(option) for option in options]

        most_recent_click = np.argmax(click_timestamps)

        if click_timestamps[most_recent_click] > 0:
            del options[most_recent_click]

        return options
    
    elif button_id == 'confirm_saved_paper':
        
        if active_paper_id:
            if options is None:
                options = []
            options.append(html.Option(value = active_paper_id, label = _data.loc[active_paper_id]['title']))

        return options

#%%
#__Searching__________

@app.callback(
    Output('explanation_infobox','style'),
    [Input('got_it','n_clicks')]
)
def got_it_callback(n_clicks):
    if not n_clicks is None and n_clicks > 0:
        return {'display' : 'none'}
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)

