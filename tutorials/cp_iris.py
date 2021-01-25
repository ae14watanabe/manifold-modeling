from gmmzoo.som import SOM
from sklearn.datasets import load_iris
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output


def _main():
    iris = load_iris()
    X = iris.data

    n_dim_latent = 2
    n_grids = 30
    n_epoch = 10
    init = 'pca'
    shape_latent_space = 'unit_hypercube'
    schedule_sigma = {'max': 0.5, 'min': 0.1}

    som = SOM(X=torch.tensor(X), n_dim_latent=n_dim_latent, init=init,
              shape_latent_space=shape_latent_space, n_grids=n_grids, n_epoch=n_epoch,
              schedule_sigma=schedule_sigma)

    som.fit()
    # fig = plt.figure()
    # ax = fig.add_subplot(111,aspect='equal')
    # ax.scatter(som.ls.data[:, 0], som.ls.data[:, 1])
    # plt.show()

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    # ファイル名をアプリ名として起動。その際に外部CSSを指定できる。
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # この`layout`にアプリの外観部分を指定していく。
    # `dash_html_components`がHTMLタグを提供し、React.jsライブラリを使って実際の要素が生成される。
    # HTMLの開発と同じ感覚で外観を決めることが可能

    # fig = px.scatter(x=som.ls.data[:, 0], y=som.ls.data[:, 1])
    fig_ls = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text='Latent space'),
            # xaxis={'range': [som.ls.data[:, 0].min(), som.ls.data[:, 0].max()]},
            # yaxis={'range': [som.ls.data[:, 1].min(), som.ls.data[:, 1].max()]}
            showlegend=False
        )
    )
    fig_ls.add_trace(go.Contour(x=som.ls.grids[:, 0], y=som.ls.grids[:, 1],
                                z=som.os.grids[:, 0], colorscale='viridis',
                                line_smoothing=0.85,
                                contours_coloring='heatmap', name='cp'
                                )
                     )
    index_cp = 1
    fig_ls.add_trace(
        go.Scatter(x=som.ls.grids[:, 0], y=som.ls.grids[:, 1], mode='markers',
                   visible=True,
                   marker=dict(symbol='square', size=10, opacity=0.0,color='black'),
                   name='latent space')
    )

    fig_ls.add_trace(
        go.Scatter(
            x=som.ls.data[:, 0], y=som.ls.data[:, 1],
            mode='markers', name='latent variable',
            marker=dict(
                size=10,
                color=np.array(px.colors.qualitative.Plotly)[iris.target],
                line=dict(
                    width=2,
                    color="grey"
                )
            ),
            text=iris.target_names[iris.target]
        )
    )
    index_z = 2
    fig_ls.add_trace(
        go.Scatter(
            x=np.array(0.0), y=np.array(0.0),
            visible=False,
            marker=dict(
                size=10,
                symbol='x',
                color='black',
                line=dict(
                    width=1,
                    color="white"
                )
            ),
            name='clicked_point'
        )
    )
    fig_bar = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text='Feature bars'),
            yaxis={'range': [0, X.max()]}
        )
    )
    fig_bar.add_trace(go.Bar(x=iris.feature_names, y=np.zeros(som.os.data.shape[1])))

    config = {'displayModeBar': False}
    app.layout = html.Div(children=[
        # `dash_html_components`が提供するクラスは`childlen`属性を有している。
        # `childlen`属性を慣例的に最初の属性にしている。
        html.H1(children='Visualization iris dataset by SOM'),
        # html.Div(children='by component plance of SOM.'),
        # `dash_core_components`が`plotly`に従う機能を提供する。
        # HTMLではSVG要素として表現される。
        html.Div(
            [
                dcc.Graph(
                    id='left-graph',
                    figure=fig_ls,
                    config=config
                ),
                html.P('showd feature'),
                dcc.Dropdown(
                    id='feature_dropdown',
                    options=[{"value": i, "label": x}
                             for i, x in enumerate(iris.feature_names)],
                    value=0
                )
            ],
            style={'display': 'inline-block', 'width': '49%'}
        ),
        html.Div(
            [dcc.Graph(
                id='right-graph',
                figure=fig_bar,
                config=config
            )],
            style={'display': 'inline-block', 'width': '49%'}
        )
    ])

    # Define callback function when data is clicked
    @app.callback(
        Output(component_id='right-graph', component_property='figure'),
        Input(component_id='left-graph', component_property='clickData')
    )
    def update_bar(clickData):
        print(clickData)
        if clickData is not None:
            index = clickData['points'][0]['pointIndex']
            print('index={}'.format(index))
            if clickData['points'][0]['curveNumber'] == index_z:
                print('clicked latent variable')
                # if latent variable is clicked
                fig_bar.update_traces(y=som.os.data[index], marker=dict(color='#ff7f0e'))
                fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))
            elif clickData['points'][0]['curveNumber'] == index_cp:
                print('clicked map')
                # if contour is clicked
                fig_bar.update_traces(y=som.os.grids[index], marker=dict(color='#1f77b4'))
            # elif clickData['points'][0]['curveNumber'] == 0:
            #     print('clicked heatmap')
            return fig_bar
        else:
            return dash.no_update

    @app.callback(
        Output(component_id='left-graph', component_property='figure'),
        [Input(component_id='feature_dropdown', component_property='value'),
         Input(component_id='left-graph', component_property='clickData')]
    )
    def update_ls(index_selected_feature, clickData):
        # print(clickData)
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return dash.no_update
        else:
            clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]
            print(clicked_id_text)
            if clicked_id_text == 'feature_dropdown':
                print(index_selected_feature)
                fig_ls.update_traces(z=som.os.grids[:, index_selected_feature], selector=dict(type='contour'))
                return fig_ls
            elif clicked_id_text == 'left-graph':
                index_clicked = clickData['points'][0]['pointIndex']
                if clickData['points'][0]['curveNumber'] == index_cp:
                    # if contour is clicked
                    print('clicked map')
                    fig_ls.update_traces(
                        x=np.array(clickData['points'][0]['x']),
                        y=np.array(clickData['points'][0]['y']),
                        visible=True,
                        marker=dict(
                            symbol='x'
                        ),
                        selector=dict(name='clicked_point')
                    )
                elif clickData['points'][0]['curveNumber'] == index_z:
                    print('clicked latent variable')
                    fig_ls.update_traces(
                        x=np.array(clickData['points'][0]['x']),
                        y=np.array(clickData['points'][0]['y']),
                        visible=True,
                        marker=dict(
                            symbol='circle'
                        ),
                        selector=dict(name='clicked_point')
                    )
                    # if latent variable is clicked
                    # fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))
                return fig_ls
            else:
                return dash.no_update


    app.run_server(debug=True)


if __name__ == '__main__':
    _main()
