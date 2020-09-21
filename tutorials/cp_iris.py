from gmmzoo.som import SOM
from sklearn.datasets import load_iris
import torch
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output

def _main():
    iris = load_iris()
    X = iris.data

    n_dim_latent=2
    n_grids = 20
    n_epoch = 10
    init='pca'
    shape_latent_space='unit_hypercube'
    schedule_sigma = {'max': 0.5, 'min': 0.1}

    som = SOM(X=torch.tensor(X),n_dim_latent=n_dim_latent,init=init,
              shape_latent_space=shape_latent_space,n_grids=n_grids,n_epoch=n_epoch,
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
            title=go.layout.Title(text='Latent space')
            # xaxis={'range': [som.ls.data[:, 0].min(), som.ls.data[:, 0].max()]},
            # yaxis={'range': [som.ls.data[:, 1].min(), som.ls.data[:, 1].max()]}
        )
    )
    fig_ls.add_trace(go.Scatter(x=som.ls.grids[:, 0], y=som.ls.grids[:, 1], mode='markers',
                                visible=True, marker_symbol='square', marker_size=10,
                                name='grid', opacity=0.5))
    fig_ls.add_trace(go.Scatter(x=som.ls.data[:, 0], y=som.ls.data[:, 1],
                                mode='markers', name='latent variable'))
    fig_bar = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text='Feature bars'),
            yaxis={'range': [0, X.max()]}
        )
    )
    fig_bar.add_trace(go.Bar(x=iris.feature_names, y=np.zeros(som.os.data.shape[1])))
    app.layout = html.Div(children=[
        # `dash_html_components`が提供するクラスは`childlen`属性を有している。
        # `childlen`属性を慣例的に最初の属性にしている。
        html.H1(children='Visualization iris dataset by SOM'),
        #html.Div(children='by component plance of SOM.'),
        # `dash_core_components`が`plotly`に従う機能を提供する。
        # HTMLではSVG要素として表現される。
        html.Div(
            [dcc.Graph(
                id='left-graph',
                figure=fig_ls
            )],
            style={'display': 'inline-block', 'width': '49%'}
        ),
        html.Div(
            [dcc.Graph(
                id='right-graph',
                figure=fig_bar
            )],
            style={'display': 'inline-block', 'width': '49%'}
        )
    ])
    @app.callback(
        Output(component_id='right-graph', component_property='figure'),
        [Input(component_id='left-graph', component_property='hoverData')]
    )
    def update_bar(hoverData):
        print(hoverData)
        if hoverData is not None:
            index = hoverData['points'][0]['pointIndex']
            if hoverData['points'][0]['curveNumber'] == 1:
                fig_bar.update_traces(y=som.os.data[index], marker=dict(color='#ff7f0e'))
            elif hoverData['points'][0]['curveNumber'] == 0:
                fig_bar.update_traces(y=som.os.grids[index], marker=dict(color='#1f77b4'))
        return fig_bar

    app.run_server(debug=True)
if __name__ == '__main__':
    _main()