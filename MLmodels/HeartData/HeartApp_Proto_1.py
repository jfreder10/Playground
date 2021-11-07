import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go
import datetime as dt 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import datetime


heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW6\HeartDisease.csv")  

#figure out how to arrange th inputs so they dont look like shit 

app = dash.Dash('NCAA_FootballApp')

app.layout = html.Div(children=[
    html.H1('Jonathan Frederick'),
    
    html.H3('Heart App Proto'),

    #make several of these for each x var where the max is the max of the var and the min is the min 
    html.P(children=[
    html.Label('Choose a value for x1')
   ,dcc.Dropdown(id="input", options=[{'label': c, 'value': c } for c in heart['x1'].unique()] , value=np.mean(heart['x1']), multi=False)],	style = {'width': '400px'} ),



    # dcc.Graph(id='output-graph1')
])

# @app.callback(
#     Output(component_id='output-graph1', component_property='figure'),
#     [Input(component_id='input', component_property='value'),]
# )






if __name__ == '__main__':
    app.run_server(debug=True)