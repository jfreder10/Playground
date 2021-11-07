import sklearn
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statistics as st
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from statsmodels  import regression as reg
import statsmodels.api as regMods
import dash_table as dt 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px



#make app where user can select the x any y levels of the item variable and output will be the statistics from the associationRule function, generalize to work for any
#transactional df 

#first make input template to make ava. to select multi levels per input 
steam=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\SteamGamesAssocRules\steam-200k.csv',header=None)
steam=steam.drop(4, axis=1)
steam.columns= ['custID', 'game', 'platVsOnlPur', 'HrPlay']

def associationRules(df,xlevels1,ylevels1):
  for i in range(0,len(xlevels1)):
    if(i==0):
      dfx1=steam[df['game'].isin([xlevels1[0]])]
      dfx1=dfx1['custID'].unique()
      dfx1=pd.DataFrame(dfx1)
      dfx1.columns=['custID']
    else:
      dfx21=df[df['game'].isin([xlevels1[i]])]
      dfx21=dfx21['custID'].unique()
      dfx21=pd.DataFrame(dfx21)
      dfx21.columns=['custID']
      dfx1=dfx1[dfx1['custID'].isin(dfx21['custID'])]
  for i in range(0,len(ylevels1)):
    if(i==0):
      dfy1=df[df['game'].isin([ylevels1[0]])]
      dfy1=dfy1['custID'].unique()
      dfy1=pd.DataFrame(dfy1)
      dfy1.columns=['custID']
    else:
      dfy21=df[df['game'].isin([ylevels1[i]])]
      dfy21=dfy21['custID'].unique()
      dfy21=pd.DataFrame(dfy21)
      dfy21.columns=['custID']
      dfy1=dfy1[dfy1['custID'].isin(dfy21['custID'])]
  priorprob=len(dfy1)/len(steam['custID'].unique())
  if(len(dfx1)<1):
    condprob=0
  else:
    condprob=len(dfx1[dfx1['custID'].isin(dfy1['custID'])])/len(dfx1)
  
  lift=condprob/priorprob
  stats=pd.DataFrame({'priorprobYLEVELS':[priorprob], 'condprobYGIVENXLEVELS': [condprob], 'lift':[lift]})
  return stats




#paste this in a py file to see if it runs but go to Dr. Ballings about how to dislay an app in python 
app = dash.Dash("SteamApp")

app.layout = html.Div(children=[
    html.H1('Choose several games per x and y'),
    html.Div(children='''
        X levels game:
    '''),
    #this only includes the unique levels from the game column as selections the first input 
    dcc.Dropdown(id="input", options=[{'label': c, 'value': c } for c in steam['game'].unique()]  , value="Choices",multi=True), #
    html.Div(children='''
        Y levels game:
    '''),
    dcc.Dropdown(id="input2", options=[{'label': c, 'value': c } for c in steam['game'].unique()]  , value="Choices",multi=True),
    dcc.Graph(id='output-graph'), 

    
])

@app.callback(
    Output(component_id='output-graph', component_property='figure'),
    [Input(component_id='input', component_property='value'),
    Input(component_id='input2', component_property='value'),]
    
    
)

#figure out how to fix this so that it changes for each added x level have to figure out how to update this successively 
def update_value(input_data,input2_data):
  xlevels=[input_data]
  ylevels=[input2_data]
  df1=associationRules(steam,xlevels[0],ylevels[0])
  fig = go.Figure(data=[go.Table(
        header=dict(values=list(df1.columns),
                fill_color='paleturquoise',
                align='left'),
        cells=dict(values=[df1.priorprobYLEVELS, df1.condprobYGIVENXLEVELS,df1.lift],
               fill_color='lavender',
               align='left'))
    ])
  return fig

if __name__ == '__main__':
    app.run_server(debug=True)



