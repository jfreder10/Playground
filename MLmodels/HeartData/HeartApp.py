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
import itertools
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import t 
import plotly.express as px
import plotly.figure_factory as ff


heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW6\HeartDisease.csv")  

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

train, test = train_test_split(heart[['x1', 'x2', 'x3', 'x4', 'x5','HeartDisease']], test_size=0.2)
y_train=train['HeartDisease']
x_train=train[['x1', 'x2', 'x3', 'x4', 'x5']]



#set the range for the parameter values:
n_estimators=np.arange(300, 450, 50) #the number of trees to fit 
max_depth=np.arange(3, 5, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.001,0.004,0.001)
a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)

bestPos=8

bestMod=HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[bestPos],
    max_depth=params['Var2'].iloc[bestPos],
    learning_rate=params['Var4'].iloc[bestPos],max_iter=params['Var1'].iloc[bestPos]).fit(x_train, y_train)

#figure out how to arrange th inputs so they dont look like shit 

app = dash.Dash('HeartDisPred')


app.layout = html.Div([
            html.H1(children="Heart Disease Prediction",className="hello",style={
    'color':'#00361c','text-align':'right'}),
    html.Div([ #main div that holds all inputs 
           
           
           html.Div(children=dcc.Dropdown(id="input", options=[{'label': c, 'value': c } for c in heart['x1'].unique()] , value=np.mean(heart['x1']), multi=False),
                        style={
                        
                        'color':'steelblue',
                        'height':'100px',
                        'margin-left':'5px',
                        'width':'7%',
                        'text-align':'center',
                        'display':'inline-block'
                        }),
            
            html.Div(children=dcc.Dropdown(id="input2", options=[{'label': c, 'value': c } for c in heart['x2'].unique()] , value=np.mean(heart['x2']), multi=False),
               style={
                        
                        'color':'steelblue',
                        'height':'100px',
                        'margin-left':'5px',
                        'text-align':'center',
                        'width':'7%',
                        'display':'inline-block'
               }),


               html.Div(children=dcc.Dropdown(id="input3", options=[{'label': c, 'value': c } for c in heart['x3'].unique()] , value=np.mean(heart['x3']), multi=False),
               style={
                        
                        'color':'steelblue',
                        'height':'100px',
                        'margin-left':'5px',
                        'text-align':'center',
                        'width':'7%',
                        'display':'inline-block'
               }),
                         #seperates the inputs to disserent rows if ]) closes a html.Div then you put stuff in the most outer one 
            

            html.Div(children=dcc.Dropdown(id="input4", options=[{'label': c, 'value': c } for c in heart['x4'].unique()] , value=np.mean(heart['x4']), multi=False),
               style={
                        
                        'color':'steelblue',
                        'height':'100px',
                        'margin-left':'5px',
                        'text-align':'center',
                        'width':'7%',
                        'display':'inline-block'
               }),


            html.Div(children=dcc.Dropdown(id="input5", options=[{'label': c, 'value': c } for c in heart['x5'].unique()] , value=np.mean(heart['x5']), multi=False),
               style={
                        
                        'color':'steelblue',
                        'height':'100px',
                        'margin-left':'5px',
                        'text-align':'center',
                        'width':'7%',
                        'display':'inline-block'
               }),
            
            html.Div(id='pred'),

            dcc.Graph(id='graph1',style = {'height': '300px'}),
           
      ]),
       
      ])

@app.callback(dash.dependencies.Output('pred', 'children'),
              [dash.dependencies.Input('input', 'value'),
              dash.dependencies.Input('input2', 'value'),
              dash.dependencies.Input('input3', 'value'),
              dash.dependencies.Input('input4', 'value'),
              dash.dependencies.Input('input5', 'value')])


def page_3_dropdown(input,input2,input3,input4,input5):
    df_i=pd.DataFrame({'x1':input, 'x2':input2,'x3':input3,'x4':input4,'x5':input5},index=[0])
    a=bestMod.predict(df_i)
    if(a==0):
        b=': No Heart Disease'
    else:
        b=': Has Heart Disease'
    
    
    

    return 'Predicted: "{}"'.format(b)


@app.callback(dash.dependencies.Output('graph1', 'figure'),
              [dash.dependencies.Input('input', 'value'),])

def updateGraph1(input):
    x1=input
    mean1=np.mean(heart['x1'])
    sd1=np.std(heart['x1'])
    meanx1_2=x1
    xActual1=norm.rvs(size=len(heart),loc=mean1,scale=sd1)
    xInput1=norm.rvs(size=len(heart),loc=meanx1_2,scale=sd1)

    group_labels1 = ['actual','center_selected']
    hist_data1=[xActual1,xInput1]
    fig1 = ff.create_distplot(hist_data1,group_labels1)
    return fig1


if __name__ == '__main__':
    app.run_server(debug=True)


#for each variable plot the density centered on the mean to the closest distro it resembles, then when select a value plot the same distro but make the mean 
#of it the value that has been selected, plot the 2 curves together, start with treating each var as a normal distro then plot a density curve where the 
#mean is the mean of the var and another curve on same plot where the mean is the selected value from the input of a normal distro 
