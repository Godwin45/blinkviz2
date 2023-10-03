from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

app = Flask(__name__)


df = pd.read_csv('Sample Data.csv')
def separate_timestamp_and_weight(df):
    df[['Timestamp', 'WEIGHT']] = df['Timestamp;WEIGHT'].str.split(';', expand=True)
    df.drop(columns=['Timestamp;WEIGHT'], inplace=True)
    
    return df
df = separate_timestamp_and_weight(df)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['WEIGHT'] = pd.to_numeric(df['WEIGHT'], errors='coerce')

df_reset = pd.read_csv('Sample Data.csv')

def separate_timestamp_and_weight(df_reset):
    df_reset[['Timestamp', 'WEIGHT']] = df_reset['Timestamp;WEIGHT'].str.split(';', expand=True)
    df_reset.drop(columns=['Timestamp;WEIGHT'], inplace=True)
    return df_reset

df_reset = separate_timestamp_and_weight(df_reset)
df_reset['Timestamp'] = pd.to_datetime(df_reset['Timestamp'])

df_reset['WEIGHT'] = pd.to_numeric(df_reset['WEIGHT'], errors='coerce')
df_reset['Timestamp'] = df_reset['Timestamp'].astype('datetime64[ns]').view('int64')


# Create Graph 1
fig1 = px.line(df, x='Timestamp', y='WEIGHT', title='Weight Over Time')
fig1.update_traces(mode='lines+markers') 
fig1.update_layout(xaxis_title='Timestamp', yaxis_title='Weight', xaxis_tickangle=-45,template="plotly_dark")
graph1 = go.Figure(fig1)

# Create Graph 2
daily_averages = df.groupby(df['Timestamp'].dt.date)['WEIGHT'].mean().reset_index()
daily_averages['Timestamp'] = pd.to_datetime(daily_averages['Timestamp'])

fig2 = px.bar(daily_averages, x='Timestamp', y='WEIGHT', title='Daily Average Weight', color_discrete_sequence=['skyblue'])
fig2.update_layout(xaxis_title='Date', yaxis_title='Average Weight', xaxis_tickangle=-45,template="plotly_dark")

graph2 = go.Figure(fig2)

# Create 3 Graph
fig3 = px.histogram(df, x='WEIGHT', nbins=20, title='Weight Distribution', labels={'WEIGHT': 'Weight', 'count': 'Frequency'})
fig3.update_traces(marker=dict(line=dict(color='black', width=1)))
fig3.update_layout(xaxis_title='Weight', yaxis_title='Frequency',template="plotly_dark")
graph3 = go.Figure(fig3)
#graph4 
fig4 = px.box(df, x='WEIGHT', title='Weight Distribution (Box Plot)')
fig4.update_layout(xaxis_title='Weight',template="plotly_dark")
graph4 = go.Figure(fig4)

#graph 5
fig5 = px.scatter(df_reset, x='Timestamp', y='WEIGHT', trendline='ols', labels={'Timestamp': 'Timestamp', 'WEIGHT': 'Weight'})
fig5.update_traces(marker=dict(opacity=0.5))
fig5.update_traces(line=dict(color='red'))

fig5.update_layout(
    title='Weight Over Time with Regression Line',
    xaxis_title='Timestamp',
    yaxis_title='Weight',
    showlegend=False,
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    template="plotly_dark"
)

graph5 = go.Figure(fig5)


# Calculate autocorrelation
lag_values = np.arange(1, len(df['WEIGHT']))
autocorrelation_values = [df['WEIGHT'].autocorr(lag=lag) for lag in lag_values]

# Create a DataFrame for the autocorrelation values
autocorrelation_df = pd.DataFrame({'Lag': lag_values, 'Autocorrelation': autocorrelation_values})

# Create an autocorrelation plot using Plotly Express
fig6 = px.line(autocorrelation_df, x='Lag', y='Autocorrelation', title='Autocorrelation of Weight',template="plotly_dark")
fig6.update_xaxes(title='Lag (hours)')
fig6.update_yaxes(title='Autocorrelation')
graph6 = go.Figure(fig6)

df_reset['7-Day Moving Average'] = df_reset['WEIGHT'].rolling(window=7).mean()

# Create a line chart with Plotly
fig7 = go.Figure()

# Plot the original WEIGHT data
fig7.add_trace(go.Scatter(x=df_reset['Timestamp'], y=df_reset['WEIGHT'], mode='lines', name='WEIGHT', opacity=0.5))

# Plot the 7-Day Moving Average
fig7.add_trace(go.Scatter(x=df_reset['Timestamp'], y=df_reset['7-Day Moving Average'], mode='lines', name='7-Day Moving Average', line=dict(color='red')))

# Customize the layout
fig7.update_layout(
    title='Weight and 7-Day Moving Average',
    xaxis_title='Timestamp',
    yaxis_title='WEIGHT',
    xaxis=dict(tickangle=45),
    yaxis=dict(gridcolor='lightgray'),
    legend=dict(x=0, y=1),
    template="plotly_dark"  # You can change the template as needed
)

graph7 = go.Figure(fig7)
df_reset['Cumulative Weight'] = df_reset['WEIGHT'].cumsum()

# Create a cumulative sum area chart using Plotly Express
fig8 = px.area(df_reset, x='Timestamp', y='Cumulative Weight', title='Cumulative Weight Over Time')
fig8.update_xaxes(title='Timestamp')
fig8.update_yaxes(title='Cumulative Weight')

# Customize the layout
fig8.update_layout(template="plotly_dark")  
graph8 = go.Figure(fig8)

df['Weight Change'] = df['WEIGHT'] - df['WEIGHT'].shift(1)

# Create a waterfall chart using Plotly
trace = go.Bar(
    x=df['Timestamp'],
    y=df['Weight Change'],
    text=df['Weight Change'],
    marker=dict(
        color=df['Weight Change'].apply(lambda x: 'green' if x > 0 else 'red'),
    ),
)

data = [trace]

layout = go.Layout(
    title='Weight Changes Over Time (Waterfall Chart)',
    xaxis=dict(title='Timestamp'),
    yaxis=dict(title='Weight Change'),
    xaxis_tickangle=-45,
    showlegend=False,
    template="plotly_dark", 
)

fig9 = go.Figure(data=data, layout=layout)
graph9 = go.Figure(fig9)




@app.route('/')
def index():
    return render_template('index.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4, graph5=graph5, graph6=graph6, graph7=graph7, graph8=graph8, graph9=graph9)

if __name__ == '__main__':
    app.run(debug=True)
