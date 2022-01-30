# random-forest
# %%
# Key dependencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Standard dependencies
import plotly.graph_objects as go
import numpy as np
# from dataclasses import dataclass
# from scipy import stats

rng = np.random.default_rng(143339217099324139246236590887535066961)

style_dict = {
    'layout.plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'layout.font.family': 'Times New Roman',
    'layout.xaxis.linecolor': 'black',
    'layout.xaxis.ticks': 'inside',
    'layout.xaxis.mirror': True,
    'layout.xaxis.showline': True,
    'layout.yaxis.linecolor': 'black',
    'layout.yaxis.ticks': 'inside',
    'layout.yaxis.mirror': True,
    'layout.yaxis.showline': True,
}

# %%
# > Generate dataset
dim = 2
xdata, ydata = make_classification(
    n_samples=1000,
    n_classes=dim,
    n_features=dim,
    n_informative=dim,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=0,)


# %%
# > Visualize data points

trace0 = go.Scatter(
    x=xdata[:, 0][ydata == 0],
    y=xdata[:, 1][ydata == 0],
    name='0', mode='markers', line=dict(color='green'))

trace1 = go.Scatter(
    x=xdata[:, 0][ydata == 1],
    y=xdata[:, 1][ydata == 1],
    name='1', mode='markers', line=dict(color='pink'))

traces = (trace0, trace1)
fig = go.Figure(traces, **style_dict)
fig

# %%
# Split data
x_train, x_valid, y_train, y_valid = train_test_split(
    xdata, ydata, random_state=42)
# Define model
model = RandomForestClassifier(max_depth=2, random_state=0)
# Train model
model.fit(x_train, y_train)
# Evaluate model
y_pred = model.predict(x_valid)
print(classification_report(y_valid, y_pred))

# %%
