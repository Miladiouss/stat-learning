# %%
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from dataclasses import dataclass

rng = np.random.default_rng(143339217099324139246236590887535066961)


@dataclass
class sim:
    ndata = 50
    slope = 2.5
    intercept = -1.2
    sigma_noise = 2.5
    xmin = -5
    xmax = 10


# setup a uniform distribution
uniform_dist = stats.uniform(sim.xmin, sim.xmax - sim.xmin)
uniform_dist.random_state = rng

# set up a normal distribution
norm_dist = stats.norm(0, sim.sigma_noise)
norm_dist.random_state = rng

# Sample X1 (aka x) and X2 (aka y):
X1 = uniform_dist.rvs(sim.ndata)
noise = norm_dist.rvs(sim.ndata)
X2 = sim.intercept + sim.slope * X1 + noise


# %% Visualize

# Visualize data points
trace0 = go.Scatter(
    x=X1, y=X2, name='data',
    mode='markers', line=dict(color='black'))

# Visualize simulation parameters
X1_sim = np.array([sim.xmin, sim.xmax])
X2_sim = sim.intercept + sim.slope * X1_sim

trace1 = go.Scatter(
    x=X1_sim, y=X2_sim, name='sim',
    line=dict(color='pink'))

traces = (trace0, trace1)
fig = go.Figure(traces)
fig


# %% Save the figure
fig.write_html('scatter-plot.html')
fig.write_image('scatter-plot.png')

# %%
