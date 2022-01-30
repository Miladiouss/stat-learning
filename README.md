# Statistical Learning

This is a collection of my computational notes on statistical learning, machine learning, and data science. It is new and growing. The topics are oriented for experienced python programmers with a strong mathematical background who may not be familiar with the field of statistical learning.

The following points make this collection unique that you may want to consider before continuing:

1. **Plotly**: I believe Plotly is a superior data visualization library. I will not be using Matplotlib since it doesn't provide the customization I need and I'm much more productive using Plotly. On top of that, Plotly is JS based and interactive; you cannot beat that.
2. **Scipy**: Majority of machine learning tutorials out there are using sk-learn. Instead, here I will try using the stats module of Scipy since it is more fundamental, has better documentation, and is more customizable. Having said that, I will use sk-learn whenever it is superior to Scipy.
3. **Data Simulation**: Most importantly, whenever possible, I will simulate the data in the code since I believe data simulation is one of the main pillars of data science: data wrangling, data simulation, modeling, visualization.

## Terminology

$$
\bm{y'} = f(\bm{x}; \bm{\theta})
$$

- Model ($f$)
    - `param` ($\bm{\theta}$) - a.k.a: model parameters, $\bm{\Theta}$
    - Modes:
        - Training (`'train'`)
        - Evaluation (`'eval'`)
- Data
    - `x` ($\bm{x}$) - a.k.a: model input, input data, input features , $X$
    - `target` ($\bm{y}$) - a.k.a: target, target feature, y_true, $Y$
    - `output` ($\bm{y\prime}$) - a.k.a: model output, output feature, y_pred, y_true, label
- Data sets
    - Training dataset (`train_ds`)
    - Validation dataset (`valid_ds`)
    - Test dataset (`test_ds`)
- Trining
    - Loss function (`loss`)
    - Deep Learning:
        - Forward Propagation
        - Backward Propagation
        - Optimization algorithm
        - Learning rate (`lr`, $\eta$)
        - Gradient
