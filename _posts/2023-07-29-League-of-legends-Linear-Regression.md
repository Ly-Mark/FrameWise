---
layout : post
title: League of Legends - Exploring the Relationship Between Kills and Win Rate
description:  Part 2 - Linear Regression
author: mark
image:  '/images/06.jpg'
video_embed:
tags:   [Supervised Learning, Statistical Analysis]
tags_color: '#4c49cb'
featured: true
---

## Previously

In our [previous article](https://ly-mark.github.io/blog/league-of-legends-eda), we successfully prepared our dataset of historical player data spanning the 2011 to 2022 Worlds Championships of League of Legends.
Our primary objective is to answer the following question:

> What is the connection between a player's average number of kills per game and their corresponding win rate?

Based on the insights from the earlier correlation heatmap, we've identified several variables with notable correlations, one of which is **`kills`**. This paves the way for our initial attempt to accurately predict win rate. To accomplish this, the first model we will construct is a **linear regression**.

## Linear Regression

> Linear regression is a supervised machine learning algorithm.

Supervised algorithms are like mentors that learn from previous experiences to make intelligent predictions about the future. Imagine that you have a large collection of data about houses including their sizes and price. If we wanted to understand how housing size influences the cost, we could create a graph of these two variables and see if we can draw a line that best fits the data points. The line that is drawn represents a pattern in the data of how house size relates to house prices.

With this line, we have the new ability to predict the price of a new house based on its size. Let's say we find a new 2000 square foot house that is currently being built. We can estimate the price of this house by using the line on the graph. It's like saying, "Based on the sizes and prices of other houses, I predict that this 2000 square feet house will cost $250,000."

![HousingLinearRegression]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-housing-scatterplot.png){:style="display:block; margin-left:auto; margin-right:auto"}
*Visualization of a house size and house price linear regression algorithm*

> The magic of supervised learning lies in its ability to generalize from the training data to make predictions on new, unseen data.

Supervised algorithms don't memorize all the data points, which would be an example of overfitting, but rather, it learns the underlying relationship between the two variables of interest.

Linear regression is great for predicting continuous values (like house prices); however, there are multiple types of regression that can handle different data types. The fundamental idea of supervised learning is the same - learning from labeled data to make informed prediction about unseen examples. Let's attempt to use linear regression to predict **`win rate`** with the number of **`kills`**.

### Linear Regression assumptions

For linear regression to deliver good results, we have to make sure that our dataset satisfies certain assumptions. An easy way to remember these assumptions is using the **LINE** acronym. This includes:

1. **L**inearity: The linearity assumption assumes that there is a linear relationship between our independent (**`kills`**) and dependent variable (**`win rate`**). When the independent variable changes the dependent variable will change proportionally in a straight line.
2. **I**ndependence: The independence assumption assumes that each data point is independent of each other. This means that the data should not be influenced by any hidden factors related in a way that would skew the results.
3. **N**ormality: The normality assumption assumes that the residuals follow a normal distribution. Residuals are the difference between the actual values and the predicted values by the linear regression model.
4. **E**qual Variance: The equal variance model assumes that the spread of the data points should be consistent along a straight line. This is also commonly referred to as Homoscedasticity.

We also have to check for multicollinearity, which is having multiple variables that are strongly correlated with each other.

After building our model, we will check to make sure that we satisfy our assumptions using regression plots and some statistical tests.

## Gameplan

We will break down the steps of this project into 5 steps:

1. **Data Preparation and Exploration**: This was completed in Part 1: Data Wrangling and Exploratory Data Analysis
2. **Data Preprocessing**: Most of this was completed in Part 1: Data Wrangling and Exploratory Data Analysis. We will subset our data and make sure we deal with any further issues.
3. **Linear Regression Modeling**: We will build our model using *scikit-learn* library.
4. **Interpretation and Visualization**: We will check our model statistically and visualize our data.
5. **Assumption checking**: As mentioned above, we will make sure of regression plots and statically tests to make sure that we statisfy the assumptions for linear regression.

## Packages
This project will utilize the following packages for data analysis, visualization, and model creation:

1. pandas - Data wrangling
2. numpy - Filling in missing data
4. seaborn - Data Visualization
5. plotly - Data Visualization
6. matplotlib - Data Visualization 
7. sklearn - Model creation
8. statsmodels - Statistical Analysis


```python
# Package importing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

```

## Step 1: Data Preparation and Exploration

In our previous post, we cleaned and prepared a data set compiled by Pedro Cesar on Kaggle, which scrapped data from https://lol.fandom.com/ to get historical match data, historical player data, and historical champion data from the 2011 to 2022 Worlds Championships. We filled in missing data for **`gold share`**, **`kill share`**, and **`kill participation`** variables using dataframe functions.

Afterward, we explored the dataset using histograms, box plots, pairwise plots, scatterplot matrices, and correlation heatmaps to examine each variable's distribution, correlation, and potential outliers.

## Data Preprocessing

Our current dataset has 1,283 rows of player data and 19 different variables. We need to process the data and select the correct variables from our data in a way that makes sense to put in our model.

League of Legends is a 5 vs. 5 game, so it would make sense to combine players from the same team and use their average total kills combined and the win rate as a team instead of individual players. Statistics for the play-in stage were recorded starting in season 7, but we are only interested in the Main event matches for this project.

Another issue in the dataset is when a player gets substituted during the tournament. The win rate will differ for the substituted player compared to the other team members playing earlier. We can take the most common for each team and use that as the win rate.

Our goal is to create a subset of data that includes only Main event matches, team name, combined win rate, and total number of team kills.


```python
# Subset the columns we are interested in
lr_subset = df2[['event','season','team', 'win_rate', 'kills']]

# Filtering out the "Play in" Tournament results
lr_subset = lr_subset[lr_subset['event']!="Play in"]
```


```python
# summarizing team kills per season
lr_team_kills = lr_subset.groupby(['season','team'])['kills'].sum().rename("team_kills").reset_index()

# getting win_rate for team using most common win_rate
lr_win_rate = lr_subset.groupby(['season', 'team'])['win_rate'].agg(lambda x: x.mode().iloc[0]).reset_index()

# Combining into 1 table for analysis
lr_subset2 = lr_win_rate.merge(lr_team_kills,how='outer',on=['season','team'])
```


```python
lr_subset2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>team</th>
      <th>win_rate</th>
      <th>team_kills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Against_All_authority</td>
      <td>58.3</td>
      <td>13.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Counter_Logic_Gaming</td>
      <td>50.0</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Epik_Gamer</td>
      <td>42.9</td>
      <td>17.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Fnatic</td>
      <td>72.7</td>
      <td>21.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Pacific_eSports</td>
      <td>0.0</td>
      <td>7.00</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have a subset of data that includes the win rate and the number of team kills for each team, we can start building our linear regression model using Scikit-learn.

## Part 3: Linear Regression Modeling

The general steps for model development and evaluation follow these steps:

* Splitting our data into training and testing sets
* Creating the model
* Fit the model to our dataset
* Make predictions using the testing set
* Evaluate the model's performance

* The details in each step will vary depending on which specific algorithm you are using.

### Splitting our data
When analyzing data, ensuring the results can be trusted is essential. To do this, we divide the data into two parts: training data and testing data. The training data teaches the computer to recognize patterns and make predictions based on the data. The testing data is then used to see how well the computer can predict new data it has yet to see.

Think of it like studying for an exam. You study the material (training data) to learn how to answer questions correctly. Then, when you take the exam (testing data), you use what you've learned to answer new questions. By using training and testing data, we can ensure that our computer models are accurate and reliable.

We will use 80% of our data for our training set and the remaining 20% to test.


```python
# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Splitting our data X is our independent variable y is our dependent
X = lr_subset2[['team_kills']]
y = lr_subset2[['win_rate']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
```

### Creating our model

After successfully splitting our data, we can create our model using Scikit-learn. The Scikit-learn (sklearn) library has a variety of machine-learning algorithms built on top of other libraries, including NumPy, SciPy, and Matplotlib. We will be using more from Sklearn in future articles as it equips us with the perfect tools to handle classification, regression, clustering, and more.


```python
# Creating the linear regression model
lr_reg_model = LinearRegression()
```

### Fitting the model

Now that we have a linear model, we must train it using our dataset. This is called **fitting our model**. This will allow the generic model to recognize patterns and relationships within the data. We want to fit the model using the training data we split off earlier, which contains 80% of the dataset.




```python
# Fitting our data to the model
lr_reg_model.fit(X_train, y_train)
```

### Making Predictions

Finally, we can make predictions using the remaining 20% of the data, which we split into the testing dataset.


```python
y_pred = lr_reg_model.predict(X_test)
```

### Evaluating the model

How can we tell how well our linear regression model performed? We can use statistical methods to evaluate our model to ensure we are on the right track.

#### Mean Squared Error
Mean squared error (MSE) helps us measure the average amount by which our predictions differ from the actual values. If we obtain a small MSE, our predictions are close to our actual values, which is good! But if the MSE is large, our predictions are off, and we might need to improve our model.

MSE is calculated with the formula:

![MSE]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-kills-winrate-MSE.png){:style="display:block; margin-left:auto; margin-right:auto"}

#### Root Mean Squared Error
Root Mean squared error (RMSE) is the square root of MSE and has the advantage of being in the same unit as the dependent variable, making it more interpretable. So that means that the value obtained here will be the win rate percentage.

#### R-squared
R-squared (r2) is another statistical value explaining the variation in our data. This is like asking;

 > How well does our model capture the patterns and trends in the data?

R2 values range from 0 to 1, where a value of 1 means that our model perfectly captures all the ways in our data, and an r2 of 0 means our model doesn't explain any of the patterns at all, meaning it is just guessing randomly. We aim for a high r2 value since that will be better at making more accurate predictions.

Both of these functions are part of the sklearn package.


```python
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error:, {rmse}")
print(f"R-squared: {r2}")

```

    Mean Squared Error: 285.6811420616395
    Root Mean Squared Error:, 16.902104663669537
    R-squared: 0.5654582082959507
    

## Part 4: Interpreting and Visualizing our model

### Statistical interpretations

For this model, we obtained an **MSE of 285.68**, a **RMSE of 16.9**, and an **R-squared value of 0.57**.

We need to find out if an MSE of 285.68 is a good or bad value since this number reflects the data we are using. However, this can be a baseline value for comparing the performance of different models.

An RMSE of 16.9 is interpreted as:

> The weighted average error between the predictions and the actual win rate in this dataset is 16.9%, which is likely a good value since the average team win rate is 42.6% in this dataset.

An R-squared value of 0.565 is interpreted as:

> 56.5% of the variance in win rate is explained by kills.

A good amount of the variance is explained by just using kills, but we would like to see the R-squared value between 0.75 - 1, which would be able to explain a significant amount of the variance.

### Visualization

Visualizing our data can also help explain our data. In linear regression, we want a straight trend line that increases or decreases as we get from fewer to more kills.


```python
## gets the slope and intercept for visualization
slope = lr_reg_model.coef_[0]
intercept = lr_reg_model.intercept_

# Plotting results with regression line
plt.scatter(X_test, y_test, alpha=0.5)
plt.plot(X_test, slope*X_test + intercept, color='red')
plt.xlabel('Number of Kills')
plt.ylabel('Win Rate (%)')
plt.show()
```


![linearregreesionscatterplot]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-kills-winrate-scatterplot)
*Scatterplot of our results created with matplotlib*    

    


The scatterplot shows there is an increasing trend between kills and win rate; however, it is not entirely linear. In total, we plotted 36 data points, so it could be that we need more points of data to determine if the number of kills is linearly related to the win rate.

## Part 5: Assumption checking

Remember that linear regression has a set of assumptions that need to be satisfied to accurately generate the predictions we want. To test this, we will use some diagnostic plots for each assumption. These plots include:

* **Residuals vs. Fitted**: This plot is used to check if the assumption of constant variance is met

* **Normal Q-Q Plot**: We use this plot to check our normality assumption.

* **Scale-Location Plot**: We also use this plot to help verify the constant variance assumption.

* **Leverage Plot**: This last plot will show us the influence of each data point on the model. If a point has high leverage, it could impact our result.

In **R**, we can generate these diagnostic plots with a single function. *plot(model)* however, in Python, we can create a similar function using the `statsmodels` and `seaborn` packages to achieve the same effect.


```python
import statsmodels
import statsmodels.formula.api as smf

res = smf.ols(formula= "win_rate ~ team_kills", data=lr_subset2).fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>win_rate</td>     <th>  R-squared:         </th> <td>   0.510</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.507</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   182.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 29 Jul 2023</td> <th>  Prob (F-statistic):</th> <td>4.91e-29</td>
</tr>
<tr>
  <th>Time:</th>                 <td>00:40:07</td>     <th>  Log-Likelihood:    </th> <td> -736.80</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   178</td>      <th>  AIC:               </th> <td>   1478.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   176</td>      <th>  BIC:               </th> <td>   1484.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>  -11.1996</td> <td>    4.136</td> <td>   -2.708</td> <td> 0.007</td> <td>  -19.362</td> <td>   -3.037</td>
</tr>
<tr>
  <th>team_kills</th> <td>    4.2792</td> <td>    0.316</td> <td>   13.524</td> <td> 0.000</td> <td>    3.655</td> <td>    4.904</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.539</td> <th>  Durbin-Watson:     </th> <td>   1.885</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.170</td> <th>  Jarque-Bera (JB):  </th> <td>   3.309</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.333</td> <th>  Prob(JB):          </th> <td>   0.191</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.049</td> <th>  Cond. No.          </th> <td>    47.5</td>
</tr>
</table>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# This function was borrowed from the statsmodel.org website
#***************************************************************************************/
#    Title: Linear regression diagnostics
#    Author: Prajwal Kafle, Matt Spinelli
#    Date: 2023
#    Code version: 0.15.0
#    Availability: https://www.statsmodels.org/devel/examples/notebooks/generated/linear_regression_diagnostics_plots.html
#
#***************************************************************************************/

import numpy as np
import seaborn as sns
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from typing import Type

style_talk = 'seaborn-talk'    #refer to plt.style.available

class LinearRegDiagnostic():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Authors:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.

        Matt Spinelli (m3spinelli@gmail.com, where 3 = r)
        (1) Fixed incorrect annotation of the top most extreme residuals in
            the Residuals vs Fitted and, especially, the Normal Q-Q plots.
        (2) Changed Residuals vs Leverage plot to match closer the y-axis
            range shown in the equivalent plot in the R package ggfortify.
        (3) Added horizontal line at y=0 in Residuals vs Leverage plot to
            match the plots in R package ggfortify and base R.
        (4) Added option for placing a vertical guideline on the Residuals
            vs Leverage plot using the rule of thumb of h = 2p/n to denote
            high leverage (high_leverage_threshold=True).
        (5) Added two more ways to compute the Cook's Distance (D) threshold:
            * 'baseR': D > 1 and D > 0.5 (default)
            * 'convention': D > 4/n
            * 'dof': D > 4 / (n - k - 1)
        (6) Fixed class name to conform to Pascal casing convention
        (7) Fixed Residuals vs Leverage legend to work with loc='best'
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object
        """

        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)
        self.nresids = len(self.residual_norm)

    def __call__(self, plot_context='seaborn-paper', **kwargs):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(ax=ax[0,0])
            self.qq_plot(ax=ax[0,1])
            self.scale_location_plot(ax=ax[1,0])
            self.leverage_plot(
                ax=ax[1,1],
                high_leverage_threshold = kwargs.get('high_leverage_threshold'),
                cooks_threshold = kwargs.get('cooks_threshold'))
            plt.show()

        return self.vif_table(), fig, ax,

    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        fig = QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for i, x, y in self.__qq_top_resid(QQ.theoretical_quantiles, abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(x, y),
                ha='right',
                color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')

        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None, high_leverage_threshold=False, cooks_threshold='baseR'):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color = 'C3')

        factors = []
        if cooks_threshold == 'baseR' or cooks_threshold is None:
            factors = [1, 0.5]
        elif cooks_threshold == 'convention':
            factors = [4/self.nresids]
        elif cooks_threshold == 'dof':
            factors = [4/ (self.nresids - self.nparams)]
        else:
            raise ValueError("threshold_method must be one of the following: 'convention', 'dof', or 'baseR' (default)")
        for i, factor in enumerate(factors):
            label = "Cook's distance" if i == 0 else None
            xtemp, ytemp = self.__cooks_dist_line(factor)
            ax.plot(xtemp, ytemp, label=label, lw=1.25, ls='--', color='red')
            ax.plot(xtemp, np.negative(ytemp), lw=1.25, ls='--', color='red')

        if high_leverage_threshold:
            high_leverage = 2 * self.nparams / self.nresids
            if max(self.leverage) > high_leverage:
                ax.axvline(high_leverage, label='High leverage', ls='-.', color='purple', lw=1)

        ax.axhline(0, ls='dotted', color='black', lw=1.25)
        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_ylim(min(self.residual_norm)-0.1, max(self.residual_norm)+0.1)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        plt.legend(loc='best')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        return (vif_df
                .sort_values("VIF Factor")
                .round(2))


    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y


    def __qq_top_resid(self, quantiles, top_residual_indices):
        """
        Helper generator function yielding the index and coordinates
        """
        offset = 0
        quant_index = 0
        previous_is_negative = None
        for resid_index in top_residual_indices:
            y = self.residual_norm[resid_index]
            is_negative = y < 0
            if previous_is_negative == None or previous_is_negative == is_negative:
                offset += 1
            else:
                quant_index -= offset
            x = quantiles[quant_index] if is_negative else np.flip(quantiles, 0)[quant_index]
            quant_index += 1
            previous_is_negative = is_negative
            yield resid_index, x, y

```


```python
cls = LinearRegDiagnostic(res)
cls()
```

![LRdiagnosticplots]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-kills-winrate-diagnostic-plots.png){:style="display:block; margin-left:auto; margin-right:auto"}
*Diagnostic plots from the function written by Prajwal Kafle and Matt Spinelli*


### Residual vs. Fitted
Ideally, we want the plot to show no fitted pattern for the Residuals vs. Fitted plot, and our reference line should be horizontal at the zero markers. That is not the case for our graph, so we can assume the relationship between kills and win rate is non-linear. We need to use a non-linear transformation on the predictors like log(x), sqrt(x), or $x^2$ to fix this.

### Normal Q-Q Plot
We want to see most of the points fall along the 45-degree reference line for our Normal Q-Q plot. A few points near the bottom of the plot fall outside the reference line, but most of the points are along the line. To check normality thoroughly, we can plot a kernel density plot (KDE) of the residuals to see the distribution and use the Shapiro-Wilk test for normality from the scipy library.

#### Residual Kernel Density Plot


```python
from scipy import stats

residuals = y_test - y_pred
sns.kdeplot(data=residuals)

```

![KDEresidual]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-kills-winrate-KDE.png){:style="display:block; margin-left:auto; margin-right:auto"}
*Kernel Density Plot of residuals to test for normality*
    


The KDE is slightly left-skewed since the center is a little past 0 on the x-axis, and the shape of the plot is precisely bell-shaped.

#### Shapiro-Wilk test for normality
Before we perform a hypothesis test, we must state our acceptance criteria and null hypothesis.
Ho: The residuals are normally distributed
Ha: The residuals are not normally distributed


```python
stats.shapiro(residuals)
```




    ShapiroResult(statistic=0.9698702096939087, pvalue=0.4219997823238373)



We obtain a test statistic 0.97 and a p-value of 0.42 for the Shapiro-Wilk test. Since the p-value is >0.05, at a confidence level of 0.95, we fail to reject our null hypothesis, meaning our residuals are normally distributed and pass the normality assumption.

### Scale-Location plot

For constant variance, we want to see a scale-location plot where

* That the red line is approximately horizontal.
* The spread around the red line doesn't vary with the fitted values.

Our line could be more straight in our scale-location plot, but the spread about the line looks OK. We can add a non-linear transformation to our data to reduce non-constant variances.

### Residual vs. Leverage Plot

The Residuals vs. leverage plot will help us identify outliers causing some issues in our model. Points 151, 115, and 58 are the top 3 more extreme points, but fortunately, they do not exceed the 3 standard deviations.

To check if these outliers are influential, we can check using Cook's Distance.

> A rule of thumb is that an observation has high influence if Cook’s distance exceeds 4/(n - p - 1)(P. Bruce and Bruce 2017)

where n is the number of observations and p the number of predictor variables.


```python
#suppress scientific notation
np.set_printoptions(suppress=True)

#create instance of influence
influence = res.get_influence()

#obtain Cook's distance for each observation
cooks = influence.cooks_distance
```


```python
plt.scatter(lr_subset2.team_kills, cooks[0])
plt.xlabel('x')
plt.ylabel('Cooks Distance')
plt.show()
```



![CooksDistanceLR]({{site.baseurl}}/images/2023-07-29-League-of-legends-linear-regression-kills-winrate-cooks-distance.png){:style="display:block; margin-left:auto; margin-right:auto"}
*Visualization of Cook's Distance to determine influential points*


With 178 data points and 1 predictor variable, Cook's distance data limit is 0.22, and from our graph, none of the points exceed that maximum. This means we have no influential issues and do not need to remove them from our dataset.

### Assumption checking summary

Assumption checking is critical when evaluating regression models to ensure we provide the best model for our data.
To recap:
1. Our model **does not satisfy** the linearity assumption because our residual vs. fitted graph shows a non-linear trend line.
2. Our model **satisfies** the normality assumption using the Normal Q-Q plot, Residual kernel density plot, and the Shapiro-Wilk test.
3. We also **satisfy** the constant variance assumption using the scale-location plot.
4. The 3 outliers in our data do not exceed 3 standard deviations and are also **not influential values** based on Cook's distance values.

## Conclusion

It looks like linear regression did an average job at predicting win rates, but there are other things we can do to improve our model.

To summarize:
1. **Preprocessed data**: We took our previous dataset and created a subset of data for our linear regression analysis. We dealt with issues surrounding player substitutions and varying win rates, decided to focus on only main-stage events, and used a team kills metric due to the 5 vs. 5 nature of the game.
2. **Linear regression model**: We created a linear regression model using sklearn. We split our data into training and testing sets, built and fit our model using the training dataset, predicted the win rate using the testing dataset, and evaluated our model using Mean Squared, Root Mean Squared, and R-Squared statistical methods. We also visualized our data using a scatterplot and trendline.
3. **Assumption checking**: We utilized diagnostic plots to see if our model satisfies the assumptions of linear regression and determined that we have an issue with the linearity assumption from the Residual vs. Fitted plot.

For the next part of our analysis, we will continue with supervised learning to see if we can develop a more accurate model and discover more essential factors to predict win rate
