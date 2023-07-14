---
title: "Post: Standard"
excerpt_separator: "<!--more-->"
categories:
- Blog
  tags:
- Post Formats
- readability
- standard
- League of Legends
- Data analysis
- Exploratory data analysis
- Statistical analysis
- Data visualization
- Gaming
- Esports
- Win rate
- Player performance
- Data exploration
- Summary statistics
- Outliers
- Correlation analysis
- Data preprocessing
---


# League of Legends: Exploring the Relationship Between Kills and Win Rate, Part 1: Data Exploration

## Introduction to League of Legends

Since its release in 2009 by Riot Games, League of Legends has become a massively popular multiplayer online battle arena (MOBA) game. It features intense team-based gameplay where two teams of five players compete to destroy the opposing team's Nexus, the main structure. Each player takes control of a unique Champion, each with their own distinct abilities and playstyle, adding depth and strategic complexity to the game. But League of Legends is more than just a game—it has been a driving force behind the meteoric rise of competitive esports.

Esports has revolutionized the world of gaming, transforming it into a realm of professional competition with massive prize pools and devoted fan bases. League of Legends has been at the forefront of this esports revolution, attracting millions of viewers worldwide and offering astonishing prize pools for its tournaments. The League of Legends World Championship, for example, boasted a staggering prize pool of $2,225,000 USD in 2022, solidifying the game's position as a cornerstone of the esports industry.

## Goal of the Project

Our project's goal is to discover meaningful relationships that can help us predict match outcomes in League of Legends. Specifically, we aim to answer the question: "What is the connection between a player's average number of kills per game and their corresponding win rate?"

## Packages

This project will utilize the following packages for data analysis, visualization, and model creation:

1. pandas - Data wrangling
2. numpy - Filling in missing data
3. missingno - visulization of missing values
4. seaborn - data visualization
5. plotly - data visualization


```python
# Package importing

import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import plotly.express as px

```

## Data Preparation

The data we will be using for this project is complied by Pedro Cesar on Kaggle and is scrapped from https://lol.fandom.com/ to get historical match data, historical player data and, historical champion data from the 2011 worlds to 2022 worlds. For this project we will be using the historical player data.


```python
# Loading the Kaggle dataset
df = pd.read_csv("\data\players_stats.csv")
```


```python
# checking shape of
df.shape
```




    (1283, 21)



## Variables Dictionary

We have 21 different variables in this dataset which are defined below:

* `season` - the championship season. 1 = 2011 which is the first Worlds
* `event` - Main stage event or play-in
* `team` - team name
* `player` - player name
* `games_played` - number of games played in the tournament season
* `wins` - number of wins throughout the tournament season
* `loses` - number of loses throughout the tournament season
* `win_rate` - how many successful wins throughout all the matches played throughout the tournament
* `kills` - average number of kills per game throughout the tournament
* `deaths` - average number of deaths per game throughout the tournament
* `assists` - average number of assists per game throughout the tournament
* `kill_death_assist_ratio` - KDA kills + assists divided by deaths
* `creep_score` - average number of "minion's" that were killed per game throughout the tournament
* `cs/min` - average creep score per minute per game throughout the tournament
* `gold` - average gold earned per game throughout the tournament
* `gold/min` - average gold earned per minute per game throughout the tournament
* `damage` - average damage done per game throughout the tournament
* `damage/min` - average damage done per minute per game throughout the tournament
* `kill_participation` - average kills + assists / total team kills pre game throughout the tournament
* `kill_share` - average kills / total team kills per game throughout the tournament
* `gold_share` - average gold / total team gold per game throughout the tournament

## Missing Data and Outliers

We can use pandas to get a summary of the missing values using the `.info()`


```python
# Summary of column data types of counts
df.info()
```

    Data columns (total 21 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   season                   1283 non-null   int64  
     1   event                    1283 non-null   object 
     2   team                     1283 non-null   object 
     3   player                   1283 non-null   object 
     4   games_played             1283 non-null   int64  
     5   wins                     1283 non-null   int64  
     6   loses                    1283 non-null   int64  
     7   win_rate                 1283 non-null   float64
     8   kills                    1283 non-null   float64
     9   deaths                   1283 non-null   float64
     10  assists                  1283 non-null   float64
     11  kill_death_assist_ratio  1283 non-null   float64
     12  creep_score              1283 non-null   float64
     13  cs/min                   1283 non-null   float64
     14  gold                     1283 non-null   float64
     15  gold/min                 1283 non-null   int64  
     16  damage                   409 non-null    float64
     17  damage/min               409 non-null    float64
     18  kill_participation       1111 non-null   float64
     19  kill_share               1111 non-null   float64
     20  gold_share               1111 non-null   float64
    dtypes: float64(13), int64(5), object(3)
    memory usage: 210.6+ KB
    

We can get the percentage of missing values using `isnull()` and `sum()`


```python
# Calculating the percentage of missing values
(df.isnull().sum() / df.shape[0]).sort_values(ascending=False).rename('Percentage of missing values') * 100
```
    damage/min                 68.12159
    damage                     68.12159
    gold_share                 13.40608
    kill_share                 13.40608
    kill_participation         13.40608
    kill_death_assist_ratio     0.00000
    gold/min                    0.00000
    gold                        0.00000
    cs/min                      0.00000
    creep_score                 0.00000
    season                      0.00000
    event                       0.00000
    deaths                      0.00000
    kills                       0.00000
    win_rate                    0.00000
    loses                       0.00000
    wins                        0.00000
    games_played                0.00000
    player                      0.00000
    team                        0.00000
    assists                     0.00000
    Name: Percentage of missing values, dtype: float64

In this dataset, the variables `kill_participation`, `kill_share`, and `gold_share` were not calculated until the 2014 4th Worlds tournament. This accounts for a 13.4% data gap in these categories. Fortunately, we can easily compute these parameters using numpy.

However, the variables 'Damage' and 'Damage/Min' present a different situation. These metrics were only tracked starting from the 2020 10th Worlds tournament, resulting in a significant 68.1% data deficiency in our current dataset. We will explore alternative data sources to fill in this missing statistic.

> Just a quick note: This information isn't really important for what we're currently studying, but it will be useful later on when we start looking at other models like multi-linear regression.

## missingno

Another way to visualize the missing values is from the package `missingno`.
Using bar charts and matrix plots we can see the significance of the missing values in our dataset.

```python
# Provides a bar chart of missing values
msno.bar(df)
```

![missingno bar chart for missing values](E:\ModelDiversity\assets\images\output_14_1.png)

The bar graph reiterates how significant the missing values are.

```python
# Positional information on where the missing values are
msno.matrix(df)
```
![missingno bar chart for missing values from where in the dataset](E:\ModelDiversity\assets\images\output_16_1.png)

We can use the matrix in the `missingno` package to visualize the missing values in our dataset. Looking at the matrix, we can see that the variables `kill_participation`, `kill_share`, and `gold_share` are missing data for the initial 3 years of the Worlds tournaments. Similarly, the variables `damage` and `damage/min` have missing values until the most recent editions of the Worlds tournaments. This aligns with our prior understanding of the dataset and confirms the presence of missing data in these specific variables.

## Missing values

Now that we have identified our missing values, we can start tackle them to get a more complete dataset.

### kill_participation, kill_share, gold_share

As mentioned above these variables can be derived with the information we currently have.

**kill_percentage**:
-  This is calculated by taking the average kills and assists for a player and dividing that by the number of total team kills during the tournament.

$$
    \text{kill percentage} = \frac{\text{average kills} + \text{average assist}}{\text{total team kills}}
$$

**kill_share**:
-  This is calculated by taking the average kills for a player and dividing that by the number of total team kills during the tournament.

$$
    \text{kill share} = \frac{\text{average kills}} {\text{total team kills}}
$$

**gold_share**:
-  This is calculated by taking the average gold for a player and dividing that by the number of total team gold during the tournament.

```python
# Subset missing values
missing_values1 = df[df['season']<4]

# calculating average team gold per team each season
team_gold_1_3 = missing_values1.groupby(['season','team'])['gold'].sum().rename("team_gold")

# calculating average team kills per team each season
team_kills_1_3 = missing_values1.groupby(['season','team'])['kills'].sum().rename("team_kills")

# merging team_gold and team_kills back into dataframe
missing_values2 = missing_values1.merge(team_gold_1_3,how='outer',on=['season','team']).merge(team_kills_1_3,how='outer',on=['season','team'])
```

```python
# filling in the gold_share using the calculation above
missing_values2['gold_share'] = round(missing_values2['gold']/missing_values2['team_gold'] * 100,2)

# filling in the kill_share using the calculation above
missing_values2['kill_share'] = round(missing_values2['kills']/missing_values2['team_kills'] * 100,2)

# filling in the kill_participation using the calculation above
missing_values2['kill_participation'] = round((missing_values2['kills']+missing_values2["assists"])/missing_values2['team_kills'] *100,2)

```

```python
# Updates our existing df
df.update(missing_values2)
```

After filling in our missing values we can quickly check to see if they filled in properly.

```python
msno.bar(df)
```
![missingno bar chart for missing values](E:\ModelDiversity\assets\images\output_23_1.png)

Based on the bar graph, we can observe that the only information we are still missing are the damage and damage per minute columns. The game client began reporting damage starting from 2016, which reduces the amount of missing data from 68.1% to approximately 50%.

For our current research question, we can exclude these columns as we only require the win percentage. In future projects, we may explore methods to predict or estimate the missing damage values such as imputation or prediction using regression.

## Outliers

We can use two different methods to determine if there are any outliers within our dataset. First we will drop the two damage columns so that we have a complete dataset. Afterward, we examine each season using a histogram and a boxplot to see if there are any patterns and outliers in our dataset. Detecting outliers in our data is important for the integrity and reliability of our data since they might have a large influence in our analysis. We will need to decide how to deal with outliers (remove or stratify our analysis) so that our conclusions are representative of our data.


```python
# subset your dataset to get no missing values
df2 = df.drop(['damage','damage/min'],axis=1)
```


```python
# generating histograms for each of the 16 remaining variables
df2.hist(figsize=(15,15))
```
    
![histogram for all numerical variables](E:\ModelDiversity\assets\images\output_27_1.png)

## Histograms

From the histograms, we can see that some of the distributions are slightly skewed left and right.

* `season` - When we examine the number of players in each season, we notice that the graph is skewed towards the left side. This means that in earlier seasons, there were more players compared to recent seasons. However, it's important to note that starting from season 7, the tournament introduced a preliminary round called the **play-in** stage, where teams compete for a spot in the main event. This change impacted the number of players we see in the graph.

* `games played` - The histogram for games played shows a slightly skewed distribution towards the right side. This means that most teams played a moderate number of games, with a peak between 6 and 7 games. As the tournament progresses, only one team emerges as the winner, which explains why the number of teams decreases as we move towards higher game counts.

* `wins` - Similar to the games played variable, we see a right skewed distribution of wins. This indicates that most teams have a relatively lower number of wins, and as we move towards higher win counts, the number of teams decreases. This is expected since there can only be one winner in the tournament.

* `loses` - In the losses histogram, you may notice an absence of 4 losses. This is because tournament games are played in a best-of-one (BO1) or best-of-five series (BO5). In a best-of-five series, the first team to win three games out of five is declared the winner. Therefore, there is no opportunity for a team to have exactly 4 losses.

* `win rate` - The histogram for win rate is almost normally distributed with teams having a slightly higher than 50% win rate.

* `kills` - Kills are heavily right skewed since it is rare for a single individual to consistently have a high number of kills in each game. The median number of kills is 2.25 but keep in mind this includes all roles. Players who play **support** roles will often get no kills during the game and have higher assists and deaths compare to the **ADC** or **Middle** roles.

* `deaths` - Deaths are also heavily right skewed were the median number of deaths is about 2 to 3. Compared to other MOBAs, League of Legends games are usually lower scoring in kills and deaths as the momentum swings are more drastic which is a reason why we see a lower number of deaths and kills.

* `assists` - Assists also follow the same right skewed distribution as previous kills and deaths. We the median number of deaths to be around 6.

* `kill death assist ratio` - Kill death assist (KDA) ratio is a metric that can be used to determine a players performance during the game. Since this is a ratio, A higher KDA means that you were able to get a lot of kills and assists while maintaining a lower number of deaths. A low KDA means the opposite, where you have a higher number of deaths and a lower number of kills and assists. Having a low or high KDA doesn't necessary mean that you are a good or bad player. The goal of the game is to take down the enemies main structure (Nexus) and it can take only 1 fight to do so. Also, depending on the role that you play, your KDA can be drastically different. ADC and Middle champions will generally have a higher KDA since their role involves attacking enemy players. Support and top lane tanks usually have a lower KDA since their role is to assist the team rather than get kills and are okay with having more deaths. In this histogram we see that we are heavily right skewed as these are professionals that play at the highest levels and do not make many mistakes since the stakes are high.

* `creep score` - Creep score (CS) is slightly left skewed with a large number of players having <100 CS. Creep or minions are neutral enemies and are the main way of getting gold and experience to level up your champion. In terms of the game, 3 positions focus on killing creep which are Top Middle and, the ADC. The support and jungle champions are more focused on building advantages in lane and positioning around the map.

* `cs/min` - Creep score per minute is has the same left skew distribution as creep score. The average cs/min is 6.0 with 50% of the players having a cs/min higher than the average at 7.1 cs/min

* `gold` - Gold is slightly right skewed. Gold is measured in thousands where the average gold earned per player is 11.5k.

* `gold/min` - Interestingly, gold/min is left skewed compared to gold. However, gold per minute is not measured in thousands like gold.

* `kill participation` - Kill participation is one of the few variables that are normally distributed in this dataset. There are very few with 0% and 100% kill participation. The mean is 65.9% and half the players are also at 66.3%.

* `kill share` - Kill share is heavily left skewed. the difference between kill share and kill participation is removing assists from the calculation. Again, roles such as support or top lane tanks would have very low kill share but could have high kill participation.

* `gold share` - Gold share is right skewed with a mean of 19.9% gold share. The distribution for gold share is simliar to the distributions for gold/min, creep_score and cs/min. This can be furthered investigated with the use of a correlation heatmap. If we wanted to run a multi-linear regression we would have to test for collinearity and address that before making our model.

## Summary statistics

Summary statistics are essential numerical values that provide a concise overview of key characteristics within a dataset. They offer a quick snapshot of the data's central tendencies, dispersion, and distribution. These statistics, including measures like mean, median, standard deviation, and percentiles, play a vital role in data analysis. They allow us to grasp important insights promptly and make informed decisions. By examining summary statistics, we can identify outliers, assess data quality, compare different groups, and track changes over time. These concise yet informative statistics serve as a foundation for further analysis and assist in guiding decision-making. They empower researchers and analysts to draw meaningful conclusions and uncover valuable patterns in the data with clarity and precision.



```python
# Describes our dataframe except for median
df2.describe()
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
      <th>games_played</th>
      <th>wins</th>
      <th>loses</th>
      <th>win_rate</th>
      <th>kills</th>
      <th>deaths</th>
      <th>assists</th>
      <th>kill_death_assist_ratio</th>
      <th>creep_score</th>
      <th>cs/min</th>
      <th>gold</th>
      <th>gold/min</th>
      <th>kill_participation</th>
      <th>kill_share</th>
      <th>gold_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
      <td>1283.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.596259</td>
      <td>8.339829</td>
      <td>4.169914</td>
      <td>4.169914</td>
      <td>43.983632</td>
      <td>2.422377</td>
      <td>2.678675</td>
      <td>5.573624</td>
      <td>3.625511</td>
      <td>203.790436</td>
      <td>6.048987</td>
      <td>11.459938</td>
      <td>339.505066</td>
      <td>65.911715</td>
      <td>20.075284</td>
      <td>19.942471</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.121670</td>
      <td>4.126872</td>
      <td>3.569781</td>
      <td>1.735865</td>
      <td>25.033173</td>
      <td>1.446430</td>
      <td>0.998543</td>
      <td>2.242979</td>
      <td>2.577616</td>
      <td>105.616603</td>
      <td>3.054957</td>
      <td>2.633076</td>
      <td>69.859524</td>
      <td>9.607751</td>
      <td>10.733704</td>
      <td>3.595423</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.290000</td>
      <td>4.670000</td>
      <td>0.170000</td>
      <td>4.800000</td>
      <td>173.000000</td>
      <td>26.700000</td>
      <td>0.000000</td>
      <td>11.890000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>28.600000</td>
      <td>1.330000</td>
      <td>2.000000</td>
      <td>4.110000</td>
      <td>2.080000</td>
      <td>128.700000</td>
      <td>3.795000</td>
      <td>9.600000</td>
      <td>288.500000</td>
      <td>60.200000</td>
      <td>12.600000</td>
      <td>17.615000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>44.400000</td>
      <td>2.250000</td>
      <td>2.600000</td>
      <td>5.250000</td>
      <td>3.050000</td>
      <td>231.380000</td>
      <td>7.080000</td>
      <td>11.600000</td>
      <td>347.000000</td>
      <td>66.300000</td>
      <td>19.900000</td>
      <td>20.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>60.000000</td>
      <td>3.380000</td>
      <td>3.220000</td>
      <td>6.815000</td>
      <td>4.330000</td>
      <td>284.750000</td>
      <td>8.545000</td>
      <td>13.400000</td>
      <td>393.000000</td>
      <td>72.200000</td>
      <td>27.200000</td>
      <td>22.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.000000</td>
      <td>21.000000</td>
      <td>15.000000</td>
      <td>8.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>7.830000</td>
      <td>17.000000</td>
      <td>25.000000</td>
      <td>477.000000</td>
      <td>11.170000</td>
      <td>21.700000</td>
      <td>519.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>26.900000</td>
    </tr>
  </tbody>
</table>
</div>

After applying the describe() function to the dataset, we observed that the majority of variables appeared within the anticipated ranges, with one exception: KDA. While 75% of players exhibited a KDA of 4.33, an intriguing finding emerged when considering the maximum KDA value of 25. This outlier demands our attention and prompts us to investigate further to ascertain its validity and potential impact on our analysis.

## Boxplots

Using boxplots, we can detect potential outliers within the dataset, which could impact our model's performance. Outliers have been observed in various columns, including kill_share, kill_participation, gold, KDA, assists, deaths, kills, wins, and games played.

```python
## Boxplots for each variable

# select only numerical columns
num_column = df2.select_dtypes(include=['int','float']).columns

# Creating horizontal boxplot
fig = px.box(df2, x=num_column, orientation='h', title='Boxplots for Each player variable')

# Update layout settings
fig.update_layout(
    yaxis_title='Variable',
    xaxis_title='Value',
    showlegend=False,
    height=600,
    width=1000
)

fig.show()
```
![boxplots for each player](E:\ModelDiversity\assets\images\output_30_2.png)

To address these outliers, several approaches can be considered:

1. Exclusion of data: Setting a specific threshold and replacing values above or below this limit. However, it's essential to exercise domain knowledge to determine whether these values hold any particular significance.
2. Data transformation: Applying mathematical transformations such as logarithmic, square root, or power transformations to compress the value ranges.
3. Separate handling: Utilizing other outlier detection techniques like z-scores or more advanced statistical methods.

Not all outliers necessarily indicate problematic data. It's crucial to assess the influence of these points on our model. Cook's Distance and other diagnostic measures can be employed to evaluate the potential impact of removing these outliers from our analysis.

## Scatterplot Matrix

A pairwise plot allows us to visualize the relationship between multiple variables by creating scatterplots for each pair of variables.

```python
# creating a pairwise plot of all 16 numerical variables
sns.pairplot(df2, height=3)
```
![pairwise plot](E:\ModelDiversity\assets\images\output_35_2.png)

Visually, some variables look highly correlated with each other such as creep_score and gold_share. To further investigate these relationships, we can utilize a correlation heatmap, which provides us with correlation coefficients. These coefficients help us understand how different variables might impact each other.

## Correlation Heat Map
Correlation coefficients range from -1 to 1, indicating the strength and direction of the relationship. Variables with values close to the limits of this range exhibit a pronounced positive or negative relationship.

In general, if the coefficient falls between 0 and 0.3, it suggests a weak or negligible relationship. A coefficient between 0.3 and 0.7 indicates a moderate relationship, while a value greater than 0.7 suggests a strong relationship between the variables. These coefficients enable us to gauge the level of association between different factors in the dataset, shedding light on the interconnectedness of various aspects of League of Legends gameplay.

> 0 - 0.3 - Weak or no Relationship
> 0.3 - 0.7 - Moderate Relationship
> 0.7 - Strong relationship


```python
# subsetting only the numerical columns
df2_num = df2.select_dtypes(include=np.number)

# creates a correlation matrix for our data set
lol_cor = df2_num.corr()

# creates a heatmap with labeled correlation coefficient
sns.heatmap(lol_cor, annot=True,annot_kws={'size': 8}, fmt=".1f",linewidth=.5)
```

![correlation coefficient heatmap](E:\ModelDiversity\assets\images\output_38_1.png)

From the heatmap, we can see there aren't too many variables that have high correlation when it comes to examining win rate. Strong correlations associated with win rate are wins with a correlation of 0.8 and KDA with a correlation of 0.7. Moderate relationships include assist with 0.6, games played with 0.5, deaths with -0.5, kills with 0.4, loses with -0.4, and gold/min and gold both at 0.3.

## Conclusion

To recap our exploration of the relationship between kills and win rate in League of Legends:

*  **Dataset and Goal**: We analyzed a League of Legends dataset that covered historical player data from the 2011 to 2022 World Championships. Our goal was to investigate the connection between a player's average number of kills per game and their win rate.

*  **Missing Data and Outliers**: We encountered missing values in variables such as kill_participation, kill_share, gold_share, damage, and damage per minute. We derived missing variables except for damage-related columns due to significant data deficiency. Outliers were observed in various columns, and we discussed potential approaches for handling them.

* **Correlation Analysis**: Through scatterplot matrices and correlation heatmaps, we explored the relationships between variables. Strong positive correlations with win rate were observed for variables like wins and KDA, while variables such as assists, games played, deaths, kills, loses, gold/min, and gold showed moderate correlations.

* **Linear Relationship**: Although we did not find a strong linear relationship between kills and win rate in this initial exploration, we identified other variables that may play a more substantial role in predicting match outcomes.

In the next part of our analysis, we will delve deeper into statistical modeling techniques, particularly linear regression, to build predictive models and uncover the key factors that contribute to success in League of Legends matches.

Stay tuned for Part 2, where we will further investigate the predictive power of different variables and analyze their impact on win rates in more detail.
