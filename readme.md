# Football player analysis

Identification of player clusters in the [2022-2023 European Leagues Player Stats](https://www.kaggle.com/datasets/vivovinco/20222023-football-player-stats/data) dataset and intent to find the best possible combination of players using a genetic algorithm.

## Rough description of the dataset

It contains +2500 rows and 124 columns. Each entry corresponds to a player. The first few columns are qualitative variables describing the player (its name, age, club, etc.). The others are "in-game" statistics (number of chances created, goals scored, assists, etc.).

## 1st task : players clustering

The aim was to form clusters of players with similar statistics. Due to the large number of variables, we performed a PCA and then a K-means clustering.

## 2nd task : find the best team possible

The goal was to find the best possible team based on a series of criteria such as number of goals scored, number of passes completed, etc. while minimizing its market value.

### Step 1 : players value prediction

As we didn't know the value of the players, we trained a model (RandomForest) to predict the value of a player based on his characteristics using another dataset.

### Step 2 : genetic algorithm

Once we had the players' values, we used a genetic algorithm to obtain an optimal team according to our objective function.

## Drawbacks of our approach

<!-- ### Clustering

- "bad" players are hard to classify. For instance, a striker who doesn't score goals might belong to the same cluster as defenders since the number of goals plays a big role in the clustering.

### Value prediction + genetic algorithm -->

- poor score the value prediction using only variables common to both datasets
