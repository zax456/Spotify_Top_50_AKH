# Imports/Data reading


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
plt.rcParams["patch.force_edgecolor"] = True
```


```python
top50_df = pd.read_csv("top50.csv", encoding='ISO-8859-1', index_col=0)
top50_df.head()
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
      <th>Track.Name</th>
      <th>Artist.Name</th>
      <th>Genre</th>
      <th>Beats.Per.Minute</th>
      <th>Energy</th>
      <th>Danceability</th>
      <th>Loudness..dB..</th>
      <th>Liveness</th>
      <th>Valence.</th>
      <th>Length.</th>
      <th>Acousticness..</th>
      <th>Speechiness.</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Señorita</td>
      <td>Shawn Mendes</td>
      <td>canadian pop</td>
      <td>117</td>
      <td>55</td>
      <td>76</td>
      <td>-6</td>
      <td>8</td>
      <td>75</td>
      <td>191</td>
      <td>4</td>
      <td>3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>Anuel AA</td>
      <td>reggaeton flow</td>
      <td>105</td>
      <td>81</td>
      <td>79</td>
      <td>-4</td>
      <td>8</td>
      <td>61</td>
      <td>302</td>
      <td>8</td>
      <td>9</td>
      <td>92</td>
    </tr>
    <tr>
      <th>3</th>
      <td>boyfriend (with Social House)</td>
      <td>Ariana Grande</td>
      <td>dance pop</td>
      <td>190</td>
      <td>80</td>
      <td>40</td>
      <td>-4</td>
      <td>16</td>
      <td>70</td>
      <td>186</td>
      <td>12</td>
      <td>46</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beautiful People (feat. Khalid)</td>
      <td>Ed Sheeran</td>
      <td>pop</td>
      <td>93</td>
      <td>65</td>
      <td>64</td>
      <td>-8</td>
      <td>8</td>
      <td>55</td>
      <td>198</td>
      <td>12</td>
      <td>19</td>
      <td>86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Goodbyes (Feat. Young Thug)</td>
      <td>Post Malone</td>
      <td>dfw rap</td>
      <td>150</td>
      <td>65</td>
      <td>58</td>
      <td>-4</td>
      <td>11</td>
      <td>18</td>
      <td>175</td>
      <td>45</td>
      <td>7</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>




```python
# info on null fields in data
top50_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50 entries, 1 to 50
    Data columns (total 13 columns):
    Track.Name          50 non-null object
    Artist.Name         50 non-null object
    Genre               50 non-null object
    Beats.Per.Minute    50 non-null int64
    Energy              50 non-null int64
    Danceability        50 non-null int64
    Loudness..dB..      50 non-null int64
    Liveness            50 non-null int64
    Valence.            50 non-null int64
    Length.             50 non-null int64
    Acousticness..      50 non-null int64
    Speechiness.        50 non-null int64
    Popularity          50 non-null int64
    dtypes: int64(10), object(3)
    memory usage: 5.5+ KB
    


```python
cat_cols = ['Track.Name', 'Artist.Name', 'Genre']
int_cols = [name for name in top50_df.columns if top50_df[name].dtype in ['int64']]
```

### Data cleaning


```python
# standardise all int64 columns to same scale
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
transformed = pd.DataFrame(scaler.fit_transform(top50_df[int_cols]), columns=int_cols, index=top50_df.index)
```


```python
# join back with categorical columns
top50_scaled = top50_df[cat_cols].join(transformed)
```


```python
top50_scaled.head()
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
      <th>Track.Name</th>
      <th>Artist.Name</th>
      <th>Genre</th>
      <th>Beats.Per.Minute</th>
      <th>Energy</th>
      <th>Danceability</th>
      <th>Loudness..dB..</th>
      <th>Liveness</th>
      <th>Valence.</th>
      <th>Length.</th>
      <th>Acousticness..</th>
      <th>Speechiness.</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Señorita</td>
      <td>Shawn Mendes</td>
      <td>canadian pop</td>
      <td>0.304762</td>
      <td>0.410714</td>
      <td>0.770492</td>
      <td>0.555556</td>
      <td>0.056604</td>
      <td>0.764706</td>
      <td>0.391753</td>
      <td>0.040541</td>
      <td>0.000000</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>Anuel AA</td>
      <td>reggaeton flow</td>
      <td>0.190476</td>
      <td>0.875000</td>
      <td>0.819672</td>
      <td>0.777778</td>
      <td>0.056604</td>
      <td>0.600000</td>
      <td>0.963918</td>
      <td>0.094595</td>
      <td>0.139535</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>boyfriend (with Social House)</td>
      <td>Ariana Grande</td>
      <td>dance pop</td>
      <td>1.000000</td>
      <td>0.857143</td>
      <td>0.180328</td>
      <td>0.777778</td>
      <td>0.207547</td>
      <td>0.705882</td>
      <td>0.365979</td>
      <td>0.148649</td>
      <td>1.000000</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beautiful People (feat. Khalid)</td>
      <td>Ed Sheeran</td>
      <td>pop</td>
      <td>0.076190</td>
      <td>0.589286</td>
      <td>0.573770</td>
      <td>0.333333</td>
      <td>0.056604</td>
      <td>0.529412</td>
      <td>0.427835</td>
      <td>0.148649</td>
      <td>0.372093</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Goodbyes (Feat. Young Thug)</td>
      <td>Post Malone</td>
      <td>dfw rap</td>
      <td>0.619048</td>
      <td>0.589286</td>
      <td>0.475410</td>
      <td>0.777778</td>
      <td>0.113208</td>
      <td>0.094118</td>
      <td>0.309278</td>
      <td>0.594595</td>
      <td>0.093023</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>



## Descriptive info


```python
top50_scaled[cat_cols].describe()
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
      <th>Track.Name</th>
      <th>Artist.Name</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>50</td>
      <td>38</td>
      <td>21</td>
    </tr>
    <tr>
      <th>top</th>
      <td>China</td>
      <td>Ed Sheeran</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



<div class='alert alert-info'>
    
In 2019, `Ed Sheeran` was the most popular artist with 4 of his songs being in the top 50. While the most popular genre turns out to be `dance pop`.

</div>


```python
top50_scaled.describe()
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
      <th>Beats.Per.Minute</th>
      <th>Energy</th>
      <th>Danceability</th>
      <th>Loudness..dB..</th>
      <th>Liveness</th>
      <th>Valence.</th>
      <th>Length.</th>
      <th>Acousticness..</th>
      <th>Speechiness.</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.333905</td>
      <td>0.572500</td>
      <td>0.694754</td>
      <td>0.593333</td>
      <td>0.182264</td>
      <td>0.524706</td>
      <td>0.443093</td>
      <td>0.285946</td>
      <td>0.220465</td>
      <td>0.70000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.294270</td>
      <td>0.254141</td>
      <td>0.195572</td>
      <td>0.228494</td>
      <td>0.209779</td>
      <td>0.262777</td>
      <td>0.201773</td>
      <td>0.256697</td>
      <td>0.259572</td>
      <td>0.17966</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.104762</td>
      <td>0.415179</td>
      <td>0.622951</td>
      <td>0.472222</td>
      <td>0.056604</td>
      <td>0.332353</td>
      <td>0.318299</td>
      <td>0.097973</td>
      <td>0.046512</td>
      <td>0.64000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.185714</td>
      <td>0.616071</td>
      <td>0.729508</td>
      <td>0.555556</td>
      <td>0.113208</td>
      <td>0.535294</td>
      <td>0.427835</td>
      <td>0.189189</td>
      <td>0.093023</td>
      <td>0.72000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.500000</td>
      <td>0.763393</td>
      <td>0.831967</td>
      <td>0.777778</td>
      <td>0.202830</td>
      <td>0.700000</td>
      <td>0.528351</td>
      <td>0.442568</td>
      <td>0.279070</td>
      <td>0.83000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sns.distplot(a=top50_scaled['Energy'], kde=True, bins=10)
```


```python
# sns.distplot(a=top50_scaled['Valence.'], kde=True, bins=10)
```

<div class='alert alert-info'>

In general, the top 50 songs have high `Energy`, `Danceability`, `Loudness`. They have a neutral vibe since `Valence` hovers at 0.5 range. The length of each song is about 3 minutes plus. Surprisingly, majority of the popular songs do not have high `Beats Per Minute`.
<br><br>
    
However, they do not have much `Acousticness` with a skewed mean from the max value. They tend not to have much words inside them as well (from the low mean of `Speechiness`).
</div>

# Relationship between most popular songs


```python
# sort by popularity
top50_sorted = top50_scaled.sort_values('Popularity', ascending=False)

# Top 10 songs 
top50_sorted.head(10)
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
      <th>Track.Name</th>
      <th>Artist.Name</th>
      <th>Genre</th>
      <th>Beats.Per.Minute</th>
      <th>Energy</th>
      <th>Danceability</th>
      <th>Loudness..dB..</th>
      <th>Liveness</th>
      <th>Valence.</th>
      <th>Length.</th>
      <th>Acousticness..</th>
      <th>Speechiness.</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>bad guy</td>
      <td>Billie Eilish</td>
      <td>electropop</td>
      <td>0.476190</td>
      <td>0.196429</td>
      <td>0.672131</td>
      <td>0.000000</td>
      <td>0.094340</td>
      <td>0.541176</td>
      <td>0.407216</td>
      <td>0.432432</td>
      <td>0.813953</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Goodbyes (Feat. Young Thug)</td>
      <td>Post Malone</td>
      <td>dfw rap</td>
      <td>0.619048</td>
      <td>0.589286</td>
      <td>0.475410</td>
      <td>0.777778</td>
      <td>0.113208</td>
      <td>0.094118</td>
      <td>0.309278</td>
      <td>0.594595</td>
      <td>0.093023</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Callaita</td>
      <td>Bad Bunny</td>
      <td>reggaeton</td>
      <td>0.866667</td>
      <td>0.535714</td>
      <td>0.524590</td>
      <td>0.666667</td>
      <td>0.358491</td>
      <td>0.164706</td>
      <td>0.701031</td>
      <td>0.797297</td>
      <td>0.651163</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Money In The Grave (Drake ft. Rick Ross)</td>
      <td>Drake</td>
      <td>canadian hip hop</td>
      <td>0.152381</td>
      <td>0.321429</td>
      <td>0.885246</td>
      <td>0.777778</td>
      <td>0.132075</td>
      <td>0.000000</td>
      <td>0.463918</td>
      <td>0.121622</td>
      <td>0.046512</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>Anuel AA</td>
      <td>reggaeton flow</td>
      <td>0.190476</td>
      <td>0.875000</td>
      <td>0.819672</td>
      <td>0.777778</td>
      <td>0.056604</td>
      <td>0.600000</td>
      <td>0.963918</td>
      <td>0.094595</td>
      <td>0.139535</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ransom</td>
      <td>Lil Tecca</td>
      <td>trap music</td>
      <td>0.904762</td>
      <td>0.571429</td>
      <td>0.754098</td>
      <td>0.555556</td>
      <td>0.037736</td>
      <td>0.152941</td>
      <td>0.082474</td>
      <td>0.013514</td>
      <td>0.604651</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Otro Trago</td>
      <td>Sech</td>
      <td>panamanian pop</td>
      <td>0.866667</td>
      <td>0.678571</td>
      <td>0.754098</td>
      <td>0.666667</td>
      <td>0.113208</td>
      <td>0.611765</td>
      <td>0.572165</td>
      <td>0.175676</td>
      <td>0.720930</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Panini</td>
      <td>Lil Nas X</td>
      <td>country rap</td>
      <td>0.657143</td>
      <td>0.482143</td>
      <td>0.672131</td>
      <td>0.555556</td>
      <td>0.132075</td>
      <td>0.447059</td>
      <td>0.000000</td>
      <td>0.445946</td>
      <td>0.116279</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Piece Of Your Heart</td>
      <td>MEDUZA</td>
      <td>pop house</td>
      <td>0.371429</td>
      <td>0.750000</td>
      <td>0.639344</td>
      <td>0.444444</td>
      <td>0.037736</td>
      <td>0.623529</td>
      <td>0.195876</td>
      <td>0.040541</td>
      <td>0.000000</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Truth Hurts</td>
      <td>Lizzo</td>
      <td>escape room</td>
      <td>0.695238</td>
      <td>0.535714</td>
      <td>0.704918</td>
      <td>0.888889</td>
      <td>0.132075</td>
      <td>0.364706</td>
      <td>0.298969</td>
      <td>0.135135</td>
      <td>0.186047</td>
      <td>0.84</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical feature relationships (Top 50)


```python
# counts of each genres in Top 50 with more than 1 song
top50_scaled['Genre'].value_counts()[top50_scaled['Genre'].value_counts()>1]
```




    dance pop           8
    pop                 7
    latin               5
    edm                 3
    canadian hip hop    3
    electropop          2
    dfw rap             2
    panamanian pop      2
    brostep             2
    canadian pop        2
    country rap         2
    reggaeton flow      2
    reggaeton           2
    Name: Genre, dtype: int64



<div class='alert alert-info'>

`dance pop`, `pop` and `latin` are some of the most popular genres in the Top 50 songs.
</div>


```python
# plt.figure(figsize=(10,5))
# top_genres = top50_scaled[top50_scaled['Genre'].isin(top50_scaled['Genre'].value_counts()[top50_scaled['Genre'].value_counts()>1].index)]
# sns.countplot(top_genres['Genre'], color='lightblue')
# plt.title("Songs per genre in Top 50")
# plt.xticks(rotation=90)
```


```python
# counts of each artist in Top 50 with more than 1 song
top50_scaled['Artist.Name'].value_counts()[top50_scaled['Artist.Name'].value_counts()>1]
```




    Ed Sheeran          4
    Lil Nas X           2
    Sech                2
    J Balvin            2
    Marshmello          2
    Post Malone         2
    Ariana Grande       2
    The Chainsmokers    2
    Billie Eilish       2
    Shawn Mendes        2
    Name: Artist.Name, dtype: int64



<div class='alert alert-info'>

`Ed Sheeran` is the most popular artist with the most songs in the Top 50. While the remaining artists have either 1 or 2 songs in this list. 
</div>


```python
# count of songs per artist
# plt.figure(figsize=(10,5))
# top_artists = top50_scaled[top50_scaled['Artist.Name'].isin(top50_scaled['Artist.Name'].value_counts()[top50_scaled['Artist.Name'].value_counts()>1].index)]
# sns.countplot(top_artists['Artist.Name'], color='lightgreen')
# plt.title("Songs per artist in Top 50")
# plt.xticks(rotation=90)
```

### Numeric feature relationships (Top 50)


```python
# Distributions and relationship between features (pairwise)
g2 = sns.PairGrid(top50_scaled[int_cols])
g2.map_offdiag(sns.regplot, ci=None)
g2.map_diag(sns.distplot, bins=10)

for axes in g2.axes.flat:
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
```


![png](output_26_0.png)


<div class='alert alert-info'>

From the pair grid, we can see the relationships between features pairwise and their distributions. We see that `Energy`, `Loudness`, `Danceability`, `Valence`, `Length` have a relatively normal distribution. While `Liveness`, `Acousticness`, `Speechiness` are right skewed. Although `Popularity` seems to be normally distributed, it is slightly left skewed, with more songs having a popularity of between 0.7 ~ 0.9 range. This is expected to even out to a normal distribution if we expand the top songs to a larger value.
</div>


```python
# correlations heatmap
correlations2 = top50_scaled[int_cols].corr()
plt.figure(figsize=(14,7))
sns.heatmap(data=correlations2, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e558d59808>




![png](output_28_1.png)


<div class='alert alert-info'>

There are a few features that have a positive correlation. These includes `Loudness`, `Energy`, `Speechiness`, `Beats Per Minute`, `Valence`. 

However these features do not have a noticable correlation with popularity of a song. Features that have a weak positive correlation includes `Beats Per Minute` and `Speechiness`. Feature that have a weak negative correlation includes `Valence`. 
</div>

### Categorical feature relationships (Top 10)


```python
# top 10 songs
top10_songs = top50_sorted.iloc[:10,]
```


```python
# get top 10 artist songs
top10_art = top10_songs['Artist.Name'].unique().tolist()

# get top 10 genre songs
top10_gen = top10_songs['Genre'].unique().tolist()
```


```python
print("Artists that appeared in Top 10 songs:")
for idx, art in enumerate(top10_art):
    print("{}. {}".format(idx+1, art))
    
print()

print("Genres that appeared in Top 10 songs:")
for idx, gen in enumerate(top10_gen):
    print("{}. {}".format(idx+1, gen))
```

    Artists that appeared in Top 10 songs:
    1. Billie Eilish
    2. Post Malone
    3. Bad Bunny
    4. Drake
    5. Anuel AA
    6. Lil Tecca
    7. Sech
    8. Lil Nas X
    9. MEDUZA
    10. Lizzo
    
    Genres that appeared in Top 10 songs:
    1. electropop
    2. dfw rap
    3. reggaeton
    4. canadian hip hop
    5. reggaeton flow
    6. trap music
    7. panamanian pop
    8. country rap
    9. pop house
    10. escape room
    

<div class='alert alert-info'>

Surprisingly, the most popular artist (`Ed Sheeran`) is not in the top 10 songs. This is the same with Genres. 
</div>

### Numeric feature relationships (Top 10)


```python
# Distributions and relationship between features (pairwise)
# from pandas.plotting import scatter_matrix

# scatter_matrix(top10_songs[int_cols], hist_kwds={'bins':10})
# plt.gcf().set_size_inches(30, 30)
# plt.show()
```


```python
# Distributions and relationship between features (pairwise)
g = sns.PairGrid(top10_songs[int_cols])
g.map_offdiag(sns.regplot, ci=None)
g.map_diag(sns.distplot, bins=10)

for axes in g.axes.flat:
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
```


![png](output_37_0.png)


<div class='alert alert-info'>

At a glance, there seems to be no obvious relationship between any of the features. However, we can see that `Energy`, `Danceability`, `Length` has a normal distribution for the top 10 songs. `Popularity`, `Acousticness`, `Liveness` are right skewed. While `Loudness` is slightly left skewed.
</div>


```python
# correlations heatmap
correlations = top10_songs[int_cols].corr()
plt.figure(figsize=(14,7))
sns.heatmap(data=correlations, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c6b3ac388>




![png](output_39_1.png)


<div class='alert alert-info'>

Using pandas corr() function, we see most features pairwise has a weak (positive/negative) correlation. `Acousticness` and `Danceability` have a strong negative correlation while `Acousticness` and `Liveness` have a strong positive correlation. 
<br><br>
We also see that a few features are correlated with popularity. Examples are `Energy`, `Danceability`, `Loudness`, `Acousticness` and `Speechiness`. However these features have a moderate correlation with popularity. The rest are either having a weak correlation or no linear correlation at all.
</div>

## Sentiment Analysis on Top 50 song titles

In this section, we will see some of the more prominent words used in titles of the top 50 popular songs. 


```python
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import string
```


```python
# stopwords 
stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))

# punctuations
punctuations = list(string.punctuation)
```


```python
## tokenising
titles = top50_scaled['Track.Name'].map(TextBlob)

# print first 5 tokenised titles
for i in range(5):
    print(titles.iloc[i].words)
```

    ['Señorita']
    ['China']
    ['boyfriend', 'with', 'Social', 'House']
    ['Beautiful', 'People', 'feat', 'Khalid']
    ['Goodbyes', 'Feat', 'Young', 'Thug']
    


```python
sentiments = {}
for i in range(titles.shape[0]):
    sentiments[top50_scaled['Track.Name'].iloc[i]] = titles.iloc[i].sentiment.polarity
    
sentiments = pd.DataFrame(sentiments.values(), index=top50_scaled['Track.Name'], columns=['sentiment'])
sentiments.reset_index(drop=False, inplace=True)
sentiments.head()
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
      <th>Track.Name</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Señorita</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>China</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>boyfriend (with Social House)</td>
      <td>0.033333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beautiful People (feat. Khalid)</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Goodbyes (Feat. Young Thug)</td>
      <td>0.100000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments.describe()
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
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.001659</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.235479</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.700000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.850000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(sentiments['sentiment'], kde=True, bins=6)
plt.title("Distribution of sentiment of top 50 song titles")
```




    Text(0.5, 1.0, 'Distribution of sentiment of top 50 song titles')




![png](output_47_1.png)


<div class='alert alert-info'>

While titles usually express some kind of sentiment about the song, we can see that most songs in the top 50 are neutral in their titles. This could be because most of the emotions are expressed through song lyrics instead. While titles are only an indication of what is to be expected from the song.
</div>


```python
title_str = top50_scaled['Track.Name'].map(nltk.word_tokenize)

# text cleaning - lower caps, stopwords, punctuations
for i in range(title_str.shape[0]):
    title_str.iloc[i] = [w.lower() for w in title_str.iloc[i]]
    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in punctuations]
    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in stop_words_en]
    title_str.iloc[i] = [w for w in title_str.iloc[i] if w not in stop_words_es]

# forms long paragraph of string for wordcloud
long_titles = ""
for i in range(title_str.shape[0]):
    temp = " ".join(title_str.iloc[i])
    long_titles = long_titles + " " + temp
    
# remove leading and trailing whitespaces
long_titles = long_titles.strip()
long_titles = long_titles.replace('feat', '').replace('ft.', '')
print(long_titles)
```

    señorita china boyfriend social house beautiful people  khalid goodbyes  young thug n't care justin bieber ransom sleep old town road remix bad guy callaita loco contigo  j. balvin tyga someone loved trago remix money grave drake  rick ross guidance  drake canción sunflower spider-man spider-verse lalala truth hurts piece heart panini conoce remix soltera remix bad guy justin bieber ca n't dance monkey 's calma pretendes takeaway 7 rings 0.958333333333333 london  j. cole travis scott never really summer days  macklemore patrick stump fall boy trago antisocial travis scott sucker fuck 'm lonely anne-marie 13 reasons season 3 higher love need calm shallow talk altura one thing right robaré happier call mine cross  chance rapper pnb rock
    


```python
plt.figure(figsize=(12,8))
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1000,
                      height=1000).generate(long_titles)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](output_50_0.png)


<div class='alert alert-info'>

First, we observe that `remix` was used the most in song titles. This makes sense if the remixed song turns out to be better than the original. <br><br>

Second, hypothetically speaking, songs that we can relate to turns out to be more popular. Songs that deal with emotion such as love contain words such as `boyfriend`, `guy`. Singers such as `justin bieber` tends to make more songs about relationships, which is why they are featured in certain song collaborations. 
</div>

# Conclusion

<div class='alert alert-success'>

Having separated songs into top 50 and top 10, we observed that `Speechiness` has a positve correlation with popularity. While `Beats Per Minute` does not correlate with popularity. While the top 10 songs showed that `Energy`, `Danceability`, `Loudness`, `Acousticness` has correlations with popularity, they are mostly songs relating to relationships such as love. <br><br>

This is further supplemented by the result shown sentiment analysis of song titles. Songs titles that contain words relating to relationships or sung by singers associated with emotional songs tend to be more popular. Furthermore, **remixed** songs are also popular with audiences. 
</div>


```python

```
