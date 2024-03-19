
"""

Importing requisite libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Concatenate
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)

df= pd.read_csv('universal_top_spotify_songs.csv') # read csv file from colab input directory
df.head(5)

df[df.duplicated()].sum() # search for duplicates

df.isna().sum() # check for nul values

df.info()

df[df['artists'].isnull()] #check for null values in artists

df.dropna(subset=['artists'], inplace=True) # drop those record with null values

def explicit(i):
    return(
    df.loc[:,'is_explicit']
    .replace(False, 0)
    .replace(True, 1)
    )
df = df.assign(is_explicit=explicit)
df.head()

df2 = df['artists'].str.split(', ', expand=True) # split the artist based on comma as delimiter
df2.nunique()

df2.info()

df2.head(100)

df = pd.concat([df2, df], axis=1) # concat the two datframes

df.rename(columns = {0: 'main_artist', 1:'feat_1',2 : 'feat_2'} , inplace =True) # rename the column names

df.head(5)

df.drop(df.iloc[:, 3:26 ], axis=1, inplace=True) # droping the additional columns which contains None values
df

df.drop(['artists'], axis=1, inplace= True) # dropping the previous artist column as we have added new columns

df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
df['album_release_date'] = pd.to_datetime(df['album_release_date'])

df['release_year'] = df['album_release_date'].dt.year
df['days_since_album_release'] = (df['snapshot_date'] - df['album_release_date']).dt.days

df['duration_min']= round(df['duration_ms'] / (1000*60), 2)

def key_full(key):
    return(
    df.loc[:,'key']
    .replace(0, 'C')
    .replace(1, 'C#')
    .replace(2, 'D')
    .replace(3, 'Eb')
    .replace(4, 'E')
    .replace(5, 'F')
    .replace(6, 'F#')
    .replace(7, 'G')
    .replace(8, 'G#')
    .replace(9, 'A')
    .replace(10, 'Bb')
    .replace(11, 'B')
    )
df = df.assign(key = key_full)

pip install pycountry

import pycountry as pc

# replace country code with actual names using pycountry
def country_name(dataframe, col):


    for index, code in enumerate(dataframe[col]):
        if code:
            try:
                country = pc.countries.get(alpha_2=code)
                df.at[index, col]= country.name
            except AttributeError:
                pass
            #if coutry code is null replace it with global
            except LookupError:
                df.at[index, col] = 'Global'
country_name(df, 'country')

df.sample(5)

df.loc[df['country']=='Global']

df.isna().sum()

df[df['main_artist'].isna()]

df.isna().sum()

df['country'].unique()

df.loc[df['country']=='AE', 'country']='United Arab Emirates'

df['country'].unique()

continents = {
    'North America': ['United States', 'Mexico' , 'Canada'],
    'Central America': ['Costa Rica', 'Dominican Republic', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama', 'El Salvador'],
    'South America': ['Argentinta', 'Bolivia, Plurinational State of', 'Brazil', 'Colombia', 'Chile', 'Ecuador', 'Paraguay',
                      'Peru', 'Uruguay', 'Venezuela, Bolivarian Republic of'],
    'Europe': ['Austria', 'Belgium', 'Bulgaria', 'Belarus', 'Switzerland', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Spain',
              'Finland', 'France', 'United Kingdom', 'Greece', 'Hungary', 'Ireland', 'Iceland', 'Italy', 'Lithuania', 'Luxembourg',
              'Latvia', 'Norway', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Sweden', 'Slovakia', 'Ukraine'],
    'Asia': ['United Arab Emirates', 'Egypt', 'Hong Kong', 'Israel', 'India', 'Indonesia', 'Japan', 'Korea, Republic of', 'Kazakhstan',
             'Malaysia', 'Philippines', 'Pakinstan', 'Saudi Arabia', 'Singapore', 'Thailand', 'Turkey', 'Taiwan, Province of China',
             'Viet Nam'],
    'Africa': ['Morocco', 'Nigeria', 'South Africa'],
    'Oceania': ['Australia', 'New Zealand'],
    'Global': ['Global']
}
sum(len(i) for i in continents.values())

# Creating a function to assing those values to a new column
def assign_continent(country):
    for continent, countries_in_continent in continents.items():
        if country in countries_in_continent:
            return continent

df['continent'] = df['country'].map(assign_continent)

df['continent'].value_counts()

df['tempo'] = round(df['tempo'])

# Creating a column with the past week rank
df['past_week_rank'] = df['daily_rank'] + df['weekly_movement']

df.shape

df.describe().T

df[(df['popularity']>0)].describe().T

df

most_no1s = df[df['daily_rank']==1].groupby(['main_artist'])['daily_rank'].count().sort_values(ascending=False)[:10]
most_no1s

most_no1s = most_no1s.sort_values(ascending=True)

#Artists with Most No.1 Songs

plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
bar_colors = ['#964B00'] * len(most_no1s)

bars = plt.bar(most_no1s.index, most_no1s.values, color=bar_colors)
plt.title('Artists with Most No.1 Songs')
plt.xlabel('Artists')
plt.ylabel('Number of No.1 Songs')
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotating x-axis labels for better visibility
plt.yticks(fontsize=12)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, str(int(bar.get_height())), fontsize=10)
plt.show()


#Most No.1 Songs

most_no1s_songs = df[df['daily_rank']==1].groupby(['name'])['daily_rank'].count().sort_values(ascending=False)[:10]
most_no1s_songs

plt.figure(figsize=(10, 6))

bar_colors = ['#964B00'] * len(most_no1s)

bars = plt.bar(most_no1s_songs.index, most_no1s_songs.values, color=bar_colors)
plt.title(' Most No.1 Songs')
plt.xlabel('Songs')
plt.ylabel('Number of No.1 Songs')
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotating x-axis labels for better visibility
plt.yticks(fontsize=12)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, str(int(bar.get_height())), fontsize=10)
plt.show()


#Popular Continents based on Popularity of Songs
df_con = df.groupby('continent')['popularity'].agg('mean').sort_values(ascending= True).reset_index()

df_con

df_con = df_con[0:7]

plt.figure(figsize=(10, 6))

bar_colors = ['#964B00'] * len(most_no1s)

bars = plt.bar(df_con['continent'], df_con['popularity'], color=bar_colors)
plt.title('Popular Continents based on Popularity of Songs')
plt.xlabel('Continents')
plt.ylabel('Popularity of Songs')
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotating x-axis labels for better visibility
plt.yticks(fontsize=12)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, str(int(bar.get_height())), fontsize=10)
plt.show()


#Most Popular Song by Year

df_yr_pop = df.groupby('release_year')['popularity'].agg('max').reset_index()

df_yr_pop

df[df['release_year'] == 1942]

df_year_song = df.groupby(['release_year', 'name'])['popularity'].mean().reset_index()

most_popular_per_year = df_year_song.loc[df_year_song.groupby('release_year')['popularity'].idxmax()]

most_popular_per_year = most_popular_per_year.tail(10).reset_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(most_popular_per_year['release_year'], most_popular_per_year['popularity'], color='#964B00')

# Adding song names as labels on the bars
for bar, song_name in zip(bars, most_popular_per_year['name']):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, song_name, fontsize=10, ha='center')
plt.title('Most Popular Song by Year')
plt.xlabel('Release Year')
plt.ylabel('Popularity')
plt.tight_layout()
plt.grid(axis='y')
plt.show()



pointBiserialCorr = stats.pointbiserialr(x=df['is_explicit'], y=df['danceability'])
print(f"Point Biserial Correlation: {pointBiserialCorr}")
Point Biserial Correlation: SignificanceResult(statistic=0.34633359217555554, pvalue=0.0)

#We used Point Biserial Correlation, since it is useful when comparing binomial and continuos variables. We see a slight correlation and a significant pvalue.

plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', 
            linewidths=1, annot_kws={'size': 7})

#Loudness and energy share a relationship. This may seem obvious, louder music is more intense and conveys and energetic feeling.

px.scatter(df, x='loudness', y='energy', template='plotly_dark',
          color_discrete_sequence=['#1DB954'],
           title= 'Strong correlation between energy and loudness'
      ).update_layout(
        font = dict(size=14,family="Franklin Gothic"))

px.scatter(df, x='danceability', y='energy', template='plotly_dark',
          color_discrete_sequence=['#1DB954'],
           title= 'Strong correlation between danceability and energy'
      ).update_layout(
        font = dict(size=14,family="Franklin Gothic"))      

#In this section we see there is not likely to be presence of people during recordings.
      
px.scatter(df, x='acousticness', y='energy', template='plotly_dark',
          color_discrete_sequence=['#1DB954'],
           title= 'Negative correlation between acousticness and energy'
      ).update_layout(
        font = dict(size=14,family="Franklin Gothic"))

"""Data Transformation:Label Encoding for Categorical Variables, Standardization of Continous Variables"""

df.dropna(subset=['artists'], inplace=True)

def explicit(i):
    return(
    df.loc[:,'is_explicit']
    .replace(False, 0)
    .replace(True, 1)
    )
df = df.assign(is_explicit=explicit)

df['country'] = df['country'].fillna('Global')

label_encoder = LabelEncoder()
df['artists_encoded'] = label_encoder.fit_transform(df['artists'])
df['country_encoded'] = label_encoder.fit_transform(df['country'])
# Display the DataFrame after encoding
#print("\nDataFrame after Label Encoding:")
#print(df.head())


# Assuming 'df' is your DataFrame and 'cols_to_standardize' is the list of column names to standardize
cols_to_standardize = ['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

# Extract the selected columns
selected_cols = df[cols_to_standardize]

# Standardize the selected columns
scaler = StandardScaler()
standardized_cols = scaler.fit_transform(selected_cols)

# Replace the original columns with the standardized ones
df[cols_to_standardize] = standardized_cols

# Now, 'df' contains the standardized values for the specified columns

"""Model 1: Linear Regression"""

# Split the data
X_cat = df[['artists_encoded', 'country_encoded', 'is_explicit']].values  # Categorical features
X_num = df[['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].values
y = df['popularity'].values
X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X_cat, X_num, y, test_size=0.2, random_state=42)

# Create a linear regression model for comparison
model_lr = LinearRegression()

# Train the linear regression model
model_lr.fit(X_train_num, y_train)

# Make predictions using the linear regression model
y_pred_lr = model_lr.predict(X_test_num)

# Evaluate the linear regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Mean Squared Error (Linear Regression): {mse_lr}')

# Calculate R2 metric for the linear regression model
r_squared_lr = r2_score(y_test, y_pred_lr)
print(f'R-squared (Linear Regression): {r_squared_lr}')

"""Model 2: Neural Network with 1 Hidden layer containing 64 units with ReLu activation and Loss function as Mean Squared Error; No validation data while model fitting

"""

# Split the data
X_cat = df[['artists_encoded', 'country_encoded', 'is_explicit']].values  # Categorical features
X_num = df[['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].values
y = df['popularity'].values
X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X_cat, X_num, y, test_size=0.2, random_state=42)

# Build a neural network with both categorical and numerical features
input_cat = Input(shape=(X_train_cat.shape[1],))
embedding = Embedding(input_dim=np.max(X_train_cat) + 1, output_dim=10, input_length=1)(input_cat)
flatten_cat = Flatten()(embedding)

input_num = Input(shape=(X_train_num.shape[1],))
concatenated = Concatenate()([flatten_cat, input_num])

dense_layer = Dense(64, activation='relu')(concatenated)
output_layer = Dense(1, activation='linear')(dense_layer)

# Create the model
model = Model(inputs=[input_cat, input_num], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train_cat, X_train_num], y_train, epochs=10, batch_size=32)

# Make predictions
y_pred = model.predict([X_test_cat, X_test_num])

# Evaluate the model
mse = model.evaluate([X_test_cat, X_test_num], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

#Calculating R2 metric for the model
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

"""Model 3: Neural Network with 1 Hidden layer containing 64 units with ReLu activation and Loss function as Mean Squared Error; validation data while model fitting

"""

# Split the data
X_cat = df[['artists_encoded', 'country_encoded', 'is_explicit']].values  # Categorical features
X_num = df[['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].values
y = df['popularity'].values
# Split the data into training, validation, and test sets
X_train_cat, X_temp_cat, X_train_num, X_temp_num, y_train, y_temp = train_test_split(X_cat, X_num, y, test_size=0.4, random_state=42)
X_val_cat, X_test_cat, X_val_num, X_test_num, y_val, y_test = train_test_split(X_temp_cat, X_temp_num, y_temp, test_size=0.5, random_state=42)

# Build a neural network with both categorical and numerical features
input_cat = Input(shape=(X_train_cat.shape[1],))
embedding = Embedding(input_dim=np.max(X_train_cat) + 1, output_dim=10, input_length=1)(input_cat)
flatten_cat = Flatten()(embedding)

input_num = Input(shape=(X_train_num.shape[1],))
concatenated = Concatenate()([flatten_cat, input_num])

dense_layer = Dense(64, activation='relu')(concatenated)
output_layer = Dense(1, activation='linear')(dense_layer)

# Create the model
model = Model(inputs=[input_cat, input_num], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train_cat, X_train_num], y_train, epochs=10, batch_size=32, validation_data=([X_val_cat, X_val_num], y_val))
# Make predictions
y_pred = model.predict([X_test_cat, X_test_num])

# Evaluate the model
mse = model.evaluate([X_test_cat, X_test_num], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

#Calculating R2 metric for the model
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

"""Model 4: Neural Network with 2 Hidden layers containing 64 units and 32 units with ReLu activation and Loss function as Mean Squared Error"""


# Split the data
X_cat = df[['artists_encoded', 'country_encoded', 'is_explicit']].values  # Categorical features
X_num = df[['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].values
y = df['popularity'].values
X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X_cat, X_num, y, test_size=0.2, random_state=42)

# Build a neural network with both categorical and numerical features
input_cat = Input(shape=(X_train_cat.shape[1],))
embedding = Embedding(input_dim=np.max(X_train_cat) + 1, output_dim=10, input_length=1)(input_cat)
flatten_cat = Flatten()(embedding)

input_num = Input(shape=(X_train_num.shape[1],))
concatenated = Concatenate()([flatten_cat, input_num])

dense_layer = Dense(64, activation='relu')(concatenated)
dense_layer2 = Dense(32, activation='relu')(dense_layer)
output_layer = Dense(1, activation='linear')(dense_layer2)

# Create the model
model = Model(inputs=[input_cat, input_num], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train_cat, X_train_num], y_train, epochs=10, batch_size=32)

# Make predictions
y_pred = model.predict([X_test_cat, X_test_num])

# Evaluate the model
mse = model.evaluate([X_test_cat, X_test_num], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

#Calculating R2 metric for the model
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')


"""Model 5: Neural Network with 3 Hidden layers containing 96 units,  64 units and 32 units with ReLu activation and Loss function as Mean Squared Error"""

# Split the data
X_cat = df[['artists_encoded', 'country_encoded', 'is_explicit']].values  # Categorical features
X_num = df[['daily_rank', 'daily_movement', 'weekly_movement', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].values
y = df['popularity'].values
X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X_cat, X_num, y, test_size=0.2, random_state=42)

# Build a neural network with both categorical and numerical features
input_cat = Input(shape=(X_train_cat.shape[1],))
embedding = Embedding(input_dim=np.max(X_train_cat) + 1, output_dim=10, input_length=1)(input_cat)
flatten_cat = Flatten()(embedding)

input_num = Input(shape=(X_train_num.shape[1],))
concatenated = Concatenate()([flatten_cat, input_num])

dense_layer = Dense(96, activation='relu')(concatenated)
dense_layer2 = Dense(64, activation='relu')(dense_layer)
dense_layer3 = Dense(32, activation='relu')(dense_layer2)
output_layer = Dense(1, activation='linear')(dense_layer3)

# Create the model
model = Model(inputs=[input_cat, input_num], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train_cat, X_train_num], y_train, epochs=10, batch_size=32)

# Make predictions
y_pred = model.predict([X_test_cat, X_test_num])

# Evaluate the model
mse = model.evaluate([X_test_cat, X_test_num], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

#Calculating R2 metric for the model
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')