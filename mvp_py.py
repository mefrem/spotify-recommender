from flask import Flask, request, render_template, jsonify
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io



df = pd.read_csv("https://raw.githubusercontent.com/Build-Week-Spotify-Song-Suggester-1/Data-science/master/MusicWithGenresFiltered.csv")
def process_input(song_id, return_json = True):
    c = ["duration_ms", "index", "genre", "artist_name", "track_id", "track_name", "key", "mode"] # Columns to Omit
    song = df[df["track_id"] == song_id].iloc[0] # Get Song
    df_selected = df.copy()
    if not pd.isnull(song["genre"]): # If genre, set subset to only genre
        df_selected = df[df["genre"] == song["genre"]]
    nn = NearestNeighbors(n_neighbors=31, algorithm="kd_tree") # Nearest Neighbor Model
    nn.fit(df_selected.drop(columns=c))
    song = song.drop(index=c)
    song = np.array(song).reshape(1, -1)
    if return_json is False:
        return df_selected.iloc[nn.kneighbors(song)[1][0][1:]] # Return results as df
    return df_selected.iloc[nn.kneighbors(song)[1][0][1:]].to_json(orient="records") # Return results as json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/song/<song_id>', methods = ['GET'])
def song(song_id):
    """Route for recommendations based on song selected."""
    return  process_input(song_id) #jsonify(recomendations)

@app.route('/favorites',methods = ['GET', 'POST'])
def favorites():
    my_dict = request.get_json(force=True)
    track_list = pd.DataFrame()
    for i in my_dict.values():
        track_list = track_list.append(process_input(i,False))
    track_list.drop_duplicates()
    return track_list.sample(30).to_json(orient="records")

@app.route('/image/<song_id>', methods = ['GET'])
def radar_map(song_id):
    """Route for returning radar graph."""
    c = ["acousticness", "danceability", "energy", "valence"] # Columns to Show
    N = len(c)
    values=df[df["track_id"] == song_id].iloc[0][c].tolist()
    values += values[:1]
    print(values)
    angles = [n / float(N) * 2 * 3.141 for n in range(N)]
    angles += angles[:1]
    print(angles)
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], c, color='grey', size=8)
    plt.yticks([], [], color="grey", size=7)
    ax.set_rlabel_position(0)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    pic_bytes = io.BytesIO()
    plt.savefig(pic_bytes, format="png")
    pic_bytes.seek(0)
    data = base64.b64encode(pic_bytes.read()).decode("ascii")
    plt.clf()
    return "<img src='data:image/png;base64,{}'>".format(data)
