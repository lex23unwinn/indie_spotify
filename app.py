from flask import Flask, render_template, request
import os
import pandas as pd
import kaggle
from rank_songs import BM25IndieSpotify
from rank_songs_2 import KNNIndieSpotify

if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset_path = "datasets/songs_with_attributes_and_lyrics.csv/songs_with_attributes_and_lyrics.csv"
if not os.path.exists(dataset_path):
    print("Dataset not found locally. Downloading from Kaggle...")
    kaggle.api.dataset_download_files("bwandowando/spotify-songs-with-attributes-and-lyrics", path="datasets/", unzip=True)
else:
    print("Dataset already downloaded.")

print("Loading dataset...")
data = pd.read_csv(dataset_path, nrows=20000)
print("Dataset loaded.")

bm25 = BM25IndieSpotify(data)

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    print("Home page accessed.")
    return render_template('home.html') 

@app.route('/results', methods=['POST'])
def results():
    query = request.form.get('query')
    popularity_threshold = int(request.form.get('popularity_threshold', 50)) 
    print(f"Received query: '{query}' with popularity threshold: {popularity_threshold}")
    
    ranked_songs = bm25.rank_songs(query, popularity_threshold=popularity_threshold)
    return render_template('results.html', query=query, ranked_songs=ranked_songs)

@app.route('/knn')
def knn_home():
    print("KNN Home page accessed.")
    return render_template('knn_home.html')

@app.route('/knn_results', methods=['POST'])
def knn_results():
    song_ids = [request.form.get(f"song{i}") for i in range(1, 6) if request.form.get(f"song{i}")]
    popularity_threshold = int(request.form.get('popularity_threshold', 50))
    
    print(f"Received song IDs: {song_ids} with popularity threshold: {popularity_threshold}")

    if not song_ids:
        print("No song IDs provided.")
        return render_template('knn_results.html', error="Please provide at least one song ID.")

    knn_model = KNNIndieSpotify(data)
    recommendations = knn_model.recommend_songs(song_ids, popularity_threshold=popularity_threshold)
    return render_template('knn_results.html', recommendations=recommendations)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)