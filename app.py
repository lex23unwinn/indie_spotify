from flask import Flask, render_template, request
import os
import pandas as pd
import kaggle
from rank_songs import BM25IndieSpotify

if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset_path = "datasets/songs_with_attributes_and_lyrics.csv/songs_with_attributes_and_lyrics.csv"
if not os.path.exists(dataset_path):
    print("Dataset not found locally. Downloading from Kaggle...")
    kaggle.api.dataset_download_files("bwandowando/spotify-songs-with-attributes-and-lyrics", path="datasets/", unzip=True)
else:
    print("Dataset already downloaded.")

print("Loading dataset...")
data = pd.read_csv(dataset_path, nrows=5000)
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

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)