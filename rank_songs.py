import os
import pandas as pd
import numpy as np
import kaggle
from collections import defaultdict

class BM25IndieSpotify:
    
    def __init__(self, data, k1=1.5, b=0.75, lyrics_weight=0.7, features_weight=0.3):

        self.data = data
        self.k1 = k1
        self.b = b
        self.lyrics_weight = lyrics_weight 
        self.features_weight = features_weight 
        
        self.avg_lyrics_len = data['lyrics'].str.split().str.len().mean()
        
        self.df_dict = defaultdict(int)
        for lyrics in self.data['lyrics'].dropna():
            unique_terms = set(lyrics.lower().split())
            for term in unique_terms:
                self.df_dict[term] += 1


    def calculate_textual_score(self, query, song_lyrics, song_artists, song_album, song_title, artist_boost=5.0, album_boost=3.0, title_boost=2.0):

        if pd.isna(song_lyrics):
            return 0

        query_terms = query.lower().split()
        lyrics_terms = song_lyrics.lower().split()
        doc_length = len(lyrics_terms)
        avg_doc_length = self.avg_lyrics_len

        score = 0

        if isinstance(song_artists, str) and any(any(term in artist.lower() for artist in song_artists.split(', ')) for term in query_terms):
            score += artist_boost  

        if isinstance(song_album, str) and any(term in song_album.lower() for term in query_terms):
            score += album_boost 

        if isinstance(song_title, str) and any(term in song_title.lower() for term in query_terms):
            score += title_boost

        tf_lyrics = defaultdict(int)
        for term in lyrics_terms:
            tf_lyrics[term] += 1

        for term in query_terms:
            tf = tf_lyrics.get(term, 0)
            if tf > 0:
                df = self.df_dict.get(term, 0)

                idf = np.log(len(self.data) / (df + 1))
                
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length)))
                
                score += idf * tf_component

        return score
    

    def calculate_numerical_score(self, query, song):

        score = 0

        query_features = {
            'danceability': (
                (0.6, 1.0) if any(word in query for word in [
                    'party', 'dance', 'fun', 'energetic', 'groovy', 'upbeat', 'bouncy', 'lively', 'chill', 'laid-back',
                    'disco', 'pop', 'funky', 'salsa'
                ]) 
                else (0.0, 0.4) if any(word in query for word in [
                    'calm', 'still', 'subdued', 'somber', 'melancholic', 'introspective'
                ])
                else (0.4, 0.6)  
            ),
            
            'energy': (
                (0.7, 1.0) if any(word in query for word in [
                    'upbeat', 'high-energy', 'intense', 'exciting', 'powerful', 'fast', 'dynamic', 'energetic', 'vibrant',
                    'rock', 'electronic', 'metal', 'disco', 'pop'
                ])
                else (0.0, 0.3) if any(word in query for word in [
                    'calm', 'peaceful', 'relaxed', 'soothing', 'gentle', 'soft', 'ambient'
                ])
                else (0.3, 0.7)  
            ),
            
            'valence': (
                (0.5, 1.0) if any(word in query for word in [
                    'happy', 'cheerful', 'positive', 'joyful', 'uplifting', 'bright', 'optimistic', 'feel-good', 'sunny',
                    'pop', 'reggae', 'disco', 'folk'
                ])
                else (0.0, 0.4) if any(word in query for word in [
                    'sad', 'dark', 'moody', 'melancholic', 'blue', 'downbeat', 'introspective'
                ])
                else (0.4, 0.6)  
            ),
            
            'acousticness': (
                (0.6, 1.0) if any(word in query for word in [
                    'acoustic', 'folk', 'unplugged', 'soft', 'mellow', 'organic', 'natural', 'stripped', 'minimal',
                    'country', 'classical', 'folk'
                ])
                else (0.0, 0.4) if any(word in query for word in [
                    'electronic', 'synth', 'techno', 'dance', 'disco', 'pop'
                ])
                else (0.4, 0.6)
            ),
            
            'instrumentalness': (
                (0.5, 1.0) if any(word in query for word in [
                    'instrumental', 'background', 'focus', 'ambient', 'non-vocal', 'soundscape', 'score', 'classical', 'lo-fi'
                ])
                else (0.0, 0.3) if any(word in query for word in [
                    'vocal', 'singing', 'rap', 'lyrics'
                ])
                else (0.3, 0.5) 
            ),
            
            'liveness': (
                (0.6, 1.0) if any(word in query for word in [
                    'live', 'concert', 'performance', 'crowd', 'raw', 'spontaneous', 'unfiltered', 'intimate', 'on-stage',
                    'jazz', 'blues'
                ])
                else (0.0, 0.4) if any(word in query for word in [
                    'studio', 'produced', 'polished', 'edited'
                ])
                else (0.4, 0.6) 
            ),
            
            'speechiness': (
                (0.5, 1.0) if any(word in query for word in [
                    'spoken', 'rap', 'lyrics', 'speech', 'spoken-word', 'monologue', 'conversation', 'narrative',
                    'hip-hop', 'rap'
                ])
                else (0.0, 0.3) if any(word in query for word in [
                    'instrumental', 'ambient', 'background', 'focus'
                ])
                else (0.3, 0.5)
            ),
            
            'tempo': (
                (120, 200) if any(word in query for word in [
                    'fast', 'upbeat', 'quick', 'high-tempo', 'rapid', 'energetic', 'lively', 'speedy',
                    'disco', 'electronic', 'dance', 'salsa'
                ])
                else (60, 100) if any(word in query for word in [
                    'slow', 'chill', 'calm', 'relaxed', 'mellow', 'ambient'
                ])
                else (100, 120) 
            ),
            
            'loudness': (
                (-10, 0) if any(word in query for word in [
                    'loud', 'intense', 'high-volume', 'powerful', 'booming', 'strong', 'amplified', 'forceful',
                    'rock', 'metal', 'hip-hop'
                ])
                else (-30, -20) if any(word in query for word in [
                    'soft', 'quiet', 'gentle', 'ambient', 'mellow', 'chill', 'relax'
                ])
                else (-20, -10) 
            ),
            
            'duration_ms': (
                (180000, 300000) if any(word in query for word in [
                    'long', 'extended', 'lengthy', 'drawn-out', 'full-length', 'epic', 'unabridged', 'continuous'
                ])
                else (0, 120000) if any(word in query for word in [
                    'short', 'brief', 'quick', 'snippet', 'intro', 'quick'
                ])
                else (120000, 180000) 
            ),
            
            'key': (
                (5, 7) if any(word in query for word in [
                    'minor', 'melancholic', 'dark', 'moody', 'somber', 'reflective', 'sad', 'blue',
                    'classical', 'jazz', 'blues'
                ])
                else (0, 4) if any(word in query for word in [
                    'bright', 'positive', 'happy', 'major', 'upbeat', 'joyful', 'cheerful', 'optimistic'
                ])
                else (4, 5) 
            ),
            
            'mode': (
                (1, 1) if any(word in query for word in [
                    'major', 'happy', 'positive', 'joyful', 'bright', 'cheerful', 'uplifting', 'content'
                ])
                else (0, 0) if any(word in query for word in [
                    'minor', 'sad', 'melancholic', 'dark', 'moody', 'reflective', 'pain', 'cry'
                ])
                else (0, 1) 
            ),
        }
        
        for feature, (min_val, max_val) in query_features.items():
            try:
                feature_value = float(song[feature])
                if min_val <= feature_value <= max_val:
                    score += 1 
            except (ValueError, TypeError):
                continue

        return score
    

    def rank_songs(self, query):
        ranked_songs = []

        for _, song in self.data.iterrows():
            bm25_lyrics_score = self.calculate_textual_score(query, song['lyrics'], song['artists'], song['album_name'], song['name'])
            numeric_score = self.calculate_numerical_score(query, song)
            
            combined_score = (self.lyrics_weight * bm25_lyrics_score) + (self.features_weight * numeric_score)
            ranked_songs.append((song['name'], combined_score))
        
        ranked_songs = sorted(ranked_songs, key=lambda x: x[1], reverse=True)
        return ranked_songs[:10]
    
    
if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset_path = "datasets/songs_with_attributes_and_lyrics.csv"

if not os.path.exists(dataset_path):
    print("Dataset not found locally. Downloading from Kaggle...")
    kaggle.api.dataset_download_files("bwandowando/spotify-songs-with-attributes-and-lyrics", path="datasets/", unzip=True)
else:
    print("Dataset already downloaded.")

dataset_path_contents = "datasets/songs_with_attributes_and_lyrics.csv/songs_with_attributes_and_lyrics.csv"
# dataset_path_contents = "datasets/songs_with_attributes_and_lyrics.csv"

data = pd.read_csv(dataset_path_contents, nrows=1000)

bm25 = BM25IndieSpotify(data=data)

query = "funky upbeat dance music"
ranked_songs = bm25.rank_songs(query)
print("Ranked Songs:")
for idx, (song_name, score) in enumerate(ranked_songs, start=1):
    print(f"{idx}. {song_name} - Score: {score:.2f}")
