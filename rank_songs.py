import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

SPOTIFY_CLIENT_ID = '6c95c84aa7374880b523c85f05b54530'
SPOTIFY_CLIENT_SECRET = 'a5341d26f52d40bd84258c4d3fab2f5b'
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=3)

class BM25IndieSpotify:
    
    def __init__(self, data, k1=1.5, b=0.75):
        print("Initializing BM25IndieSpotify...")
        self.data = data
        self.k1 = k1
        self.b = b
        
        self.avg_lyrics_len = data['lyrics'].str.split().str.len().mean()
        print(f"Average lyrics length calculated: {self.avg_lyrics_len}")
        
        self.df_dict = defaultdict(int)
        for lyrics in self.data['lyrics'].dropna():
            unique_terms = set(lyrics.lower().split())
            for term in unique_terms:
                self.df_dict[term] += 1
        print("Document frequencies computed.")


    def fetch_track_info(self, track_id, popularity_threshold=50): 
        print(f"Fetching track info for track ID: {track_id}")
        try:
            track = sp.track(track_id)
            album_image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
            popularity = track['popularity']

            if popularity >= popularity_threshold:
                print(f"Track {track_id} is too popular (popularity: {popularity}), skipping.")
                return None, None  

            if popularity is None:
                return None, None
            
            return album_image_url, popularity
        except Exception as e:
            print(f"Error fetching track info: {e}")
            return None, None
        

    def calculate_textual_score(self, query, song_lyrics, song_artists, song_album, song_title, artist_boost=5.0, album_boost=3.0, title_boost=1.0):
        print(f"Calculating textual score for query: '{query}'")

        if pd.isna(song_lyrics):
            print("No lyrics available, score set to 0.")
            return 0

        query_terms = query.lower().split()
        lyrics_terms = song_lyrics.lower().split()
        doc_length = len(lyrics_terms)
        avg_doc_length = self.avg_lyrics_len

        score = 0

        if isinstance(song_artists, str):
            song_artists_lower = song_artists.lower().split()
            for term in query_terms:
                if term in song_artists_lower:
                    score += artist_boost
                    break 
        if isinstance(song_album, str):
            song_album_lower = song_album.lower().split()
            for term in query_terms:
                if term in song_album_lower:
                    score += album_boost
                    break
        if isinstance(song_title, str):
            song_title_lower = song_title.lower().split()
            for term in query_terms:
                if term in song_title_lower:
                    score += title_boost
                    break

        tf_lyrics = defaultdict(int)
        for term in lyrics_terms:
            tf_lyrics[term] += 1

        total_docs = len(self.data)

        for term in query_terms:
            tf = tf_lyrics.get(term, 0)
            if tf > 0:
                df = self.df_dict.get(term, 0)
                idf = np.log(total_docs / (df + 1))
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length)))
                score += idf * tf_component

        print(f"Score calculated: {score}")
        return score

    

    def rank_songs(self, query, popularity_threshold=50):
        ranked_songs = []
        
        for _, song in self.data.iterrows():
            try:
                album_image_url, popularity = self.fetch_track_info(song['id'], popularity_threshold)
                if album_image_url is None: 
                    continue
                
                bm25_lyrics_score = self.calculate_textual_score(
                    query, song['lyrics'], song['artists'], song['album_name'], song['name']
                )
                
                ranked_songs.append({
                    'name': song['name'],
                    'score': bm25_lyrics_score,
                    'artists': song['artists'],
                    'album_name': song['album_name'],
                    'spotify_url': f"https://open.spotify.com/track/{song['id']}",
                    'id': song['id'],
                    'popularity': popularity,
                    'album_image_url': album_image_url
                })
            
            except Exception as e:
                print(f"Error fetching track info for {song['id']}: {e}")
                continue
        
        top_songs = sorted(ranked_songs, key=lambda x: x['score'], reverse=True)[:10]
        return top_songs