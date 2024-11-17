import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

SPOTIFY_CLIENT_ID = '6c95c84aa7374880b523c85f05b54530'
SPOTIFY_CLIENT_SECRET = 'a5341d26f52d40bd84258c4d3fab2f5b'
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

class BM25IndieSpotify:
    
    def __init__(self, data, k1=1.5, b=0.75):
        print("Initializing BM25IndieSpotify...")
        self.data = data
        self.k1 = k1
        self.b = b
        
        print("Calculating average lyrics length...")
        self.avg_lyrics_len = data['lyrics'].str.split().str.len().mean()
        print(f"Average lyrics length calculated: {self.avg_lyrics_len}")
        
        print("Calculating document frequencies...")
        self.df_dict = defaultdict(int)
        for index, lyrics in enumerate(self.data['lyrics'].dropna()):
            unique_terms = set(lyrics.lower().split())
            for term in unique_terms:
                self.df_dict[term] += 1
            if index % 100 == 0: 
                print(f"Processed {index} lyrics rows.")
        print("Document frequencies computed.")

    def fetch_track_info(self, track_id, popularity_threshold=50): 
        print(f"Fetching track info for track ID: {track_id}")
        try:
            track = sp.track(track_id)
            album_image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
            popularity = track['popularity']

            print(f"Track {track_id}: Popularity={popularity}, Album Image URL={album_image_url}")
            if popularity >= popularity_threshold:
                print(f"Track {track_id} is too popular (popularity: {popularity}), skipping.")
                return None, None  

            if popularity is None:
                print(f"Track {track_id} has no popularity data, skipping.")
                return None, None
            
            return album_image_url, popularity
        except Exception as e:
            print(f"Error fetching track info for {track_id}: {e}")
            return None, None

    def calculate_textual_score(self, query, song_lyrics, song_artists, song_album, song_title, artist_boost=5.0, album_boost=3.0, title_boost=2.0):
        print(f"Calculating textual score for query: '{query}'")
        if pd.isna(song_lyrics):
            print("No lyrics available for this song, score set to 0.")
            return 0

        query_terms = query.lower().split()
        lyrics_terms = song_lyrics.lower().split()
        doc_length = len(lyrics_terms)
        avg_doc_length = self.avg_lyrics_len

        score = 0

        if isinstance(song_artists, str):
            print(f"Checking query terms in artists: {song_artists}")
            song_artists_lower = song_artists.lower().split()
            for term in query_terms:
                if term in song_artists_lower:
                    score += artist_boost
                    print(f"Boosted score for matching artist term: {term}")
                    break 
        if isinstance(song_album, str):
            print(f"Checking query terms in album name: {song_album}")
            song_album_lower = song_album.lower().split()
            for term in query_terms:
                if term in song_album_lower:
                    score += album_boost
                    print(f"Boosted score for matching album term: {term}")
                    break
        if isinstance(song_title, str):
            print(f"Checking query terms in title: {song_title}")
            song_title_lower = song_title.lower().split()
            for term in query_terms:
                if term in song_title_lower:
                    score += title_boost
                    print(f"Boosted score for matching title term: {term}")
                    break

        print("Calculating term frequencies for lyrics...")
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
                print(f"Term '{term}': TF={tf}, DF={df}, IDF={idf}, Score Contribution={idf * tf_component}")

        print(f"Total score for this song: {score}")
        return score

    def rank_songs(self, query, popularity_threshold=50):
        print(f"Ranking songs for query: '{query}' with popularity threshold: {popularity_threshold}")
        ranked_songs = []

        for index, song in self.data.iterrows():
            try:
                print(f"Processing song ID: {song['id']}")
                bm25_lyrics_score = self.calculate_textual_score(
                    query, song['lyrics'], song['artists'], song['album_name'], song['name']
                )
                
                ranked_songs.append({
                    'name': song['name'],
                    'score': bm25_lyrics_score,
                    'artists': song['artists'],
                    'album_name': song['album_name'],
                    'spotify_url': f"https://open.spotify.com/track/{song['id']}",
                    'id': song['id']
                })
            except Exception as e:
                print(f"Error processing song ID {song['id']}: {e}")
                continue

        print(f"Sorting top 100 songs by score...")
        ranked_songs = sorted(ranked_songs, key=lambda x: x['score'], reverse=True)[:100]

        final_results = []
        for song in ranked_songs:
            if len(final_results) >= 10:
                print("Collected 10 valid results, stopping further processing.")
                break

            try:
                print(f"Fetching metadata for ranked song ID: {song['id']}")
                album_image_url, popularity = self.fetch_track_info(song['id'], popularity_threshold)
                if album_image_url is None: 
                    print(f"Song ID {song['id']} skipped due to popularity or API error.")
                    continue
                
                song['popularity'] = popularity
                song['album_image_url'] = album_image_url
                final_results.append(song)
                print(f"Added song ID {song['id']} to final results.")
            
            except Exception as e:
                print(f"Error fetching additional info for song ID {song['id']}: {e}")
                continue

        print(f"Final results prepared with {len(final_results)} songs.")
        return final_results
