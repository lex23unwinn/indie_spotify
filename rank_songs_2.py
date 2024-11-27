import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import MinMaxScaler

SPOTIFY_CLIENT_ID = '6c95c84aa7374880b523c85f05b54530'
SPOTIFY_CLIENT_SECRET = 'a5341d26f52d40bd84258c4d3fab2f5b'
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, 
client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


class KNNIndieSpotify:
    def __init__(self, data):
        self.data = data
        self.feature_columns = [
            "danceability", "energy", "valence", "liveness",
            "instrumentalness", "acousticness", "speechiness", "tempo",
            "loudness", "duration_ms"
        ]
        print(f"KNNIndieSpotify initialized with feature columns: {self.feature_columns}")
        print(f"Loaded dataset with {len(self.data)} songs.")

        self.scaler = MinMaxScaler()
        print("Normalizing dataset features...")
        self.data[self.feature_columns] = self.scaler.fit_transform(self.data[self.feature_columns])
        print("Dataset normalization complete.")

    def fetch_song_features(self, track_id):
        print(f"Fetching features for track ID: {track_id}")
        try:
            features = sp.audio_features([track_id])[0]
            if features is None:
                print(f"Features not found for track ID: {track_id}. Skipping...")
                return None

            feature_data = {key: features.get(key, 0) for key in self.feature_columns}
            feature_vector = np.array([feature_data[key] for key in self.feature_columns]).reshape(1, -1)

            normalized_features = self.scaler.transform(feature_vector)[0]
            print(f"Fetched and normalized features for {track_id}: {normalized_features}")
            return normalized_features
        except Exception as e:
            print(f"Error fetching features for track ID {track_id}: {e}")
            return None

    def fetch_track_info(self, track_id, popularity_threshold=50): 
        print(f"Fetching track info for track ID: {track_id}")
        try:
            track = sp.track(track_id)
            album_image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
            popularity = track.get('popularity')

            print(f"Track {track_id}: Popularity={popularity}, Album Image URL={album_image_url}")
            
            if popularity is not None and popularity >= popularity_threshold:
                print(f"Track {track_id} is too popular (popularity: {popularity}), skipping.")
                return None, None  

            if popularity is None:
                print(f"Track {track_id} has no popularity data, skipping.")
                return None, None
                
            return album_image_url, popularity
        except Exception as e:
            print(f"Error fetching track info for {track_id}: {e}")
            return None, None
        
    def verify_normalization(self):
        for feature in self.feature_columns:
            min_val = self.data[feature].min()
            max_val = self.data[feature].max()
            print(f"{feature}: Min={min_val}, Max={max_val}")

    def calculate_similarity(self, avg_vector, song_vector):
        avg_vector = np.array(avg_vector, dtype=float)
        song_vector = np.array(song_vector, dtype=float)
        similarity = np.dot(avg_vector, song_vector) / (
            (np.linalg.norm(avg_vector) * np.linalg.norm(song_vector)) + .001
        )
        return similarity

    def recommend_songs(self, song_ids, popularity_threshold=50):

        print("Starting recommendation process...")
        print(f"Input song IDs: {song_ids}")
        print(f"Popularity threshold: {popularity_threshold}")

        feature_vectors = []
        print("Fetching features for input songs from the Spotify API...")
        for track_id in song_ids:
            features = self.fetch_song_features(track_id)
            if features is not None:
                feature_vectors.append(features)
            else:
                print(f"Skipping track ID {track_id} due to missing features.")

        if not feature_vectors:
            print("No valid features found for the given song IDs. Returning empty recommendations.")
            return []

        avg_vector = np.mean(feature_vectors, axis=0)

        ranked_songs = []
        print("Calculating similarity for all songs in the local dataset...")
        for index, song in self.data.iterrows():
            try:
                song_vector = song[self.feature_columns].values
                similarity = self.calculate_similarity(avg_vector, song_vector)
                print(f"Song ID: {song['id']}, Similarity: {similarity}")
                ranked_songs.append({
                    "name": song["name"],
                    "artists": song["artists"],
                    "album_name": song["album_name"],
                    "spotify_url": f"https://open.spotify.com/track/{song['id']}",
                    "id": song["id"],
                    "similarity": similarity,
                    "song_vector": song_vector
                })
            except Exception as e:
                print(f"Error processing song ID {song['id']}: {e}")
                continue

        self.verify_normalization()

        print(f"Computed average feature vector for input songs: {avg_vector}")

        print(f"Sorting top 100 songs by similarity...")
        ranked_songs = sorted(ranked_songs, key=lambda x: x["similarity"], reverse=True)[:100]

        final_results = []
        for song in ranked_songs:
            album_image_url, popularity = self.fetch_track_info(song["id"])

            if popularity is not None and popularity < popularity_threshold:
                song["popularity"] = popularity
                song["album_image_url"] = album_image_url
                final_results.append(song)

                print("\nFeature Comparison for Final Recommendation:")
                print(f"Song Vector: {song['song_vector']}")
                print(f"Average Vector: {avg_vector}")
                print(f"Similarity Score: {song['similarity']}")

            if len(final_results) >= 10:
                break

        print(f"Final results prepared with {len(final_results)} songs.")
        return final_results
