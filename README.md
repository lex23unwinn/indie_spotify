# indie_spotify

## Howdy!

## How to run this core Indie Spotify algorithm (more tweaks coming soon) with a given simple test code to show it works over a small set of data:

## First, make sure to install the following dependencies:

import os (built in)
import pandas as pd
import numpy as np
import kaggle
from collections import defaultdict (built in)

## Next, go to the kaggle website, log in/create a kaggle account, click on your profile picture in the top-right corner and go to Account, scroll down to the API section and click on "Create New API Token". This will download a kaggle.json file to your computer. Then, run the following commands:

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list

## Then run "python3 rank_songs.py"

## The import os at the top should take care of downloading the dataset and creating a directory for that downloaded dataset, so you shouldn't really have to worry about that manually. 

## This script tests a custom BM25-based ranking algorithm designed to return songs ranked by textual relevance in their lyrics, with additional score boosts based on matches in artist name, album name, and song title. To enhance ranking precision, it also includes numerical scoring based on audio features (danceability, energy, etc.) that align with descriptive terms in the search query, ensuring higher rankings for songs with matching audial qualities.