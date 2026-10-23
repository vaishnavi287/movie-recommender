# Movie Recommender

A content-based movie recommendation web app using TF-IDF and cosine similarity. Built with Flask, scikit-learn and pandas, in a single Python file.



## Features
- Find movies similar to a title you pick
- Describe what you want in plain words, for example "love story on a ship"
- Filter by genre and choose how many results to see
- "More like this" button on every result
- Upload your own CSV dataset from the page (title, genres, overview columns)
- Comes with a small built-in sample dataset so it runs immediately

## Run it
Requires Python 3.9+.

```
pip install -r requirements.txt
python movie_recommender.py
```

Open http://127.0.0.1:5002. To use your own data on startup:

```
python movie_recommender.py --data my_movies.csv
```

The CSV needs the columns `title`, `genres` and `overview`. Genres are separated by spaces, for example `Sci-Fi Thriller`.

## REST API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/meta` | Titles, genres and dataset size |
| GET | `/api/similar?title=...&n=5&genres=Drama,Comedy` | Movies similar to a title |
| GET | `/api/search?q=...&n=5` | Movies matching a text description |
| POST | `/api/upload` | Replace the dataset (send the CSV as the request body, `text/csv`, max 5 MB) |

## Tech stack
Python, Flask, scikit-learn (TfidfVectorizer, cosine similarity), pandas, HTML/CSS, JavaScript.

## How it works
Each movie's genres (weighted double) and overview are combined into one text and converted to a TF-IDF vector. Similarity between two movies is the cosine similarity of their vectors. A free-text query is converted with the same vectorizer, so it can be compared against every movie in the same way.

## Limitations and ideas for improvement
- TF-IDF matches words, not meaning. Sentence embeddings would capture similar plots with different wording.
- Try the full TMDB 5000 dataset from Kaggle for much better results
- Evaluate recommendations (for example precision@k) and add user ratings
