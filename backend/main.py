import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

#DOTENV
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Add it in Render Environment Variables."
    )

#FASTAPI
app = FastAPI(
    title="Movie Recommender API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")

TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

#GLOBALS
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

loaded = False


#MODELS
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# HELPERS
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


def load_resources():

    global loaded
    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX

    if loaded:
        return

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    loaded = True


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:

    title_to_idx = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl invalid")


def get_local_idx_by_title(title: str) -> int:

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF map not loaded"
        )

    key = _norm_title(title)

    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    raise HTTPException(
        status_code=404,
        detail=f"Movie not found in dataset: {title}"
    )

#TMDB
async def tmdb_get(path: str, params: Dict[str, Any]):

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    timeout = httpx.Timeout(
        connect=10.0,
        read=20.0,
        write=10.0,
        pool=10.0
    )

    headers = {
        "Accept": "application/json",
        "User-Agent": "movie-recommender"
    }

    try:

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:

            r = await client.get(
                f"{TMDB_BASE}{path}",
                params=q
            )

            r.raise_for_status()

    except httpx.ConnectTimeout:
        raise HTTPException(504, "TMDB connection timeout")

    except httpx.ReadTimeout:
        raise HTTPException(504, "TMDB read timeout")

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.text
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

    return r.json()


async def tmdb_search_movies(query: str, page: int = 1):

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "language": "en-US",
            "page": page,
            "include_adult": "false"
        }
    )


async def tmdb_search_first(query: str):

    data = await tmdb_search_movies(query)

    results = data.get("results", [])

    return results[0] if results else None


async def tmdb_movie_details(movie_id: int):

    data = await tmdb_get(
        f"/movie/{movie_id}",
        {"language": "en-US"}
    )

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", [])
    )


async def tmdb_cards_from_results(results, limit=10):

    out = []

    for n in (results or [])[:limit]:

        out.append(
            TMDBMovieCard(
                tmdb_id=int(n["id"]),
                title=n.get("title") or "",
                poster_url=make_img_url(n.get("poster_path")),
                release_date=n.get("release_date"),
                vote_average=n.get("vote_average")
            )
        )

    return out

#TFIDF
def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
):

    global df
    global tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(500, "TF-IDF not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]

    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out = []

    for i in order:

        if int(i) == int(idx):
            continue

        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue

        out.append(
            (title_i, float(scores[int(i)]))
        )

        if len(out) >= top_n:
            break

    return out

#routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(10, ge=1, le=10)
):

    try:

        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {"language": "en-US"}
            )

            return await tmdb_cards_from_results(
                data.get("results", []),
                limit
            )

        if category not in {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing"
        }:
            raise HTTPException(400, "Invalid category")

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit
        )

    except Exception as e:
        raise HTTPException(500, str(e))



@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10)
):

    return await tmdb_search_movies(query, page)


@app.get("/movie/id/{tmdb_id}")
async def movie_details_route(tmdb_id: int):

    return await tmdb_movie_details(tmdb_id)

@app.get("/recommend/genre")
async def recommend_genre(
    tmdb_id: int,
    limit: int = Query(10, ge=1, le=20)
):

    details = await tmdb_movie_details(tmdb_id)

    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1
        }
    )

    cards = await tmdb_cards_from_results(
        discover.get("results", []),
        limit
    )

    return [c for c in cards if c.tmdb_id != tmdb_id]

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str,
    top_n: int = Query(5, ge=1, le=10)
):

    load_resources()

    recs = tfidf_recommend_titles(
        title,
        top_n=top_n
    )

    return [
        {
            "title": t,
            "score": s
        }
        for t, s in recs
    ]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str,
    tfidf_top_n: int = Query(5, ge=1, le=10),
    genre_limit: int = Query(10, ge=1, le=20)
):

    load_resources()

    best = await tmdb_search_first(query)

    if not best:
        raise HTTPException(
            404,
            f"No movie found: {query}"
        )

    tmdb_id = int(best["id"])

    details = await tmdb_movie_details(tmdb_id)

    tfidf_items = []

    recs = []

    try:
        recs = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n
        )
    except Exception:
        pass

    for title, score in recs:

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                tmdb=None
            )
        )

    genre_recs = []

    if details.genres:

        genre_id = details.genres[0]["id"]

        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1
            }
        )

        cards = await tmdb_cards_from_results(
            discover.get("results", []),
            limit=genre_limit
        )

        genre_recs = [
            c for c in cards
            if c.tmdb_id != details.tmdb_id
        ]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs
    )