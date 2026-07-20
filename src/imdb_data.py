"""IMDb non-commercial dataset loader.

Downloads the free TSV dumps from datasets.imdbws.com (see
https://developer.imdb.com/non-commercial-datasets/) and joins them into
Documents for the RAG index: titles (with rating/genres/top-billed cast) and
people (actors/actresses, with birth/death year and known-for titles).

The official IMDb API (developer.imdb.com) is GraphQL, paid, and gated behind
AWS Data Exchange with no public key — these datasets are the free path.
They ship no plot summaries, only structured facts.
"""
from __future__ import annotations

import csv
import gzip
import heapq
import logging
import time
from pathlib import Path

import requests

from .config import settings
from .rag import Document, rag_store

logger = logging.getLogger(__name__)

_NULL = "\\N"


class IMDbDataSource:
    """Fetches and parses IMDb's free non-commercial dataset dumps."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.scraper_user_agent})
        self.timeout = settings.scraper_timeout
        self.data_dir = Path(settings.imdb_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _download(self, name: str) -> Path:
        """Download a dataset file if not already cached locally."""
        path = self.data_dir / name
        if path.exists():
            logger.debug("%s already cached at %s", name, path)
            return path

        url = f"{settings.imdb_dataset_base_url}/{name}"
        logger.info("Downloading %s...", url)
        t0 = time.monotonic()
        with self.session.get(url, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        logger.info("Downloaded %s in %.1fs (%d bytes)", name, time.monotonic() - t0, path.stat().st_size)
        return path

    def _rows(self, name: str):
        """Stream-parse a cached/downloaded dataset TSV as dicts."""
        path = self._download(name)
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t")

    @staticmethod
    def _val(row: dict, key: str, default: str = "") -> str:
        v = row.get(key, _NULL)
        return default if v == _NULL else v

    def top_titles(self, limit: int | None = None, min_votes: int | None = None) -> list[dict]:
        """Top-N titles by vote count, joining title.basics + title.ratings."""
        limit = limit or settings.imdb_max_titles
        min_votes = settings.imdb_min_votes if min_votes is None else min_votes
        allowed_types = set(settings.imdb_title_types)

        logger.info("Loading ratings (min_votes=%d)...", min_votes)
        ratings: dict[str, tuple[float, int]] = {}
        for row in self._rows("title.ratings.tsv.gz"):
            votes = int(row["numVotes"])
            if votes < min_votes:
                continue
            ratings[row["tconst"]] = (float(row["averageRating"]), votes)
        logger.info("  %d titles pass min_votes filter", len(ratings))

        logger.info("Scanning title.basics for top %d titles...", limit)
        heap: list[tuple[int, str, dict]] = []
        for row in self._rows("title.basics.tsv.gz"):
            tconst = row["tconst"]
            rating = ratings.get(tconst)
            if rating is None or row["titleType"] not in allowed_types or row.get("isAdult") == "1":
                continue
            avg_rating, votes = rating
            entry = {
                "tconst": tconst,
                "title": self._val(row, "primaryTitle"),
                "titleType": row["titleType"],
                "year": self._val(row, "startYear"),
                "genres": self._val(row, "genres"),
                "rating": avg_rating,
                "votes": votes,
            }
            if len(heap) < limit:
                heapq.heappush(heap, (votes, tconst, entry))
            elif votes > heap[0][0]:
                heapq.heapreplace(heap, (votes, tconst, entry))

        top = sorted((entry for _, _, entry in heap), key=lambda e: e["votes"], reverse=True)
        logger.info("  selected %d titles", len(top))
        return top

    def cast_for_titles(self, tconsts: set[str], per_title: int | None = None) -> dict[str, list[str]]:
        """tconst -> ordered list of nconst (top-billed actor/actress), from title.principals."""
        per_title = per_title or settings.imdb_cast_per_title
        cast: dict[str, list[str]] = {}
        logger.info("Scanning title.principals for cast of %d titles...", len(tconsts))
        for row in self._rows("title.principals.tsv.gz"):
            tconst = row["tconst"]
            if tconst not in tconsts or row["category"] not in ("actor", "actress"):
                continue
            bucket = cast.setdefault(tconst, [])
            if len(bucket) < per_title:
                bucket.append(row["nconst"])
        logger.info("  found cast for %d/%d titles", len(cast), len(tconsts))
        return cast

    def people(self, nconsts: set[str]) -> dict[str, dict]:
        """nconst -> profile dict, from name.basics."""
        found: dict[str, dict] = {}
        logger.info("Scanning name.basics for %d people...", len(nconsts))
        for row in self._rows("name.basics.tsv.gz"):
            nconst = row["nconst"]
            if nconst not in nconsts:
                continue
            known_for = self._val(row, "knownForTitles")
            found[nconst] = {
                "nconst": nconst,
                "name": self._val(row, "primaryName", "Unknown"),
                "birthYear": self._val(row, "birthYear"),
                "deathYear": self._val(row, "deathYear"),
                "professions": self._val(row, "primaryProfession"),
                "knownForTitles": known_for.split(",") if known_for else [],
            }
            if len(found) == len(nconsts):
                break
        logger.info("  resolved %d/%d people", len(found), len(nconsts))
        return found


def _title_document(entry: dict, cast_names: list[str]) -> Document:
    tconst = entry["tconst"]
    actors = ", ".join(cast_names) if cast_names else "Unknown"
    metadata = {
        "type": "title",
        "tconst": tconst,
        "title_type": entry["titleType"],
        "year": entry["year"],
        "genres": entry["genres"],
        "rating": entry["rating"],
        "votes": entry["votes"],
        "actors": actors,
        "url": f"https://www.imdb.com/title/{tconst}/",
    }
    content = (
        f"Title: {entry['title']} ({entry['year']})\n"
        f"Type: {entry['titleType']}\n"
        f"Genres: {entry['genres']}\n"
        f"Rating: {entry['rating']}/10 ({entry['votes']} votes)\n"
        f"Cast: {actors}"
    )
    return Document(f"title:{tconst}", entry["title"], metadata, content)


def _person_document(person: dict, known_for_titles: list[str]) -> Document:
    nconst = person["nconst"]
    known_for = ", ".join(known_for_titles) if known_for_titles else "See IMDb profile"
    lines = [
        f"Name: {person['name']}",
        f"Born: {person['birthYear'] or 'Unknown'}",
    ]
    if person["deathYear"]:
        lines.append(f"Died: {person['deathYear']}")
    lines.append(f"Professions: {person['professions'] or 'Unknown'}")
    lines.append(f"Known for: {known_for}")

    metadata = {
        "type": "person",
        "nconst": nconst,
        "name": person["name"],
        "birth_year": person["birthYear"],
        "death_year": person["deathYear"],
        "professions": person["professions"],
        "known_for": known_for_titles,
        "url": f"https://www.imdb.com/name/{nconst}/",
    }
    return Document(f"person:{nconst}", person["name"], metadata, "\n".join(lines))


def _build_docs(limit: int | None = None) -> tuple[list[Document], list[Document]]:
    """Top titles by votes + the actor/actress profiles of their cast."""
    source = IMDbDataSource()
    titles = source.top_titles(limit=limit)
    tconst_set = {t["tconst"] for t in titles}
    cast = source.cast_for_titles(tconst_set)

    all_nconsts = {n for names in cast.values() for n in names}
    people = source.people(all_nconsts)
    title_names = {t["tconst"]: t["title"] for t in titles}

    title_docs = []
    for entry in titles:
        nconsts = cast.get(entry["tconst"], [])
        cast_names = [people[n]["name"] for n in nconsts if n in people]
        title_docs.append(_title_document(entry, cast_names))

    person_docs = []
    for person in people.values():
        known_for = [title_names[t] for t in person["knownForTitles"] if t in title_names]
        person_docs.append(_person_document(person, known_for))

    return title_docs, person_docs


async def ingest_imdb_data(limit: int | None = None) -> int:
    """One-time ingest: top titles by vote count + their cast, indexed into the vector store."""
    title_docs, person_docs = _build_docs(limit)
    docs = title_docs + person_docs
    if docs:
        rag_store.add_documents(docs)
        logger.info("Indexed %d documents (%d titles, %d people)", len(docs), len(title_docs), len(person_docs))
    return len(docs)


async def ingest_people_only(limit: int | None = None) -> int:
    """Ingest actor/actress profiles only (cast of the top titles by vote count)."""
    _, person_docs = _build_docs(limit)
    if person_docs:
        rag_store.add_documents(person_docs)
        logger.info("Indexed %d person profiles", len(person_docs))
    return len(person_docs)
