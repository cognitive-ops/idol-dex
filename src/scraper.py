"""Web scraper for JAV data from dmm.co.jp, r18.com, javlibrary.com."""
from __future__ import annotations

import logging
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .config import settings
from .rag import Document, rag_store

logger = logging.getLogger(__name__)


class JAVScraper:
    """Scrape JAV metadata from multiple sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.scraper_user_agent})
        self.timeout = settings.scraper_timeout
        logger.debug("JAVScraper init: timeout=%ss ua=%s", self.timeout, settings.scraper_user_agent)

    def scrape_javlibrary(self, title_filter: str = "") -> list[Document]:
        """Scrape from javlibrary.com (metadata: title, actors, date, plot)."""
        docs = []
        try:
            # Note: javlibrary requires careful handling (robots.txt, rate limiting)
            # This is a placeholder — real implementation would use their search API
            # or respectful scraping with delays
            pass
        except Exception as e:
            logger.error("javlibrary scrape failed: %s", e)
        return docs

    def scrape_dmm(self, search_query: str = "新作") -> list[Document]:
        """Scrape from dmm.co.jp (requires proxy/VPN or auth from outside Japan)."""
        docs = []
        try:
            # DMM blocks non-Japanese IPs; needs proxy or setup from inside Japan
            # Placeholder for now
            pass
        except Exception as e:
            logger.error("dmm scrape failed: %s", e)
        return docs

    def _list_idol_urls(self, limit: int = 20) -> list[tuple[str, str]]:
        """Get (name, url) pairs from the idols listing page.

        Real DOM (verified against a live fetch of javdatabase.com/idols/):
        each card is <div class="card borderlesscard">, name+link is
        <p class="pcard"><a class="cut-text" href="/idols/slug/">Name</a></p>.
        """
        url = "https://www.javdatabase.com/idols/"
        logger.debug("GET %s (timeout=%ss)", url, self.timeout)
        t0 = time.monotonic()
        response = self.session.get(url, timeout=self.timeout)
        logger.debug("GET %s -> %d in %.2fs, %d bytes", url, response.status_code,
                     time.monotonic() - t0, len(response.content))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        pairs = []
        for link in soup.select("p.pcard a.cut-text"):
            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not name or not href:
                continue
            if not href.startswith("http"):
                href = f"https://www.javdatabase.com{href}"
            pairs.append((name, href))
            if len(pairs) >= limit:
                break
        logger.debug("_list_idol_urls: parsed %d idol links (limit=%d)", len(pairs), limit)
        return pairs

    def _scrape_idol_detail(self, name: str, url: str) -> Document | None:
        """Fetch one idol's detail page: age, DOB, debut date, cup, height, movie codes.

        Real DOM (verified against a live fetch of an idol detail page):
        - Profile stats are plain text with inline <b>Label:</b> <a>value</a>
          pairs inside the h1.idol-name's parent column div — no dedicated
          class per field, so stats are extracted via regex on flattened text.
        - Movie cards reuse the same .card.borderlesscard markup as the
          listing page; movie links are distinguished by href containing
          '/movies/' (idol links contain '/idols/').
        """
        import re

        logger.debug("GET %s (timeout=%ss)", url, self.timeout)
        t0 = time.monotonic()
        response = self.session.get(url, timeout=self.timeout)
        logger.debug("GET %s -> %d in %.2fs, %d bytes", url, response.status_code,
                     time.monotonic() - t0, len(response.content))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        h1 = soup.select_one("h1.idol-name")
        profile_container = h1.find_parent("div") if h1 else soup

        stats_text = profile_container.get_text(separator=" ", strip=True)

        def _extract(pattern: str, default: str = "Unknown") -> str:
            m = re.search(pattern, stats_text)
            return m.group(1).strip() if m else default

        age = _extract(r"Age:\s*(\d+)")
        dob = _extract(r"DOB:\s*([\d\-\?]+)")
        debut = _extract(r"Debut:\s*([\d\-\?]+)")
        cup = _extract(r"\bCup:\s*(\w+)")
        height = _extract(r"Height:\s*([\d]+\s*cm)")
        measurements = _extract(r"Measurements:\s*([\d\-]+)")

        # Movie codes: cards whose title link points at /movies/
        movies = []
        for link in soup.select("p.pcard a.cut-text"):
            href = link.get("href", "")
            if "/movies/" in href:
                code = link.get_text(strip=True)
                if code:
                    movies.append(code)

        logger.debug("parsed idol=%s age=%s dob=%s debut=%s cup=%s height=%s movies=%d",
                     name, age, dob, debut, cup, height, len(movies))

        doc_id = f"idol:{name.replace(' ', '_')[:50]}"
        metadata = {
            "type": "idol",
            "name": name,
            "age": age,
            "dob": dob,
            "debut": debut,
            "cup": cup,
            "height": height,
            "measurements": measurements,
            "movie_count": len(movies),
            "movie_codes": movies,
            "url": url,
        }
        content = (
            f"Idol: {name}\n"
            f"Age: {age}\n"
            f"DOB: {dob}\n"
            f"Debut: {debut}\n"
            f"Cup Size: {cup}\n"
            f"Height: {height}\n"
            f"Measurements: {measurements}\n"
            f"Movies ({len(movies)}): {', '.join(movies)}"
        )
        return Document(doc_id, f"Idol: {name}", metadata, content)

    def scrape_idols(self, limit: int = 20) -> list[Document]:
        """Scrape idol profiles: name, age, debut, cup size, movie codes.

        Two-stage: the /idols/ listing page only exposes name/age/cup/height
        (no debut date or movie list), so each idol's own detail page is
        fetched for the full profile. `limit` caps the number of detail-page
        requests since this is one HTTP call per idol.
        """
        docs = []
        try:
            logger.info("Fetching idol listing page...")
            idol_links = self._list_idol_urls(limit=limit)
            logger.info("  Found %d idols, fetching profiles...", len(idol_links))

            for name, url in idol_links:
                try:
                    doc = self._scrape_idol_detail(name, url)
                    if doc:
                        docs.append(doc)
                        logger.info("  ✅ %s: age=%s, debut=%s, movies=%s",
                                    name, doc.metadata['age'], doc.metadata['debut'],
                                    doc.metadata['movie_count'])
                    time.sleep(0.3)  # rate limit
                except Exception as e:
                    logger.error("  Error scraping %s: %s", name, e)
                    continue

            if docs:
                logger.info("  ✅ Collected %d idol profiles", len(docs))

        except Exception as e:
            logger.error("  Idol scraping failed (%s): %s", type(e).__name__, e)

        return docs

    def scrape_javdatabase(self, search_query: str = "") -> list[Document]:
        """Scrape from javdatabase.com (reliable JAV catalog)."""
        docs = []

        # JAVDatabase URLs
        urls = [
            "https://www.javdatabase.com/idols"            
        ]

        for url in urls:
            try:
                logger.info("Trying %s...", url)
                t0 = time.monotonic()
                response = self.session.get(url, timeout=self.timeout)
                logger.debug("GET %s -> %d in %.2fs", url, response.status_code, time.monotonic() - t0)

                if response.status_code == 404:
                    logger.warning("  404 - URL not found, trying next...")
                    continue

                response.raise_for_status()
                soup = BeautifulSoup(response.content, "lxml")

                # JAVDatabase specific selectors
                items = (
                    soup.select(".movie-item")
                    or soup.select(".video-item")
                    or soup.select("[class*='movie']")
                    or soup.select("[class*='video']")
                    or []
                )

                if not items:
                    logger.warning("  No items found, trying next URL...")
                    continue

                logger.info("  Found %d items", len(items))

                for item in items[:15]:  # limit to first 15
                    try:
                        # Extract title (try multiple selectors)
                        title_elem = (
                            item.select_one("h2 a")
                            or item.select_one("h3 a")
                            or item.select_one(".title a")
                            or item.select_one("a[title]")
                        )
                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)
                        product_url = title_elem.get("href", "")
                        if product_url and not product_url.startswith("http"):
                            product_url = f"https://www.javdatabase.com{product_url}"

                        # Extract actors/cast
                        actors_elem = (
                            item.select_one(".actors")
                            or item.select_one(".cast")
                            or item.select_one("[class*='actor']")
                        )
                        actors = (
                            ", ".join(
                                [a.get_text(strip=True) for a in actors_elem.select("a")]
                                or [actors_elem.get_text(strip=True)]
                            )
                            if actors_elem
                            else "Unknown"
                        )

                        # Extract release date
                        date_elem = (
                            item.select_one(".release-date")
                            or item.select_one(".date")
                            or item.select_one("[class*='date']")
                        )
                        date_str = (
                            date_elem.get_text(strip=True) if date_elem else ""
                        )

                        # Extract plot/description
                        plot_elem = (
                            item.select_one(".description")
                            or item.select_one(".synopsis")
                            or item.select_one("p")
                        )
                        plot = (
                            plot_elem.get_text(strip=True) if plot_elem else ""
                        )

                        # Create document
                        doc_id = f"javdb:{title.replace(' ', '_')[:50]}"
                        metadata = {
                            "actors": actors,
                            "release_date": date_str,
                            "source": "javdatabase.com",
                            "url": product_url,
                        }
                        content = (
                            f"Title: {title}\n"
                            f"Actors: {actors}\n"
                            f"Date: {date_str}\n"
                            f"Plot: {plot}"
                        )

                        docs.append(Document(doc_id, title, metadata, content))
                        time.sleep(0.3)  # rate limit

                    except Exception as e:
                        logger.error("  Error parsing item: %s", e)
                        continue

                if docs:
                    logger.info("  ✅ Collected %d documents from %s", len(docs), url)
                    break

            except Exception as e:
                logger.error("  Failed (%s): %s", type(e).__name__, e)
                continue

        # Fallback: demo data if scraping failed
        if not docs:
            logger.warning("Scraping unavailable. Loading demo data instead...")
            docs = self._get_demo_data()

        return docs

    # Alias for backward compatibility
    def scrape_r18(self, search_query: str = "") -> list[Document]:
        """Deprecated: use scrape_javdatabase instead."""
        logger.warning("r18.com is no longer maintained. Switching to javdatabase.com...")
        return self.scrape_javdatabase(search_query)

    def _get_demo_data(self) -> list[Document]:
        """Demo data for testing when live scraping fails."""
        demo_titles = [
            {
                "title": "SDMU-605 - Temptation Of A Married Woman",
                "actors": "Tsubomi, Aiko Natsukawa",
                "date": "2023-01-15",
                "plot": "A story about temptation and desire",
            },
            {
                "title": "DASD-802 - Innocent Angel",
                "actors": "Mio Kimijima",
                "date": "2023-02-10",
                "plot": "An innocent journey turns into passion",
            },
            {
                "title": "SSIS-123 - Perfect Companion",
                "actors": "Yuki Nagano",
                "date": "2023-03-05",
                "plot": "A perfect companion for everyday life",
            },
        ]

        docs = []
        for i, item in enumerate(demo_titles, 1):
            doc_id = f"demo:{i}"
            metadata = {
                "actors": item["actors"],
                "release_date": item["date"],
                "source": "demo",
                "url": "https://www.r18.com/",
            }
            content = (
                f"Title: {item['title']}\n"
                f"Actors: {item['actors']}\n"
                f"Date: {item['date']}\n"
                f"Plot: {item['plot']}"
            )
            docs.append(Document(doc_id, item["title"], metadata, content))

        return docs

    def scrape_all(self) -> list[Document]:
        """Scrape from all sources: movies + idol profiles."""
        all_docs = []
        logger.info("Scraping javdatabase.com movies...")
        all_docs.extend(self.scrape_javdatabase())
        logger.info("Scraping javdatabase.com idols...")
        all_docs.extend(self.scrape_idols())
        logger.info("Found %d documents total", len(all_docs))
        return all_docs


async def ingest_jav_data() -> int:
    """One-time ingest: scrape all sources (movies + idols) and index into FAISS."""
    scraper = JAVScraper()
    docs = scraper.scrape_all()
    if docs:
        rag_store.add_documents(docs)
        logger.info("Indexed %d documents", len(docs))
    return len(docs)


async def ingest_idols_only() -> int:
    """Ingest idol profiles only (name, age, debut, cup size, movie codes)."""
    scraper = JAVScraper()
    docs = scraper.scrape_idols()
    if docs:
        rag_store.add_documents(docs)
        logger.info("Indexed %d idol profiles", len(docs))
    return len(docs)
