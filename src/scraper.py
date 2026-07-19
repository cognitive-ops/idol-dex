"""Web scraper for JAV data from dmm.co.jp, r18.com, javlibrary.com."""
from __future__ import annotations

import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .config import settings
from .rag import Document, rag_store


class JAVScraper:
    """Scrape JAV metadata from multiple sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.scraper_user_agent})
        self.timeout = settings.scraper_timeout

    def scrape_javlibrary(self, title_filter: str = "") -> list[Document]:
        """Scrape from javlibrary.com (metadata: title, actors, date, plot)."""
        docs = []
        try:
            # Note: javlibrary requires careful handling (robots.txt, rate limiting)
            # This is a placeholder — real implementation would use their search API
            # or respectful scraping with delays
            pass
        except Exception as e:
            print(f"javlibrary scrape failed: {e}")
        return docs

    def scrape_dmm(self, search_query: str = "新作") -> list[Document]:
        """Scrape from dmm.co.jp (requires proxy/VPN or auth from outside Japan)."""
        docs = []
        try:
            # DMM blocks non-Japanese IPs; needs proxy or setup from inside Japan
            # Placeholder for now
            pass
        except Exception as e:
            print(f"dmm scrape failed: {e}")
        return docs

    def scrape_r18(self, search_query: str = "") -> list[Document]:
        """Scrape from r18.com (English mirror of DMM). Falls back to demo data if unavailable."""
        docs = []

        # Try multiple URL patterns
        urls = [
            "https://www.r18.com/videos/vod/movies/genre/latest/",
            "https://www.r18.com/common/search/searchlist/",
            "https://www.r18.com/videos/",
        ]

        for url in urls:
            try:
                print(f"Trying {url}...")
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 404:
                    print(f"  404 - URL not found, trying next...")
                    continue

                response.raise_for_status()
                soup = BeautifulSoup(response.content, "lxml")

                # Try multiple CSS selectors (site structure changes)
                items = (
                    soup.select(".productItem")
                    or soup.select("[data-product-id]")
                    or soup.select(".content-box")
                    or []
                )

                if not items:
                    print(f"  No items found, trying next URL...")
                    continue

                print(f"  Found {len(items)} items")

                for item in items[:10]:  # limit to first 10
                    try:
                        # Try multiple selectors for title
                        title_elem = (
                            item.select_one(".title a")
                            or item.select_one(".item-title a")
                            or item.select_one("h2 a")
                        )
                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)
                        product_url = title_elem.get("href", "")

                        # Extract metadata
                        actors_text = (
                            item.select_one(".performer")
                            or item.select_one(".cast")
                        )
                        actors = (
                            actors_text.get_text(strip=True)
                            if actors_text
                            else "Unknown"
                        )

                        date_text = (
                            item.select_one(".releaseDate")
                            or item.select_one(".date")
                        )
                        date_str = (
                            date_text.get_text(strip=True) if date_text else ""
                        )

                        plot_text = (
                            item.select_one(".description")
                            or item.select_one(".synopsis")
                        )
                        plot = (
                            plot_text.get_text(strip=True) if plot_text else ""
                        )

                        # Create document
                        doc_id = f"r18:{title.replace(' ', '_')[:50]}"
                        metadata = {
                            "actors": actors,
                            "release_date": date_str,
                            "source": "r18.com",
                            "url": product_url,
                        }
                        content = (
                            f"Title: {title}\n"
                            f"Actors: {actors}\n"
                            f"Date: {date_str}\n"
                            f"Plot: {plot}"
                        )

                        docs.append(Document(doc_id, title, metadata, content))
                        time.sleep(0.5)  # rate limit

                    except Exception as e:
                        print(f"  Error parsing item: {e}")
                        continue

                if docs:
                    break  # Stop if we got results

            except Exception as e:
                print(f"  Failed ({type(e).__name__}): {e}")
                continue

        # Fallback: demo data if scraping failed
        if not docs:
            print("\n⚠️  Scraping unavailable. Loading demo data instead...")
            docs = self._get_demo_data()

        return docs

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
        """Scrape from all sources."""
        all_docs = []
        print("Scraping r18.com...")
        all_docs.extend(self.scrape_r18())
        print(f"Found {len(all_docs)} documents total")
        return all_docs


async def ingest_jav_data() -> int:
    """One-time ingest: scrape all sources and index into FAISS."""
    scraper = JAVScraper()
    docs = scraper.scrape_all()
    if docs:
        rag_store.add_documents(docs)
        print(f"Indexed {len(docs)} documents")
    return len(docs)
