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
        """Scrape from r18.com (English mirror of DMM)."""
        docs = []
        try:
            # r18.com is more accessible than dmm.co.jp from outside Japan
            url = "https://www.r18.com/common/search/searchlist/index.html"
            params = {"type": "dvd"}

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml")

            # Parse product list (CSS selectors may need adjustment based on current site structure)
            for item in soup.select(".productItem")[:10]:  # limit to first 10 for testing
                try:
                    title_elem = item.select_one(".title a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    product_url = title_elem.get("href", "")

                    # Extract metadata from item
                    actors_text = item.select_one(".performer")
                    actors = actors_text.get_text(strip=True) if actors_text else "Unknown"

                    date_text = item.select_one(".releaseDate")
                    date_str = date_text.get_text(strip=True) if date_text else ""

                    plot_text = item.select_one(".description")
                    plot = plot_text.get_text(strip=True) if plot_text else ""

                    # Create document
                    doc_id = f"r18:{title.replace(' ', '_')[:50]}"
                    metadata = {
                        "actors": actors,
                        "release_date": date_str,
                        "source": "r18.com",
                        "url": product_url,
                    }
                    content = f"Title: {title}\nActors: {actors}\nDate: {date_str}\nPlot: {plot}"

                    docs.append(Document(doc_id, title, metadata, content))

                    time.sleep(0.5)  # rate limit
                except Exception as e:
                    print(f"Error parsing item: {e}")
                    continue

        except Exception as e:
            print(f"r18 scrape failed: {e}")

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
