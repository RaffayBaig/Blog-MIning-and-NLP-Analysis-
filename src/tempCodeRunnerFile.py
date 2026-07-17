import requests
import pandas as pd

from bs4 import BeautifulSoup
from googlesearch import search

def get_blog_urls(keyword, num_results=10):

    query = f"{keyword} blog"

    urls = []

    try:

        for url in search(query, num_results=num_results):

            urls.append(url)

    except Exception as e:

        print("Error while searching:", e)

    return urls

def scrape_blog(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(response.text, "lxml")

        # TITLE
        title = soup.title.text.strip() if soup.title else "No Title"

        # AUTHOR
        author = "Unknown"

        author_tag = soup.find("meta", attrs={"name": "author"})

        if author_tag and author_tag.get("content"):

            author = author_tag.get("content")

        # DATE
        date = "Unknown"

        date_tag = soup.find("time")

        if date_tag:

            date = date_tag.get_text(strip=True)

        # MAIN CONTENT
        paragraphs = soup.find_all("p")

        content = " ".join(
            [p.get_text(strip=True) for p in paragraphs]
        )

        return {
            "url": url,
            "title": title,
            "author": author,
            "date": date,
            "content": content
        }

    except Exception as e:

        print(f"Error scraping {url}: {e}")

        return None