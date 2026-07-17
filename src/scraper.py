import re
import time
import random
import requests
import pandas as pd

from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse
from dateutil import parser as dateparser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64)"
    )
}

MIN_TEXT_LENGTH = 300


# =========================================
# DOMAIN HELPERS
# =========================================

def get_domain(url):

    parsed = urlparse(url)

    return parsed.netloc.replace("www.", "")


BLOCKED_DOMAINS = {

    "youtube.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "amazon.com"
}


def is_blocked(domain):

    return any(
        blocked in domain
        for blocked in BLOCKED_DOMAINS
    )


# =========================================
# SEARCH BLOG URLS
# =========================================

def get_blog_urls(keyword, max_results=30):

    queries = [

        f"{keyword} blog post",
        f"{keyword} tutorial",
        f"{keyword} article"
    ]

    discovered_urls = []

    with DDGS() as ddgs:

        for query in queries:

            try:

                results = ddgs.text(
                    query,
                    max_results=max_results
                )

                for result in results:

                    url = result.get("href")

                    if not url:
                        continue

                    domain = get_domain(url)

                    if is_blocked(domain):
                        continue

                    if url not in discovered_urls:

                        discovered_urls.append(url)

                time.sleep(1)

            except Exception as e:

                print(f"Search Error: {e}")

    return discovered_urls


# =========================================
# SELENIUM DRIVER SETUP
# =========================================

def create_driver():
    """
    Creates a single headless Chrome driver instance,
    reused across all blogs to avoid the overhead of
    launching a new browser per page.
    """

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")

    driver = webdriver.Chrome(options=options)

    return driver


def try_expand_comments(driver):
    """
    Best-effort attempt to trigger lazy-loaded comment
    sections (Disqus, 'load more' buttons, etc.) before
    we grab the page source.
    """

    selectors = [
        "button.show-more-comments",
        ".disqus-thread",
        "#comments",
        "#respond"
    ]

    for selector in selectors:

        try:

            el = driver.find_element(By.CSS_SELECTOR, selector)

            driver.execute_script(
                "arguments[0].scrollIntoView();", el
            )

            time.sleep(1)

        except Exception:

            continue


def fetch_page_selenium(url, driver):
    """
    Loads a page with a real (headless) browser so that
    JS-rendered content -- comment widgets especially --
    is present in the DOM before we parse it.
    """

    try:

        driver.get(url)

        try:

            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

        except Exception:

            pass

        try_expand_comments(driver)

        time.sleep(2)  # buffer for lazy-loaded widgets

        soup = BeautifulSoup(driver.page_source, "lxml")

        return soup

    except Exception:

        return None


# =========================================
# FETCH PAGE (requests, fast path)
# =========================================

def fetch_page(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:

            return None

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        return soup

    except Exception:

        return None


# =========================================
# TITLE EXTRACTION
# =========================================

def extract_title(soup):

    selectors = [

        "h1",
        "h1.entry-title",
        "h1.post-title",
        "article h1"
    ]

    for selector in selectors:

        tag = soup.select_one(selector)

        if tag:

            return tag.get_text(strip=True)

    if soup.title:

        return soup.title.text.strip()

    return "Unknown Title"


# =========================================
# AUTHOR EXTRACTION
# =========================================

def extract_author(soup):

    selectors = [

        "[class*=author]",
        ".author",
        ".byline",
        "[rel=author]"
    ]

    for selector in selectors:

        tag = soup.select_one(selector)

        if tag:

            author = tag.get_text(strip=True)

            if len(author) < 60:

                return author

    meta = soup.find(
        "meta",
        attrs={"name": "author"}
    )

    if meta and meta.get("content"):

        return meta["content"]

    return "Unknown Author"


# =========================================
# DATE EXTRACTION
# =========================================

def extract_date(soup):

    time_tag = soup.find("time")

    if time_tag:

        raw = (
            time_tag.get("datetime")
            or time_tag.get_text(strip=True)
        )

        try:

            parsed = dateparser.parse(raw)

            return parsed.strftime("%Y-%m-%d")

        except Exception:

            return "Unknown Date"

    return "Unknown Date"


# =========================================
# MAIN CONTENT EXTRACTION
# =========================================

def extract_content(soup):

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside"
    ]):

        tag.decompose()

    selectors = [

        "article",
        ".post-content",
        ".entry-content",
        ".article-body",
        "main"
    ]

    for selector in selectors:

        section = soup.select_one(selector)

        if section:

            text = section.get_text(
                separator=" ",
                strip=True
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            )

            if len(text) >= MIN_TEXT_LENGTH:

                return text

    paragraphs = soup.find_all("p")

    text = " ".join(
        p.get_text(strip=True)
        for p in paragraphs
    )

    return text


# =========================================
# COMMENTS EXTRACTION
# =========================================

def extract_comments(soup):

    comments = []

    selectors = [

        ".comment",
        ".comment-body",
        ".comment-content",
        ".wp-comment",
        "#comments .comment-list li",
        "[itemprop='comment']",
        ".disqus-comment"
    ]

    for selector in selectors:

        elements = soup.select(selector)

        for element in elements:

            text = element.get_text(
                strip=True
            )

            if len(text) > 20:

                comments.append(text)

    return comments[:20]


# =========================================
# SCRAPE SINGLE BLOG
# =========================================

def scrape_blog(url, driver=None):

    soup = fetch_page(url)

    if not soup:

        return None

    title = extract_title(soup)

    content = extract_content(soup)

    if len(content) < MIN_TEXT_LENGTH:

        return None

    author = extract_author(soup)

    date = extract_date(soup)

    comments = extract_comments(soup)

    # Fallback: if the fast requests-based fetch found no
    # comments, retry the SAME page with Selenium in case
    # the comment widget is JS-rendered (Disqus, AJAX, etc).
    if len(comments) == 0 and driver is not None:

        selenium_soup = fetch_page_selenium(url, driver)

        if selenium_soup:

            selenium_comments = extract_comments(selenium_soup)

            if len(selenium_comments) > 0:

                comments = selenium_comments

    return {

        "url": url,
        "domain": get_domain(url),
        "title": title,
        "author": author,
        "date": date,
        "content": content,
        "comments": comments,
        "comment_count": len(comments),
        "word_count": len(content.split())
    }


# =========================================
# SCRAPE MULTIPLE BLOGS
# =========================================

def scrape_multiple_blogs(
    keyword,
    num_results=10
):

    urls = get_blog_urls(keyword)

    all_blog_data = []

    domains = set()

    driver = create_driver()

    try:

        for index, url in enumerate(urls):

            if (
                len(all_blog_data)
                >= num_results
                and len(domains) >= 3
            ):

                break

            print(f"\nScraping Blog {index+1}")

            blog = scrape_blog(url, driver)

            if blog:

                all_blog_data.append(blog)

                domains.add(blog["domain"])

                print(
                    f"✓ {blog['title'][:60]}"
                )

                print(
                    f"  Domain: {blog['domain']}"
                )

                print(
                    f"  Comments: "
                    f"{blog['comment_count']}"
                )

            time.sleep(
                random.uniform(1, 2)
            )

    finally:

        driver.quit()

    print(
        f"\n✓ Scraped "
        f"{len(all_blog_data)} blogs"
    )

    print(
        f"✓ Domains Used: "
        f"{len(domains)}"
    )

    return all_blog_data


# =========================================
# SAVE CSV
# =========================================

def save_to_csv(blog_data):

    df = pd.DataFrame(blog_data)

    output_path = "data/raw/blogs.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nData saved to {output_path}"
    )