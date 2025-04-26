import os
import time
import requests
import sys
import uuid
import re
import boto3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from concurrent.futures import ThreadPoolExecutor

class HybridImageScraper:
    def __init__(self, *, base_url, aws_access_key_id, aws_secret_access_key, aws_session_token, max_pages=50, max_threads=10):
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_threads = max_threads
        self.base_domain = urlparse(base_url).netloc
        self.website_name = self.base_domain.replace("www.", "").split(":")[0].split(".")[0]

        # AWS S3 settings
        self.bucket_name = 'vapewatchers-2025'
        self.s3_prefix = f"MarketingImagesTest/{self.website_name}/"

        s3_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        self.s3_client = s3_session.client('s3')

        self.scrapy_failed_urls = set()
        self.all_image_urls = set()

    class ScrapyImageSpider(scrapy.Spider):
        name = "hybrid_image_spider"

        def __init__(self, url=None, main_scraper=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.main_scraper = main_scraper
            self.start_urls = [url]
            self.allowed_domains = [urlparse(url).netloc]

        def parse(self, response):
            img_tags = response.css('img')
            for img in img_tags:
                for attr in ['src', 'data-src', 'data-original']:
                    img_url = img.attrib.get(attr)
                    if img_url:
                        img_url = urljoin(response.url, img_url)
                        if self.is_valid_image_url(img_url):
                            self.main_scraper.all_image_urls.add(img_url)

                srcset = img.attrib.get('srcset')
                if srcset:
                    for src_item in srcset.split(','):
                        src_parts = src_item.strip().split(' ')
                        if src_parts:
                            img_url = urljoin(response.url, src_parts[0])
                            if self.is_valid_image_url(img_url):
                                self.main_scraper.all_image_urls.add(img_url)

            for link in response.css('a::attr(href)').getall():
                full_link = urljoin(response.url, link)
                if full_link.startswith(self.start_urls[0]):
                    yield scrapy.Request(full_link, callback=self.parse)

        def is_valid_image_url(self, url):
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
            url_lower = url.lower()
            skip_keywords = ['icon', 'pixel', 'tiny', 'favicon', 'logo', 'spacer']
            return (any(ext in url_lower for ext in image_extensions) and
                    not any(kw in url_lower for kw in skip_keywords))

    def run_scrapy_spider(self):
        print("\nStarting Scrapy spider...")
        process = CrawlerProcess(get_project_settings())
        process.crawl(self.ScrapyImageSpider, url=self.base_url, main_scraper=self)
        process.start()
        print(f"Scrapy found {len(self.all_image_urls)} images")

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def is_same_domain(self, url):
        return urlparse(url).netloc == self.base_domain

    def is_valid_image_url(self, url):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        url_lower = url.lower()
        skip_keywords = ['icon', 'pixel', 'tiny', 'favicon', 'logo', 'spacer']
        return ((any(ext in url_lower for ext in image_extensions) or 
                 'image' in url_lower or 'img' in url_lower) and
                not any(kw in url_lower for kw in skip_keywords))

    def get_soup(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_images_with_bs(self, url, soup):
        if not soup:
            return

        for img in soup.find_all('img'):
            for attr in ['src', 'data-src', 'data-original']:
                img_url = img.get(attr)
                if img_url:
                    img_url = urljoin(url, img_url)
                    if self.is_valid_url(img_url) and self.is_valid_image_url(img_url):
                        self.all_image_urls.add(img_url)

            srcset = img.get('srcset')
            if srcset:
                for src_item in srcset.split(','):
                    src_parts = src_item.strip().split(' ')
                    if src_parts:
                        img_url = urljoin(url, src_parts[0])
                        if self.is_valid_url(img_url) and self.is_valid_image_url(img_url):
                            self.all_image_urls.add(img_url)

    def crawl_with_bs(self):
        print("\nStarting BeautifulSoup backup crawl...")
        visited = set()
        to_visit = {self.base_url}
        page_count = 0

        while to_visit and page_count < self.max_pages:
            url = to_visit.pop()
            if url in visited:
                continue

            visited.add(url)
            print(f"BS Crawling [{page_count + 1}/{self.max_pages}]: {url}")

            soup = self.get_soup(url)
            if soup:
                self.extract_images_with_bs(url, soup)

                for a in soup.find_all('a', href=True):
                    full_url = urljoin(url, a['href'])
                    if self.is_valid_url(full_url) and self.is_same_domain(full_url):
                        to_visit.add(full_url)

            page_count += 1
            time.sleep(1)

    def download_image(self, img_url):
        try:
            if img_url.lower().endswith('.svg'):
                return False

            ext = os.path.splitext(img_url)[1].lower()
            ext = ext if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'] else '.jpg'
            img_name = f"image_{uuid.uuid4()}{ext}"
            img_name = re.sub(r'[\\/*?:"<>|]', "_", img_name)
            s3_key = f"{self.s3_prefix}{img_name}"

            response = requests.get(img_url, timeout=10, stream=True)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))

                if img.width < 100 or img.height < 100:
                    return False

                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)

                self.s3_client.upload_fileobj(buffer, self.bucket_name, s3_key, ExtraArgs={'ContentType': 'image/jpeg'})
                print(f"✅ Uploaded to S3: {s3_key}")
                return True
        except Exception as e:
            print(f"❌ Error uploading {img_url} to S3: {e}")
            return False

    def download_all_images(self):
        print(f"\nStarting download of {len(self.all_image_urls)} images...")
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            results = list(executor.map(self.download_image, list(self.all_image_urls)))
        successful = sum(results)
        print(f"\nUpload complete: {successful}/{len(self.all_image_urls)} images saved to S3")
        return successful

    def run(self):
        self.run_scrapy_spider()
        self.crawl_with_bs()
        if self.all_image_urls:
            self.download_all_images()
        else:
            print("No images found to upload.")

# def scraper_main(target_url,aws_access_key_id,aws_secret_access_key,aws_session_token, max_pages=50):
#     scraper = HybridImageScraper(target_url,aws_access_key_id,aws_secret_access_key,aws_session_token,max_pages)
#     scraper.run()
    
# def scraper_main(target_url, aws_access_key_id, aws_secret_access_key, aws_session_token, max_pages=50):
    
def scraper_main(*, target_url, aws_access_key_id, aws_secret_access_key, aws_session_token, max_pages=50):
    print("Key type:", type(aws_access_key_id))
    print("Secret type:", type(aws_secret_access_key))
    scraper = HybridImageScraper(
        base_url=target_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        max_pages=max_pages
    )
    scraper.run()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python HybridScraperAWS.py <url> <access_key> <secret_key> <session_token> [max_pages]")
        sys.exit(1)

    target_url = sys.argv[1]
    aws_access_key = sys.argv[2]
    aws_secret_key = sys.argv[3]
    aws_token = sys.argv[4]
    max_pages = int(sys.argv[5]) if len(sys.argv) > 5 else 50

    scraper_main(
        target_url=target_url,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_token,
        max_pages=max_pages
    )