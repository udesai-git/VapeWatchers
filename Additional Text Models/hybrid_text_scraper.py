import scrapy
import boto3
import json
import os
from bs4 import BeautifulSoup
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urlparse


class HybridTextScraper(CrawlSpider):
    name = 'hybrid_text_scraper'

    def __init__(self, start_url=None, bucket_name=None, *args, **kwargs):
        super(HybridTextScraper, self).__init__(*args, **kwargs)

        if not start_url or not bucket_name:
            raise ValueError("Usage: scrapy crawl hybrid_text_scraper -a start_url='https://example.com' -a bucket_name='your-bucket-name'")

        self.start_urls = [start_url]
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.website_name = urlparse(start_url).netloc.replace("www.", "")
        self.output_file = f"{self.website_name}_scraped.json"
        self.collected_data = []

    rules = (
        Rule(LinkExtractor(allow=r'(collections|products|product|partners|blog|news|about|info)'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')

        text_elements = soup.find_all(['p', 'span', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        extracted_text = "\n".join([el.get_text(strip=True) for el in text_elements if el.get_text(strip=True)])

        item = {
            'url': response.url,
            'text': extracted_text
        }

        self.collected_data.append(item)

    def closed(self, reason):
        # Save data to local JSON file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, indent=2)

        # Upload to S3
        self.s3_client.upload_file(
            Filename=self.output_file,
            Bucket=self.bucket_name,
            Key=f"scraped_text/{self.output_file}"
        )
        print(f"✅ Uploaded {self.output_file} to S3 bucket: {self.bucket_name}/scraped_text/")
