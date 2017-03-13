# Python 3.6
# Scrapy

import scrapy
import requests
import os

SEARCH_WORD = 'food anime'
base_url = 'http://www.bing.com/images/search?q='+ SEARCH_WORD +'&FORM=HDRSC2'

try:
    os.makedirs("./"+SEARCH_WORD+"_bing")
except:
    pass


class ImageSpider(scrapy.Spider):
    name = "images"
    j = 0

    def start_requests(self):
        urls = [
            base_url + '&first=1&cw=1462&ch=824'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        try:
            resp = response.xpath('//a[@class="thumb"]/@href').extract()
            for i in resp:
                img = requests.get(i)
                f = open("./" + SEARCH_WORD + "_bing" + "/" + SEARCH_WORD + "_" + str(self.j)+".png", 'wb')
                f.write(img.content)
                self.j += 1
                f.close()

            new = response.xpath("//a[@class='nav_page_next']/@href").extract_first()
            new_url = base_url+new

        except:
            new = response.xpath("//a[@class='nav_page_next']/@href").extract_first()
            new_url = base_url + new

        yield scrapy.Request(url=new_url, callback=self.parse)
