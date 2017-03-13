# -*- coding: utf-8 -*-
# python 2.7

import requests
from scrapy import Selector
import json
import os

search_list=['popping','hiphop','bboying']

for SEARCH_WORD in search_list:

    os.makedirs("./"+SEARCH_WORD)

    headers = {
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36',
    }

    urls = []
    i=0
    for idx in range(1500):
        params = (
            ('async', '_id:rg_s,_pms:s'),
            ('ei', 'HrjEWOqhCcWZ8gWh16oI'),
            ('yv', '2'),
            ('q', unicode(SEARCH_WORD, 'utf-8')),
            ('start', str(idx*100)),
            ('asearch', 'ichunk'),
            ('tbm', 'isch'),
            ('vet', '10ahUKEwjqqKiT-8_SAhXFjLwKHaGrCgEQuT0IGCgB.HrjEWOqhCcWZ8gWh16oI.i'),
            ('ved', '0ahUKEwjqqKiT-8_SAhXFjLwKHaGrCgEQuT0IGCgB'),
            ('ijn', str(idx)),
        )
        resp = requests.get('https://www.google.com/search', headers=headers, params=params)
        j = json.loads(resp.text)

        sel = Selector(text=j[1][1])
        img_metas = sel.xpath("//div[contains(@class, 'rg_di')]//div[@class='rg_meta']/text()").extract()
        if not img_metas:
            break
        for m in img_metas:
            j = json.loads(m)
            print j['ou']  # url
            try:
                resp = requests.get(j['ou'])
                f = open("./"+SEARCH_WORD+"/"+SEARCH_WORD+"_"+str(i)+".png", 'wb')
                f.write(resp.content)
                urls.append(j['ou'])
            except:
                continue
            i+=1

