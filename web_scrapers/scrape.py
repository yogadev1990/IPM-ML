# Author: Randa Yoga Saputra
# Shutter Stock Scraper
from requests import get
from bs4 import BeautifulSoup
from os import listdir
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
gen_url = lambda i: f"https://www.shutterstock.com/search/sticking+tongue+out?page={i}"
start_index = 17
n = 10

for i in range(start_index, start_index + n):
    soup = BeautifulSoup(get(gen_url(i), headers=headers).text , 'html.parser')
    srcs = [i.get("src") for i in soup.find_all("img") if i.get("src")]
    for src in srcs:
        fname = f"data/img_{len(listdir('data'))}.png"
        with open(fname, "wb") as out:
            out.write(get(src, headers=headers).content)
        print(f"created image {fname}.")
    print(len(srcs))