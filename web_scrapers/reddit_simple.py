# Author: Randa Yoga Saputra
# reddit Scraper
from requests import get
from bs4 import BeautifulSoup
from os import listdir
from functools import reduce
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
URL = "https://old.reddit.com/r/CelebrityTongues/top/?sort=top&t=all"
URL = "https://old.reddit.com/r/tongue/top/?sort=top&t=all"
LIM = 2
exts = [".jpg",".png", ".jpeg"]
master= []
errCount, errMax = 0, 3

# Part 1: grab image URLS
for i in range(LIM):
    try: 
        soup = BeautifulSoup(get(URL, headers=headers).text , 'html.parser')
        print(soup.prettify())
        try:
            print(soup.find_all("div",{"class","thing"}))
            print("spans: ", soup.find_all("span"))
            print("nav: ", soup.select(".next-button"))
            print("nav: ", soup.find("div",{"class":"nav-buttons"}))
            print("next: ", soup.find("span",{"class":"next-button"}))
            URL = soup.find("span",{"class":"next-button"}).find("a").get("href")
        except: print("next failed")
        things = [i.get("data-url") for i in soup.find_all("div",{"class","thing"})]
        for thing in things:
            if reduce(lambda a,b: a or b,[thing.find(i) > -1 for i in exts]):
                master.append(thing)
    except:
        print(f"error: {errCount}")
        if errCount >= errMax:
            break
        errCount += 1
    
    print(URL)

print(master)
print(len(master))

# part 2: grab images
for src in master:
    fname = f"data_reddit_simple_tongue/img_tongue_{len(listdir('data_reddit_simple'))}.png"
    with open(fname, "wb") as out:
        out.write(get(src, headers=headers).content)
    print(f"created image {fname}.")