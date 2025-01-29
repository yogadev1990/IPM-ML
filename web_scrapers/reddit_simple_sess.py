# Author: Randa Yoga Saputra
# reddit Scraper
from requests import get, Session
from bs4 import BeautifulSoup
from os import listdir
from functools import reduce

session = Session()

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
URL = "https://old.reddit.com/r/tongue/top/?sort=top&t=all"
LIM = 2
exts = [".jpg",".png", ".jpeg"]
master= []
errCount, errMax = 0, 3


oheaders = {
"authority": "old.reddit.com",
"method": "POST",
"path": "/over18?dest=https%3A%2F%2Fold.reddit.com%2Fr%2Ftongue%2Ftop%2F%3Fsort%3Dtop%26t%3Dall",
"scheme": "https",
"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
"accept-encoding": "gzip, deflate, br",
"accept-language": "en-US,en;q=0.9",
"cache-control": "max-age=0",
"content-length": "10",
"content-type": "application/x-www-form-urlencoded",
"cookie": "edgebucket=9MLSqX9pWFUyUgR2Li; reddaid=ROU4L46S72OLYUAA; __gads=ID=737ab13d14a3fe5b:T=1590965009:S=ALNI_MYkoxORBp8aX5uE2NklpSMQk-0nWQ; loid=00000000006mt77f5s.2.1590964992169.Z0FBQUFBQmZCN1ZucG1VcHVIRzU0UEY0SzJwTWptbHIyMlR3cjB2RERTV2dMVVFwVWFvNl9aLXEwUTZyYlozLWR5d1g3a0UtQ2cwTGxKZW1xUmRqVU0xcldGcjVGaTB5VjJSdzZ6ZWtrSmxFcHltUTkzRGhtSVkzdkVqZjJIUEhMYy1jR21VNngteUQ; csv=1; d2_token=3.1a38cf25899b0b0bb73c962c8c06c785266847ecf5c534e736247430148ebdf9.eyJhY2Nlc3NUb2tlbiI6Ii1oaUF1R0FJaWNrSkNZZW4wQ0V3YlR3RnFjcjgiLCJleHBpcmVzIjoiMjAyMC0wNy0yM1QxNDo0NToyMS4wMDBaIiwibG9nZ2VkT3V0Ijp0cnVlLCJzY29wZXMiOlsiKiIsImVtYWlsIl19; token_v2=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2MTc2NTMzNDMsInN1YiI6Ii1UQm5LTUdYdk9LQks0SWsybFliM3J1Ry1XVlUiLCJsb2dnZWRJbiI6ZmFsc2UsInNjb3BlcyI6WyIqIiwiZW1haWwiXX0.jpoAMclZNagRGtqTexiGSCmUWrTsCftHkJgNQX8zl5E; __aaxsc=2; pc=tu; over18=1; recent_srs=t5_2u80v%2Ct5_2ryt0%2Ct5_2qhj4; session_tracker=UtqiwbipRa2sL5cWyP.0.1618471430064.Z0FBQUFBQmdkLW9HYTh4Rm5fQ3I3dDhqNm1iSzNWdTI2aFo0Skg2bWNWd0tzSDZiT1lxbC1vUGJGU1FPUzh5ejJCd3RTeHdsXzhXUU8tU1Q3cEJHRkFOS3dDQ3VCdzlRNkJvbUJTLV9MMFQwUG5YSTYwY1ZXbGFFWklJSHlrcTByQ001UGJsSXJ6cGs; aasd=2%7C1618471402622",
"origin": "https://old.reddit.com",
"referer": "https://old.reddit.com/",
"sec-fetch-dest": "document",
"sec-fetch-mode": "navigate",
"sec-fetch-site": "same-origin",
"sec-fetch-user": "?1",
"upgrade-insecure-requests": "1",
"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.76"
}

xheaders = {
"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
"accept-encoding": "gzip, deflate, br",
"accept-language": "en-US,en;q=0.9",
"cache-control": "max-age=0",
"content-length": "10",
"content-type": "application/x-www-form-urlencoded",
"cookie": "edgebucket=9MLSqX9pWFUyUgR2Li; reddaid=ROU4L46S72OLYUAA; __gads=ID=737ab13d14a3fe5b:T=1590965009:S=ALNI_MYkoxORBp8aX5uE2NklpSMQk-0nWQ; loid=00000000006mt77f5s.2.1590964992169.Z0FBQUFBQmZCN1ZucG1VcHVIRzU0UEY0SzJwTWptbHIyMlR3cjB2RERTV2dMVVFwVWFvNl9aLXEwUTZyYlozLWR5d1g3a0UtQ2cwTGxKZW1xUmRqVU0xcldGcjVGaTB5VjJSdzZ6ZWtrSmxFcHltUTkzRGhtSVkzdkVqZjJIUEhMYy1jR21VNngteUQ; csv=1; d2_token=3.1a38cf25899b0b0bb73c962c8c06c785266847ecf5c534e736247430148ebdf9.eyJhY2Nlc3NUb2tlbiI6Ii1oaUF1R0FJaWNrSkNZZW4wQ0V3YlR3RnFjcjgiLCJleHBpcmVzIjoiMjAyMC0wNy0yM1QxNDo0NToyMS4wMDBaIiwibG9nZ2VkT3V0Ijp0cnVlLCJzY29wZXMiOlsiKiIsImVtYWlsIl19; token_v2=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2MTc2NTMzNDMsInN1YiI6Ii1UQm5LTUdYdk9LQks0SWsybFliM3J1Ry1XVlUiLCJsb2dnZWRJbiI6ZmFsc2UsInNjb3BlcyI6WyIqIiwiZW1haWwiXX0.jpoAMclZNagRGtqTexiGSCmUWrTsCftHkJgNQX8zl5E; __aaxsc=2; pc=tu; over18=1; recent_srs=t5_2u80v%2Ct5_2ryt0%2Ct5_2qhj4; session_tracker=UtqiwbipRa2sL5cWyP.0.1618471430064.Z0FBQUFBQmdkLW9HYTh4Rm5fQ3I3dDhqNm1iSzNWdTI2aFo0Skg2bWNWd0tzSDZiT1lxbC1vUGJGU1FPUzh5ejJCd3RTeHdsXzhXUU8tU1Q3cEJHRkFOS3dDQ3VCdzlRNkJvbUJTLV9MMFQwUG5YSTYwY1ZXbGFFWklJSHlrcTByQ001UGJsSXJ6cGs; aasd=2%7C1618471402622",
"origin": "https://old.reddit.com",
"referer": "https://old.reddit.com/",
"sec-fetch-dest": "document",
"sec-fetch-mode": "navigate",
"sec-fetch-site": "same-origin",
"sec-fetch-user": "?1",
"upgrade-insecure-requests": "1",
"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.76"
}
resp = session.post(
    #"https://old.reddit.com/over18?dest=https%3A%2F%2Fold.reddit.com%2Fr%2Ftongue%2Ftop%2F%3Fsort%3Dtop%26t%3Dall",
    "https://old.reddit.com/r/tongue/top/?sort=top&t=all",
    headers=xheaders,
    data={"over18".encode(),"yes".encode()}
)
print("posted")
# Part 1: grab image URLS
for i in range(LIM):
    try: 
        soup = BeautifulSoup(session.get("https://old.reddit.com/r/tongue/top/?sort=top&t=all", headers=headers).text , 'html.parser')
        #print(soup.prettify())
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