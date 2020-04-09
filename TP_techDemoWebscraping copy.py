#webscraping tutorial: https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460 
#another webscraping tutorital: https://www.scrapehero.com/web-scraping-tutorial-for-beginners-part-3-navigating-and-extracting-data/
#exception handling: https://www.pythonforbeginners.com/error-handling/exception-handling-in-python
import urllib.request
import requests
import time
from bs4 import BeautifulSoup

def getTitle(url, user):
    request = urllib.request.Request(url,headers={'User-Agent': user})
    html = urllib.request.urlopen(request).read()
    soup = BeautifulSoup(html,'html.parser')
    headingHTML = soup.find("h1",attrs={'id':'firstHeading', 'class':'firstHeading', 'lang':'en'})
    actualHeading = str(headingHTML)[53:-5]
    if "<i>" in actualHeading:
                actualHeading = actualHeading.replace('<i>','').replace('</i>','')
    return actualHeading

def getLinks(url, user):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    htmlUrls = soup.findAll('a')
    urls = []
    count = 1
    for i in range(len(htmlUrls)):
        toPrint = ""
        try:
            urlEnd = str(htmlUrls[i]['href'])
        except KeyError:
            continue
        if urlEnd.startswith("/wiki/"):
            if "<i>" in urlEnd:
                urlEnd = urlEnd.replace('<i>','').replace('</i>','')
            newUrl = "https://en.wikipedia.org" + urlEnd 
            urls.append(newUrl)
            print(getTitle(newUrl, user))
        count += 1
        if count == 43:
            return urls
        #time.sleep(1)
    return urls

def getUrlFromTitle(title):
    title = title.replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + title

def runner():
    print("\nWelcome to the wikipedia game. You will start at Bird. Try to get to Vitamin.")
    url = "https://en.wikipedia.org/wiki/Bird"
    user = 'Chrome/80.0.3987.149 (Macintosh; Intel Mac OS Catalina 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko)'
    title = "Bird"
    while title != "Vitamin":
        print("\nCopy and paste one of these titles below to continue:\n")
        getLinks(url, user)
        user = input("\nPaste here : ") 
        url = getUrlFromTitle(user)
        title = getTitle(url, user)
    print("\n\n\nYou Won!")

runner()
        