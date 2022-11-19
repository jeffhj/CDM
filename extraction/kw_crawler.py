# web crawling
import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# file processing 
from itertools import chain
import re
import pandas as pd

# nlp
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('en')
stopwords = nltk.corpus.stopwords.words('english')
to_ignore = set(chain(stopwords, string.punctuation))

# time
import time

# Use Google to get candidate urls
def googleSearch(query):
    g_clean = [ ]
    # search query
    url = 'https://www.google.com/search?client=ubuntu&channel=fs&q={}&ie=utf-8&oe=utf-8'.format(query)
    try:
        html = requests.get(url)
        if html.status_code==200:
            # sparse the html and get all urls
            soup = BeautifulSoup(html.text, 'lxml')
            a = soup.find_all('a')
            for i in a:
                k = i.get('href')
                try:
                    m = re.search("(?P<url>https?://[^\s]+)", k)
                    n = m.group(0)
                    rul = n.split('&')[0]
                    domain = urlparse(rul)
                    if(re.search('google.com', domain.netloc)):
                        continue
                    else:
                        if ('.jpg' not in rul) and ('.gif' not in rul) and ('.png' not in rul) and ('.jpeg' not in rul):
                            g_clean.append(rul)
                except:
                    continue
    except Exception as ex:
        print(str(ex))
    finally:
        # print (g_clean)
        return g_clean

def wiki_search(kw, reg_url):
    # visit wikipedia page and get full text
    downloaded = trafilatura.fetch_url(reg_url)
    wiki_text = trafilatura.extract(downloaded)

    # process the text
    wiki_sent = []
    for para in wiki_text.split('\n'):
        for phrase in sent_tokenize(para):
            # remove annotations
            phrase = re.sub("[\[].*?[\]]", "", phrase)
            phrase = ' '.join(phrase.split())

            # make sure the sentence is between 5 to 50 words (not too short or too long)
            phrase_len = len(phrase.split())
            if (phrase_len > 5 and phrase_len < 50 and kw in phrase.lower()):
                wiki_sent.append(phrase)

    # return plain text, splitted sentences containing the keyword
    return wiki_text, wiki_sent

def general_search(kw, reg_url):
    # visit the website and get full text
    downloaded = trafilatura.fetch_url(reg_url)
    text = trafilatura.extract(downloaded)

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # process the text
    general_sent = []
    for para in text.split('\n'):
        for sent in sent_tokenize(para):
            # make sure the sentence is between 5 to 50 words (not too short or too long)
            sent_len = len(sent.split())
            if (sent_len > 5 and sent_len < 50 and kw in sent.lower()):
                general_sent.append(sent)

    # return plain text, splitted sentences containing the keyword
    return text, general_sent


if __name__ == "__main__":
    # load candidate keywords from the CS keyword list
    data=pd.read_csv('Keywords-Springer-83K-20210405.csv')
    kw_list = []
    for kw in data['keyword']:
        kw_list.append(kw)

    # get start and end indices for keyword search
    # s_index = int(input('start index: '))
    # s_end = (s_index // 10000 + 1)*10000

    # for kw in kw_list[s_index:s_end]:
    for kw in kw_list:
        # print(kw)
        # avoid error when creating new file
        if '/' in kw:
            continue

        # start time counting
        time_start = time.time()

        # get the first 20 urls from Google search
        Google_url = googleSearch(kw)[:20]

        # start crawling
        content_result = []
        with open('full_text/{}.txt'.format(kw), 'w+') as f:
            for url in Google_url:
                if "wikipedia" in url:
                    if "%" in url:
                        continue
                    try:
                        text, wiki_result = wiki_search(kw, url)
                        for result in wiki_result:
                            content_result.append(result)
                        # save the current url and plain text to the directory ./full_text/
                        f.write('<url>'+url+'\n')
                        f.write(text+'\n')
                    except:
                        pass
                else:
                    try:
                        text, g_search = general_search(kw, url)
                        for result in g_search:
                            content_result.append(result)
                        # save the current url and plain text to the directory ./full_text/
                        f.write('<url>'+url+'\n')
                        f.write(text+'\n')
                    except:
                        pass

        # save candidate sentences to the directory ./has_kw_sent/
        with open('has_kw_sent/{}.txt'.format(kw), 'w+') as f:
            for s in content_result:
                f.write(s+'\n')
        
        # print the time interval for searching the current keyword 
        print("{} -- {} s".format(kw, time.time() - time_start))
