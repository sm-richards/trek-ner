import requests
import time
from bs4 import BeautifulSoup
import os
import json
#import jsonlines
import random
from collections import defaultdict

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

test_pages = open("fanfics.txt", 'r')
urls = []
for line in test_pages:
    urls.append(line.rstrip('\n'))
test_pages.close()
print(urls)
all = []
for i, url in enumerate(urls):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.post(url, headers = headers)
#print( requests.options(test_page) )

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        lines = soup.extract().text
        linelst = lines.split('\n')
        lines = [string.replace(u'\xa0', u' ') for string in linelst]
        begind = index_containing_substring(lines, 'Any kind of feedback is greatly appreciated.')
        if begind == -1:
            begind = index_containing_substring(lines, 'Category:')
        endind = lines.index('Comments')
        file = open('fictxt/'+str(i)+'.txt', 'w+')
        file.write('\n'.join(lines[begind+1:endind-1]))
        file.close()

        fic = {}
        fic['text'] = ' '.join(lines[begind+1:begind+51])
        print(fic)
        all.append(fic)
    else:
        pass
        print(response.text)

with open('output.jsonl', 'w') as outfile:
    for entry in all:
        json.dump(entry, outfile)
        outfile.write('\n')
