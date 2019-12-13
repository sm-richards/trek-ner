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
rootdir = './fictxt'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file != '.DS_Store':
            lines = open(os.path.join(subdir, file), 'r').readlines()
            fic = {}
            fic1 = {}
            fic2 = {}
            fic['text'] = ' '.join(lines[:30])
            fic1['text'] = ' '.join(lines[30:60])
            fic2['text'] = ' '.join(lines[60:90])
            all.extend([fic, fic1, fic2])
print(all)

with open('output1.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in all[:30]:
        json.dump(entry, outfile)
        outfile.write('\n')

with open('output2.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in all[30:60]:
        json.dump(entry, outfile)
        outfile.write('\n')

with open('output3.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in all[60:90]:
        json.dump(entry, outfile)
        outfile.write('\n')
