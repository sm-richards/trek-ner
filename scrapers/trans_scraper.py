import requests
import time
from bs4 import BeautifulSoup
import os

# Series, first ep #, last ep # + 1, min length of episode #
series = [("StarTrek/", 1, 80, 0), ("StarTrek/TAS", 1, 23, 3), ("NextGen/", 101, 278, 0), ("DS9/", 401, 576, 0),\
          ("Voyager/", 101, 723, 0), ("Enterprise/", 1, 99, 2)]
# Just do try/except on nonexistent chapters.

for serie in series[3:]:
    for episode in range(serie[1],serie[2]):
        if len(str(episode)) < serie[3]: # Enterprise naming nonsense. God, the inconsistency
            episode = "0" * (serie[3] - len(str(episode))) + str(episode)

        url = "http://chakoteya.net/" + serie[0] + str(episode) + ".htm" #49.htm
        response = requests.post(url, data="{Test data}")
        #print(response)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            #print(soup)
            #print(soup.find_all('a'))
            time.sleep(.01)

            directory = "./" + serie[0]
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = open("./" + serie[0] + str(episode) + ".txt", 'w+')

            lines = soup.extract().text

            #lines = response.text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
            #print(lines)
            #lines = re.findall(": (.*?)" + re.escape("<br>"), lines)
            #print(lines[:60])#
            file.write("".join(lines[75:-260]))
            file.close()
        else:
            print("Missed: ", url)
    print("Through", serie[0])