from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import os
import requests
import pandas as pd
import numpy as np
from time import sleep
from random import randint
import csv 

'''
The commented part is my attempt to iterate through the pages in the website, I have not finished this part
'''
#############################################

# pages= np.arrange(500,1600,1)

# for page in pages:

#     page = requests.get("https://www.wgbh.org/news/local-news?00000160-e20c-d0a5-a176-fa6f2d0d0000-page=" + str(page))

#     print('page')

#     soup = BeautifulSoup(page.text, 'html.parser')

#     links = []

#     for link in soup.findAll('a'):
#         links.append(link.get('href'))
#     print(links)

############################################
'''
This code gets the links for every article on a page, then for each article, the paragraphs and headlines are stored 

'''
# Opening the archive page that contains links to the articles that were published on 01/01/2014
req = Request("https://www.wgbh.org/news/local-news?00000160-e20c-d0a5-a176-fa6f2d0d0000-page=500")
html_page = urlopen(req)

# Parsing the HTML
soup = BeautifulSoup(html_page, "lxml")

# Getting all the links on the page
links = []
for link in soup.findAll('a'):
    links.append(link.get('href'))

# Only getting the links that contain the date 
links2=[]
for link in links:
    if link != None:
        if "2018" in link:
            links2.append(link)

# We discard the first three two and the last three links
links_needed= links2[2:-3]
print(links_needed)
print(len(links_needed))

# Function to remove html tags 

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text))


# For each link we get the paragraph 
paragraph = []
title = []

for i in links_needed:
   req = Request(i)
   html_page = urlopen(req)
 
   soup = BeautifulSoup(html_page, "lxml")

   titles = soup.find_all('h1',class_="ArticlePage-headline")
   title.append(titles)

   paragraphs = soup.find_all('p', class_="")
   paragraph.append(paragraphs)


# Removing HTML tags from each paragraph 
clean_paragraph = []
for i in paragraph:
    clean_paragraph.append(remove_html_tags(i))

print(clean_paragraph)

# Removing HTML tags from each title
clean_title = []
for i in title:
    clean_title.append(remove_html_tags(i))

print(clean_title)

# Creating a data frame 
article_data = pd.DataFrame({
    'headlines' : clean_title,
    'Content' : clean_paragraph 
})

print(article_data)

article_data.to_csv('wgbh_data.csv')