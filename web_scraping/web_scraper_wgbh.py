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
# --------------------------------------------

# Function to remove html tags 
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text))

# Used to iterate through different pages on the website 
pages= np.arange(1500,1656,1)

# To store data after iteration
clean_title = []
clean_paragraph = []
clean_tag = []

# Iterate through the pages specified previously
for page in pages:

    print(page)
    pagereq = Request("https://www.wgbh.org/news/local-news?00000160-e20c-d0a5-a176-fa6f2d0d0000-page=" + str(page))
    html_page=urlopen(pagereq)

    # Parsing the HTML
    soup = BeautifulSoup(html_page, "lxml")

    # Will vary amount of waiting time between request to mimick human activity 
    sleep(randint(2,5))

    # Getting all the links on the page
    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))

    # Only getting the links that contain the date 
    links2=[]
    for link in links:
        if link != None:
            if '/post' in link: 
                links2.append(link)

    # Printing them to keep track and make sure 
    links_needed= links2
    print(links_needed)
    print(len(links_needed))


    # For each link we get the paragraph, title and tags 
    paragraph = []
    title = []
    tag= []

    for i in links_needed:

        # Requesting access to particular link
        req = Request(i)
        html_page = urlopen(req)
     
        soup = BeautifulSoup(html_page, "lxml")

        # Appending headlines for each article
        titles = soup.find_all('h1',class_="ArticlePage-headline")
        title.append(titles)
        # Appending content for each article 
        paragraphs = soup.find_all('p', class_="")
        paragraph.append(paragraphs)
        # Appending tags for each article 
        tags = soup.find_all('ul', class_= "ArticlePage-tags-list clrfix")
        tag.append(tags)
    
    # Removing HTML tags from each paragraph 
    for i in paragraph:
        clean_paragraph.append(remove_html_tags(i))

    print(clean_paragraph)

    # Removing HTML tags from each title
    
    for i in title:
        clean_title.append(remove_html_tags(i))

    print(clean_title)

    # Removing HTML tags from each tags

    # for i in tag:
    #     clean_tag.append(remove_html_tags(i))

    # print(clean_tag)


# Creating a data frame 
article_data = pd.DataFrame({
    'Article Headline' : clean_title,
    'Article Content' : clean_paragraph,
    
})

print(article_data)

article_data.to_csv('wgbh_data_2014_2.csv')

