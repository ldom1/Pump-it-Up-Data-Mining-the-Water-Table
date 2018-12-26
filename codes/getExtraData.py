#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:20:57 2018

@author: louisgiron
"""
from bs4 import BeautifulSoup as soup
import requests
import re
from user_agent import generate_user_agent
import pandas as pd


def get_code(url):
        """Return the code html"""
        # Define the user agent
        headers = {'User-Agent': generate_user_agent(device_type="desktop",
                                                     os=('mac', 'linux'))}
        # Open the url file and get the html code of the page
        req = requests.Session()
        req = requests.get(url, headers=headers)
        return soup(req.text, "lxml")


url = 'https://fr.climate-data.org/afrique/tanzanie-132/'

code = get_code(url)

regions_list = [y.get_text().lower().replace(' ', '-').replace('/', '-').replace('&-', '')
                for y in code('ul')[5]('li')]
code_regions = [y.a['href'].replace('/region/', '').replace('/', '')
                for y in code('ul')[5]('li')]

url_regions = [url + str(i) + '-' + str(j) + '/'
               for i, j in zip(regions_list, code_regions)]

# Get all the pages
url_all_regions = []

for regions in url_regions:
    code = get_code(regions)
    try:
        all_pages = [y.get_text() for y in
                     code.findAll('div', {'class': 'pagination'})[0]('a')]
    except IndexError:
        all_pages = ['empty']

    for pages in all_pages:
        try:
            url_all_regions.append(regions + '?page=' + str(int(pages)))
        except ValueError:
            url_all_regions.append(regions)


def get_label_climat_region(regions_list, url_regions):

    # Input data
    df = pd.DataFrame()
    region = []
    label_climat = []

    for i in range(len(regions_list)):

        region.append(regions_list[i].replace('-', ' '))

        # Extract from web
        code = get_code(url_regions[i])

        # Get the climate label
        temp = code.findAll('div',
                            {'itemprop': 'text'})[0].get_text().split('\n')[1]

        # List label clim
        list_label = ['Am', 'Aw', 'BSh', 'Csb']

        for lab in list_label:
            if lab in temp:
                label_climat.append(lab)
            else:
                pass

    # Aggregation
    df['region'] = region
    df['label_climat_reg'] = label_climat

    return df


def get_informations(url_all_regions):

    # Input data
    df = pd.DataFrame()
    name = []
    infos = []
    label_climat = []
    tempe_moy = []
    pluvio = []

    for regions in url_all_regions:

        # Extract from web
        code = get_code(regions)

        # Input
        containers = code.findAll('div', {'class': 'data'})

        # Get infos
        for content in containers:
            temp = content('span')
            try:
                name.append(temp[0].get_text())
            except:
                name.append('')
            try:
                infos.append(temp[1].get_text())
                label_climat.append(temp[1].get_text().split(' ')[-1])
            except:
                infos.append('')
            try:
                tempe_moy.append(float(re.findall('\d+', temp[2].get_text())[0]
                                 + '.'
                                 + re.findall('\d+', temp[2].get_text())[1]))
            except:
                tempe_moy.append(0)
            try:
                pluvio.append(int(re.findall('\d+', temp[3].get_text())[0]))
            except:
                pluvio.append('')

    df['name'] = name
    df['infos'] = infos
    df['label_climat_wpt'] = label_climat
    df['temperature'] = tempe_moy
    df['pluviometrie'] = pluvio

    df = df.drop_duplicates(subset=['name'])

    return df
