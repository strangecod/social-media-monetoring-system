import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

with open("linkdin.html",'r',encoding='utf-8') as f:
    page = f.read()

soup=BeautifulSoup(page,'lxml')

print(soup)