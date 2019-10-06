import urllib.request as req
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"

driver = webdriver.Chrome("C:\\WorkSpace//chromedriver.exe")
driver.get(url)
ser = driver.find_element_by_class_name("gLFyf.gsfi")#검색어 입력 
ser.send_keys("스카이림")#검색어 
ser.submit()

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
soup

#####################
params = []
error = []
img_list = soup.find_all("img", class_="rg_ic rg_i")
img_list[1]['data-src']

for i in img_list:
    try:
        params.append(i['data-src'])
    except KeyError:
        continue

params

x = 1
for p in params:
    req.urlretrieve(p,"C:\\WorkSpace\\sample_img_data\\skyrim_"+str(x)+".jpg")
    x += 1

driver.close()
######################
img_list = soup.find_all("img", class_="rg_ic rg_i")