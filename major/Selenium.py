from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

### load selenium driver
driver = webdriver.Chrome("C:/Users/DEVESH TRIPATHI/major-project/chromedriver.exe")
driver.get('https://www.linkedin.com/feed/')
sleep(1)

### click the login button
sign_in = driver.find_element("xpath", "/html/body/div[1]/main/p[1]/a")      

sleep(1)
sign_in.click()

### get username and password input boxes path
username = driver.find_element("xpath", '//*[@id="username"]')
# password = driver.find_element_by_xpath('//*[@id="session_password"]')
password = driver.find_element("xpath", '//*[@id="password"]')

### input the email id and password
username.send_keys("ananytewari@gmail.com")
password.send_keys("yash@2002")

### click the login button
login_btn = driver.find_element("xpath", '//*[@id="organic-div"]/form/div[3]/button')      

sleep(1)
login_btn.click()

sleep(5)

old_height = driver.execute_script('return document.body.scrollHeight')

counter = 1
while True:


    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    sleep(7)

    new_height = driver.execute_script('return document.body.scrollHeight')

    print(counter)
    counter += 1
    print(old_height)
    print(new_height)

    if new_height == old_height:
        break

    if new_height > 700000:
        break   

    old_height = new_height



html = driver.page_source

with open('linkdin_ananya4.html','w',encoding='utf-8') as f:
    f.write(html)