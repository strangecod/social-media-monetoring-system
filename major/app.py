from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

### load selenium driver
driver = webdriver.Chrome("C:/Users/DEVESH TRIPATHI/major-project/chromedriver.exe")
driver.get('https://linkedin.com/')
sleep(1)

### get username and password input boxes path
username = driver.find_element("xpath", '//*[@id="session_key"]')
# password = driver.find_element_by_xpath('//*[@id="session_password"]')
password = driver.find_element("xpath", '//*[@id="session_password"]')

### input the email id and password
username.send_keys("devehtripathi2020@gmail.com")
password.send_keys("Devesh@123456")

### click the login button
login_btn = driver.find_element("xpath", "//button[@class='sign-in-form__submit-button']")      

sleep(1)
login_btn.click()

sleep(10)