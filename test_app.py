import unittest
import time
import os
import flask
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class FlaskTests(unittest.TestCase):

	
	
		
	def verifySentences(self, input, output):
		self.driver = webdriver.Firefox()
		self.driver.get("http://localhost:5000")
		time.sleep(2)
		elem = self.driver.find_elements(By.XPATH, '/html/body/form/center/input[1]')
		elem[0].send_keys(input)
		time.sleep(2)
		elem = self.driver.find_elements(By.XPATH, '/html/body/form/center/input[2]')
		elem[0].submit()
		time.sleep(2)
		elem = self.driver.find_elements(By.XPATH, '/html/body/center/span/div')
		self.assertEqual(elem[0].text, output)
		time.sleep(2)
		self.driver.close()
		
	def test_verifySentences(self):

		self.verifySentences("I love dogs", 'Positive')
		self.verifySentences("I hate spiders", 'Negative')
		self.verifySentences("I am walking on the street", 'Neutral')
		
		
if __name__ == '__main__':
	unittest.main()		
