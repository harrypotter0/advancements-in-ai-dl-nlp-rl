#-----------------------------------------------------------------------------
#Author: Vibhu Agarwal
#-----------------------------------------------------------------------------

from selenium import webdriver
from getpass import getpass
from text_to_speech import RecognizeSpeech
import datetime
import time
import sys


#--------------------- Email ID and password of Facebook ---------------------
usr = input('Enter email ID: ') #Hardcode the username here eg. 'abcd.efg@gmail.com'
pwd = getpass() #Hardcode the password here 'securepassword123'
#-----------------------------------------------------------------------------


#------------------------- For getting date and time -------------------------
now = datetime.datetime.now()
date = now.strftime("%d")
month = now.strftime("%B")
year = now.strftime("%Y")
time_samay = now.strftime("%X").replace(':','-')
#-----------------------------------------------------------------------------


#----------- Recorded audio file name: date and time of recording ------------
file_name = date+' '+month+' '+year+' '+time_samay+'.wav'
#-----------------------------------------------------------------------------


print("What's on your mind?\n")
while True:
    post = RecognizeSpeech(file_name, 10) #Recording for 10 seconds
    post += "\n#selenium #pyaudio #automation #python"
    print('\nRecorded Post:\n----------------------------------------------------------------------------')
    print(post)
    print('----------------------------------------------------------------------------')
    choic = input('\nEnter \n1 for ok,\n2 to record again\n')
    if choic not in ('1','2'):
        input('Invalid Choice\n')
        sys.exit()
    elif choic == '1':
        break

#---------------------- Opening Facebook in Web Browser ----------------------
driver = webdriver.Chrome()   #For Google Chrome

driver.get('https://www.facebook.com/')
#-----------------------------------------------------------------------------


#--------------- Entering user details on Facebook Welcome page --------------
try:
    username_box = driver.find_element_by_id('email')
    username_box.send_keys(usr)
except:
    print('Error: You might already be logged in\nFirst logout from Facebook\n')
    sys.exit()

pass_box = driver.find_element_by_id('pass')
pass_box.send_keys(pwd)


login_button = driver.find_element_by_id('loginbutton')
login_button.submit()
#-----------------------------------------------------------------------------


#---------------------- Typing the post in post box --------------------------
status= driver.find_element_by_xpath("//textarea[@name='xhpc_message']")
status.send_keys(post);
#-----------------------------------------------------------------------------

#input("Waiting")
postbutton = driver.find_element_by_xpath("//button[contains(.,'Post')]")
postbutton.submit()
time.sleep(3)

driver.quit()
