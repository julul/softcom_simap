# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
# https://stackoverflow.com/questions/31252037/getting-text-from-br-tags-in-beautifulsoup
# https://beautiful-soup-4.readthedocs.io/en/latest/


from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import os
import numpy as np
import json
import sys


path = '/Users/julia/Desktop/git_repos/softcom_simap/data_extraction/chromedriver'
#sys.path.append(path)

projects = {}
projects_html = {}

# open chrome
driver = webdriver.Chrome(path)
# go to start_page_url and get html (because otherwise "Your session has ended or your browser does not allow cookies")
start_page_url = 'https://www.simap.ch/shabforms/COMMON/application/applicationGrid.jsp?template=2&view=1&page=/MULTILANGUAGE/simap/content/start.jsp'
driver.get(start_page_url)
soup_start_page=BeautifulSoup(driver.page_source,'lxml')
# set start date DD.MM.YYYY
start_date = "01.01.2020" # probably exclusive
# set end date DD.MM.YYYY
end_date = "02.01.2020"  # inclusive

# get url to all publications (invitation to tender, awards, other publications)
all_publications_url = "https://www.simap.ch/shabforms/servlet/Search?TIMESPAN=VARIABLE&STAT_TM_1=" + start_date + "&STAT_TM_2=" + end_date

# go to 'all publications' (first page) and get html
driver.get(all_publications_url)
soup_page = BeautifulSoup(driver.page_source,'lxml')  # page 1

# go to lastpage to get its number
lastpage_url = 'https://www.simap.ch' + soup_page.find('div', {'class': re.compile("mitte")}).find_all('a')[-1].get('href')
driver.get(lastpage_url)
soup_lastpage = BeautifulSoup(driver.page_source,'lxml')
content_lastpage = soup_lastpage.find_all('span', {'class': re.compile("mr_10")})
lastpage_nr = int(content_lastpage[len(content_lastpage)-3].getText())

# go through each page x
for x in range(1,lastpage_nr+1): # go from page 1 to lastpage_nr (i.e. lastpage_nr+1 excluded)
    if x>1 :       
        page_url = 'https://www.simap.ch/shabforms/servlet/Search?EID=1&PAGE=' + str(x)
        driver.get(page_url)
        soup_page = BeautifulSoup(driver.page_source,'lxml')

    colon4_x = soup_page.find_all('td', {'class': re.compile("tdcol4")})
    colon3_x = soup_page.find_all('td', {'class': re.compile("tdcol3")})
    colon2_x = soup_page.find_all('td', {'class': re.compile("tdcol2")})
    colon1_x = soup_page.find_all('td', {'class': re.compile("tdcol1")})

    # go through each project p on page x
    for p in range(0,len(colon4_x)): # p loops from project 0 to len(colon4_x)-1 (i.e. len(colon4_x) excluded)
        projects[p] = {} # create dictionary
        projects[p]['project_id'] = []
        projects[p]['notice_no'] = []
        projects[p]['type'] = []
        #projects[p]['contracting_authority'] = []
        projects[p]['publication_date'] = []
        projects[p]['project_title'] = []
    #    projects[p]['submission_deadline'] = []
        #projects[p]['subheading'] = []
    #    projects[p]['type_of_order'] = []
      #  projects[p]['type_of_procedure'] = []
        #projects[p]['WTO']
        projects[p]['CPV'] = []
        #projects[p]['BCC']
        #projects[p]['NPK']
        #projects[p]['project_details_html'] = []
        projects[p]['project_details'] = []

        projects_html[p] = {}
        projects_html[p]['project_id'] = []
        projects_html[p]['notice_no'] = []
        projects_html[p]['html'] = []

        # get project_title p
        projects[p]['project_title'] = colon4_x[p].find_all(text=True)[0]
        # get notice_no
        projects[p]['notice_no'] = colon2_x[p].getText()
        # get publication date
        projects[p]['publication_date'] = colon1_x[p].getText()
 
          # get submission deadline
     #   projects[p]['submission_deadline'] = colon3_x[p].find_all('br')[-1].next_sibling
        # get type of order
     #   if len(colon3_x[p].find_all('br')) < 3:
     #       projects[p]['type_of_order'] = "unknown"
     #   else:
     #       tpo = colon3_x[p].find_all('br')[-3].next_sibling
     #       if isinstance(tpo,str): 
     #           projects[p]['type_of_order'] = tpo
     #       else:
     #           projects[p]['type_of_order'] = "unknown"
        # get type of procedure (not working yet. TypeError: Object of type 'Tag' is not JSON serializable, because of 'br', e.g. No 1099005)
      #  projects[p]['type_of_procedure'] = colon3_x[p].find_all('br')[-2].next_sibling

        # get project_url of project p
        project_url = colon4_x[p].find('a').get('href')
        
        # go to project detail and get html
        driver.get('https://www.simap.ch' + project_url)
        soup_projectdetail = BeautifulSoup(driver.page_source, 'lxml')
        project_details_html = soup_projectdetail.find('div', {'class': re.compile('preview')})
        # get project_ID 
        projects[p]['project_id'] = soup_projectdetail.find('div', {'class': re.compile("result_head")}).find_all('span')[0].next_sibling.split()[-1]
        # get type
        projects[p]['type'] = ' '.join(soup_projectdetail.find('div', {'class': re.compile("result_head")}).find_all('span')[-1].next_sibling.split())
        # get all tags with h. and dd
        list_h_dd = project_details_html.find_all([re.compile('h.'), re.compile('dd')])
        print(list_h_dd[1])
        details = {} # create dictionary
        headers = [] # create list to access by index
        for i in range(0,len(list_h_dd)):  
            if 'h' in list_h_dd[i].name:  # tag is 'h.'
                header_nr = int(list_h_dd[i].name[1:])
                header_text = ' '.join(list_h_dd[i].getText().split())
        
                if header_nr > len(headers): 
                    headers.append([header_text])
                else:                           
                    headers[header_nr-1].append(header_text)
                details[header_text] = []
        
                if header_nr > 1:
                    ded_header = headers[header_nr-2][-1] # get last element of lower header (dedicated header)
                    details[ded_header].append(header_text)
            elif 'dd' in list_h_dd[i].name: # tag is 'dd'
                draft = list_h_dd[i].find_all(text=True)
                if draft:
                    drafts = []
                    for k in range(0, len(draft)):
                        drafts.append(' '.join(draft[k].split()))            
                    dd_text = ' '.join(drafts)
                    details[header_text].append(dd_text)
                    if 'CPV' in dd_text:
                        nr_list = [int(s) for s in dd_text.split() if s.isdigit()]
                        for n in range(0, len(nr_list)):
                            if len(str(nr_list[n])) == 8:
                                projects[p]['CPV'].append(str(nr_list[n]))
                            else:
                                break
                    # add BCC, NPK?
                            

        projects[p]['project_details'] = details

        projects_html[p]['project_id'] = projects[p]['project_id']
        projects_html[p]['notice_no'] = projects[p]['notice_no']
        projects_html[p]['html'] = str(project_details_html)

        #print(projects[p])
#        directory_1 = '../projects_ID_NO/'
#        directory_2 = '../projects_ID_NO_html/'
        directory_1 = '../data/projects_ID_NO/'
        directory_2 = '../data/projects_ID_NO_html/'
        if not os.path.isdir(directory_1):
            os.makedirs(directory_1)
        if not os.path.isdir(directory_2):
            os.makedirs(directory_2)            
        filename_1 = 'ID_' + projects[p]['project_id'] + '_NO_' + projects[p]['notice_no'] + '.json'
        filename_2 = 'ID_' + projects[p]['project_id'] + '_NO_' + projects[p]['notice_no'] + '_html.json'
        with open(directory_1 + filename_1, 'w') as outfile:
            json.dump(projects[p], outfile)
        with open(directory_2 + filename_2, 'w') as outfile:
            json.dump(projects_html[p], outfile) 

    