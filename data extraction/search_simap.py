from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import os
import numpy as np
import json

projects = {}
projects_html = {}


# set notice_nr
#notice_nr = "947313"

# set list of notice_nr

# loading data
'''
df = pd.read_csv('./input/SIMAP_soft1.csv',sep=',')
df.shape  # (# items (rows), # features (columns))

notice_nr_list_ints = df['Notice_Nr'].tolist()
notice_nr_list = [str(i) for i in notice_nr_list_ints]
'''
#notice_nr_list = [1098413, 1098705, 1090913, 1098517, 1100771]
notice_nr_list_int = [478935,489529,505485,725527,727469,729067,1105009,1105005,1103179,1104507,1103211,1103269,1104491,1103245,1104503,1104485,1103743,1103755,1074407,1116855,1117135,1025881]
notice_nr_list = [str(i) for i in notice_nr_list_int]


# open chrome
driver = webdriver.Chrome('/Users/wasp/desktop/softcom_project/data_extraction/chromedriver')
# go to start_page_url and get html (because otherwise "Your session has ended or your browser does not allow cookies")
start_page_url = 'https://www.simap.ch/shabforms/COMMON/application/applicationGrid.jsp?template=2&view=1&page=/MULTILANGUAGE/simap/content/start.jsp'
driver.get(start_page_url)
soup_start_page=BeautifulSoup(driver.page_source,'lxml')


for notice_nr in notice_nr_list:
    # get url to page with specific project through notice nr
    project_url = "https://www.simap.ch/shabforms/servlet/Search?NOTICE_NR=" + notice_nr
    driver.get(project_url)
    # get url specific project
    soup_page = BeautifulSoup(driver.page_source,'lxml')  # page 1
    colon4 = soup_page.find_all('td', {'class': re.compile("tdcol4")})
    colon3 = soup_page.find_all('td', {'class': re.compile("tdcol3")})
    colon2 = soup_page.find_all('td', {'class': re.compile("tdcol2")})
    colon1 = soup_page.find_all('td', {'class': re.compile("tdcol1")})
    project = {} # create dictionary
    project['project_id'] = []
    project['notice_no'] = []
    project['type'] = []
    #projects[p]['contracting_authority'] = []
    project['publication_date'] = []
    project['project_title'] = []
    #projects[p]['submission_deadline'] = []
    #projects[p]['subheading'] = []
    #    projects[p]['type_of_order'] = []
    #  projects[p]['type_of_procedure'] = []
    #projects[p]['WTO']
    project['CPV'] = []
    #projects[p]['BCC']
    #projects[p]['NPK']
    #projects[p]['project_details_html'] = []
    project['project_details'] = []
    project_html = {}
    project_html['project_id'] = []
    project_html['notice_no'] = []
    project_html['html'] = []
    # get project_title p
    project['project_title'] = colon4[0].find_all(text=True)[0]
    # get notice_no
    project['notice_no'] = colon2[0].getText()
    # get publication date
    project['publication_date'] = colon1[0].getText()
    project_url = colon4[0].find('a').get('href')
    # go to project detail and get html
    driver.get('https://www.simap.ch' + project_url)
    soup_projectdetail = BeautifulSoup(driver.page_source, 'lxml')
    project_details_html = soup_projectdetail.find('div', {'class': re.compile('preview')})
    # get project_ID 
    project['project_id'] = soup_projectdetail.find('div', {'class': re.compile("result_head")}).find_all('span')[0].next_sibling.split()[-1]
    # get type
    project['type'] = ' '.join(soup_projectdetail.find('div', {'class': re.compile("result_head")}).find_all('span')[-1].next_sibling.split())
    # get all tags with h. and dd
    list_h_dd = project_details_html.find_all([re.compile('h.'), re.compile('dd')])
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
                            project['CPV'].append(str(nr_list[n]))
                        else:
                            break
                        # add BCC, NPK?
    project['project_details'] = details
    project_html['project_id'] = project['project_id']
    project_html['notice_no'] = project['notice_no']
    project_html['html'] = str(project_details_html)
            #print(projects[p])
            # directory_1 = '../projects_ID_NO/'
            #        directory_2 = '../projects_ID_NO_html/'
    directory_1 = './projects_ID_NO_soft/'
    directory_2 = './projects_ID_NO_html_soft/'
    filename_1 = 'ID_' + project['project_id'] + '_NO_' + project['notice_no'] + '.json'
    filename_2 = 'ID_' + project['project_id'] + '_NO_' + project['notice_no'] + '_html.json'
    print(filename_1)
    with open(directory_1 + filename_1, 'w') as outfile:
        json.dump(project, outfile)
    with open(directory_2 + filename_2, 'w') as outfile:
        json.dump(project_html, outfile) 





# exemple: https://www.simap.ch/shabforms/servlet/Search?NOTICE_NR=935643