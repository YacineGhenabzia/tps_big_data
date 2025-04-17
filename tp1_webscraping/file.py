import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest
titles=[]
links=[]
votes=[]
descriptions=[]
answers=[]
views=[]
asks=[]
the_answers=[]
reputation_scores=[]
users=[]

for i in range(1,20000):
 response = requests.get(f"https://stackoverflow.com/questions?tab=newest&page={i}")
 response=response.content
 soup = BeautifulSoup(response, "html.parser")

 title_element=soup.find_all("h3",class_="s-post-summary--content-title")


 link_elements = [title.find("a", class_="s-link") for title in title_element ]

 vote=soup.find_all("div",class_="s-post-summary--stats-item s-post-summary--stats-item__emphasized")
 answer=soup.find_all("div",class_="s-post-summary--stats-item")
 view=soup.find_all("div",class_="s-post-summary--stats-item")
 description=soup.find_all("div",class_="s-post-summary--content-excerpt")
 ask=soup.find_all("time",class_="s-user-card--time")
 reputation_score=soup.find_all("span",class_="todo-no-class-here")
 user=soup.find_all("a",class_="flex--item")
 
 
 for i in range(len(title_element)):
    titles.append(title_element[i].text.strip())
    links.append("https://stackoverflow.com" + link_elements[i].attrs['href'])
    votes.append(vote[i].text.strip())
    answers.append(answer[i].text.strip())
    views.append(view[i].text.strip())
    descriptions.append(description[i].text.strip())
    asks.append(ask[i].text.strip())
    reputation_scores.append(reputation_score[i].text.strip())
    users.append(user[i].text.strip())
 
 #time.sleep(2) 
    
file_list=[titles,links,votes,descriptions,answers,views,asks,reputation_scores,users]
exported=zip_longest(*file_list)


with open("d:/folder/questions/questions_three_1.csv","w",encoding="utf-8",newline="") as myfile:
    wr=csv.writer(myfile ,delimiter=';')
    wr.writerow(["Title","Link","Votes","Description","Answers","Views","Asks","Reputation Score","User"])
    wr.writerows(exported)
    print("file created")

