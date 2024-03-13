import streamlit as st
from datetime import datetime #, date
import requests
import time
import warnings
import multiprocessing
import pandas as pd
from better_profanity import profanity
import concurrent.futures
import sys

def task(index , xx):
    # st.write("working")
    return(index,profanity.censor(xx, "*"))

##########profanity filter
# def multiprocessing_function(text_data):
#     st.info("**Data Filtering in Progress**: This Process would take about 2-3 Minutes!")

#     with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
#         futures = [executor.submit(task, index, text) for index, text in enumerate(text_data)]

#     try:
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]
#     except Exception as e:
#         print("exception in worker process", e)
#         return text_data

#     # Sort the results based on the original index
#     results.sort(key=lambda x: x[0])
#     final_results = [result[1] for result in results]
#     return final_results
# sys.path.append("./")
# # from utils import task

# def access(client_id,secret_key,username,password):
#     auth = requests.auth.HTTPBasicAuth(client_id, secret_key)
#     data = {
#         'grant_type': 'password',
#         'username': username,
#         'password': password
#     }

#     headers = {'User-Agent': 'MyAPI/0.0.1'}

#     res = requests.post('https://www.reddit.com/api/v1/access_token', 
#                         auth = auth, 
#                         data = data,
#                         headers = headers)
#     token = res.json()['access_token']

#     headers['Authorization'] = f'bearer {token}'    

#     return headers

# def task(index , xx):
#     print("working")
#     return(index,profanity.censor(xx, "*"))

# ##########profanity filter
# def multiprocessing_function(text_data):
    
#     st.info("**Data Filtering in Progress**: This Process would take about 2-3 Minutes!")
#     try:
#         with multiprocessing.Pool(processes=6) as pool:
#             st.write("working 1")
#             res = pool.starmap(task, enumerate(text_data)) 
#     except Exception as e:
#         print("exception in worker process", e)
#         return text_data

#     res.sort(key=lambda x: x[0])
#     final_results = [result[1] for result in res]
#     return final_results

# ###################### dataframe to csv conversion
# def convert_df(df):
#    return df.to_csv(index=False).encode('utf-8')

# ###################### reddit data extraction
# def reddit_data(time_wanted,client_id,secret_key,username,password):
    
#     progress_text = "Validating the Credentials, Please wait."

#     headers = access(client_id,secret_key,username,password)

#     my_bar = st.progress(0, text=progress_text)

#     #initial set collection
#     res = requests.get('https://oauth.reddit.com/r/Dreams/new',
#                     headers = headers, params={'limit': '100', 'no_profanity':True})

#     df = pd.DataFrame()

#     for post in res.json()['data']['children']:
#         df = pd.concat([df,pd.DataFrame({'subreddit': post['data']['subreddit'],
#                                                     'title': post['data']['title'],
#                                                     'text': post['data']['selftext'],
#                                                     'date': post['data']['created']},index=[0])],ignore_index=True )
    
#     #further back collection
#     latest_key = post['kind'] + '_' + post['data']['id']

#     my_bar.progress(3, text = "Credentials Validated!")
#     my_bar.progress(5, text = "Initizlizing Data Collection From Reddit")
#     while df.tail(1)['date'][df.tail(1)['date'].index[0]] > datetime.timestamp(time_wanted):
#         for req in range(100):
        
#             res = requests.get('https://oauth.reddit.com/r/Dreams/new',
#                                 headers = headers, 
#                                 params={'limit': '100', 'after': latest_key, 'no_profanity':True})
            
#             for post in res.json()['data']['children']:
#                 df = pd.concat([df,pd.DataFrame({'subreddit': post['data']['subreddit'],
#                                                     'title': post['data']['title'],
#                                                     'text': post['data']['selftext'],
#                                                     'date': post['data']['created']},index=[0])], ignore_index= True)

#             latest_key = post['kind'] + '_' + post['data']['id']

#             if req * 15 <= 100:    
#                 my_bar.progress(req *15, text = f"{df.shape[0]} Dreams Collected")
#             else:
#                 my_bar.progress(100, text = f"{df.shape[0]} Dreams Collected")

#             if len(df) >= 985:
#                 latest = df.tail(1)['date'][df.tail(1)['date'].index[0]]
#                 st.success("Data Collection Completed!")
#                 col11, col22 = st.columns([2,4])
#                 df.date = [datetime.fromtimestamp(d) for d in df.date] 
#                 with col11:
#                     st.success(f'**Data Count**: {len(df)} Dreams')
#                 with col22:
#                     st.success(f'**Earliest Dream Upload Date**: {datetime.fromtimestamp(latest)}')
#                 time1 = time.time()
#                 try:
#                     df.text = multiprocessing_function(df.text)
#                 except:
#                     pass
#                 time2 = time.time()
#                 col33, col44 = st.columns([3,2])
#                 with col33:
#                     st.success(f'**Data Filtering Complete!**')
#                 with col44:
#                     st.success(f'**Time Consumed**: {round((time2-time1)/60,2)} minutes')
#                 return df, res.json()['data']['children'][1]

#     else: 
#         st.success("Data Collection Completed!")
#         st.success(f'**Data Count**:{len(df)}')
#         st.success(f'**Last Dream Upload Date**: {datetime.fromtimestamp(latest)}')
#         return df, res.json()['data']['children'][1]
    
# if __name__ == "__main__":
#     year = sys.argv[1]
#     month = sys.argv[2]
#     day = sys.argv[3]
#     three = sys.argv[4]
#     last = sys.argv[5]
#     client_id = sys.argv[6]
#     secret_key = sys.argv[7]
#     username = sys.argv[8]
#     password = sys.argv[9]

#     time_wanted = datetime(int(year),int(month),int(day),int(three),int(three),int(three),int(last))

#     reddit_data(time_wanted,client_id,secret_key,username,password)