import streamlit as st
from datetime import datetime #, date
import requests
import time
import warnings
import multiprocessing
import pandas as pd
from better_profanity import profanity
import sys
import subprocess
import os
import concurrent.futures

sys.path.append("./app/")
sys.path.append("./app/pages")

# import utils


########################################################################################
#############################       data collection page      ##########################
########################################################################################

warnings.filterwarnings('ignore')

def task(index , xx):
    st.write("working")
    return(index,profanity.censor(xx, "*"))

##########profanity filter
def multiprocessing_function(text_data):
    st.info("**Data Filtering in Progress**: This Process would take about 2-3 Minutes!")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(task, index, text) for index, text in enumerate(text_data.tolist())]

    try:
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    except Exception as e:
        print("exception in worker process", e)
        return text_data

    # Sort the results based on the original index
    results.sort(key=lambda x: x[0])
    final_results = [result[1] for result in results]
    return final_results

# ###################### dataframe to csv conversion
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# ###################### reddit data extraction
def reddit_data(time_wanted, headers):
    progress_text = "Validating the Credentials, Please wait."
    my_bar = st.progress(0, text=progress_text)

    #initial set collection
    res = requests.get('https://oauth.reddit.com/r/Dreams/new',
                    headers = headers, params={'limit': '100', 'no_profanity':True})

    df = pd.DataFrame()

    for post in res.json()['data']['children']:
        df = pd.concat([df,pd.DataFrame({'subreddit': post['data']['subreddit'],
                                                    'title': post['data']['title'],
                                                    'text': post['data']['selftext'],
                                                    'date': post['data']['created']},index=[0])],ignore_index=True )
    
    #further back collection
    latest_key = post['kind'] + '_' + post['data']['id']

    my_bar.progress(3, text = "Credentials Validated!")
    my_bar.progress(5, text = "Initizlizing Data Collection From Reddit")
    while df.tail(1)['date'][df.tail(1)['date'].index[0]] > datetime.timestamp(time_wanted):
        for req in range(100):
        
            res = requests.get('https://oauth.reddit.com/r/Dreams/new',
                                headers = headers, 
                                params={'limit': '100', 'after': latest_key, 'no_profanity':True})
            
            for post in res.json()['data']['children']:
                df = pd.concat([df,pd.DataFrame({'subreddit': post['data']['subreddit'],
                                                    'title': post['data']['title'],
                                                    'text': post['data']['selftext'],
                                                    'date': post['data']['created']},index=[0])], ignore_index= True)

            latest_key = post['kind'] + '_' + post['data']['id']

            if req * 15 <= 100:    
                my_bar.progress(req *15, text = f"{df.shape[0]} Dreams Collected")
            else:
                my_bar.progress(100, text = f"{df.shape[0]} Dreams Collected")

            if len(df) >= 985:
                latest = df.tail(1)['date'][df.tail(1)['date'].index[0]]
                st.success("Data Collection Completed!")
                col11, col22 = st.columns([2,4])
                df.date = [datetime.fromtimestamp(d) for d in df.date] 
                with col11:
                    st.success(f'**Data Count**: {len(df)} Dreams')
                with col22:
                    st.success(f'**Earliest Dream Upload Date**: {datetime.fromtimestamp(latest)}')
                time1 = time.time()
                try:
                    df.text = multiprocessing_function(df.text)
                except:
                    pass
                time2 = time.time()
                col33, col44 = st.columns([3,2])
                with col33:
                    st.success(f'**Data Filtering Complete!**')
                with col44:
                    st.success(f'**Time Consumed**: {round((time2-time1)/60,2)} minutes')
                return df, res.json()['data']['children'][1]

    else: 
        st.success("Data Collection Completed!")
        st.success(f'**Data Count**:{len(df)}')
        st.success(f'**Last Dream Upload Date**: {datetime.fromtimestamp(latest)}')
        return df

st.title("Data Collection")
st.write("Equipped with the background on dreams and dreaming, now you should be ready to roll up your sleeves and get right into dream data collection!")
st.write("Before anything else, you are going to first acquire the data which later will be analyzed using NLP. For that matter, shortly, you will be collecting data in real time from Reddit, an online environment for sharing and discussing information. Note that Reddit is organized in threads called “subreddits” which essentially are topics, where the discussion actually takes place. As you may have guessed – one such subreddit – in fact the only subreddit that you will use relates to reported dreams. It is a community where users share their dreams seeking interpretations or simply for the sake of sharing.")
st.write("To collect the data on dreams in real time, posted by the Reddit users, you first need to access the Reddit Application Programming Interface (API). Information on how to do that can be found here. That is a necessary step which fortunately won’t take more than a minute or so … for sure it won’t be too long to put you to sleep before the main event! The below will serve as a guideline for the readers to gain access to the Reddit Developer's account.")
st.warning("Please refrain from using Chrome for the process below! If you are a Mac user, please try using Safari, and if Windows, try using Edge!")
st.video("https://youtu.be/_xK1OEfd3iI")
st.write("Click on this [link](https://www.reddit.com/prefs/apps) to get to the Reddit API OAUTH2 Page!")
st.write("Now that you have gained access to the Reddit developer's account, you are ready to use the Reddit API in order to gather dreams that will then be used as the data for NLP. The subreddit to be used is r/Dreams, which can be easily searched on search engines for viewing purposes. In the below text boxes, please input your Reddit information in order to collect the dreams. ")

st.write("Note that the raw data collected from Reddit are in JSON (JavaScript Object Notation) format. For clarity, a JSON file has a nested format, where information is stored like a hierarchical tree (not a dataframe!). As an important pre-processing step the necessary portions of that JSON data will be selected and put into a dataframe. But worry not – that is going to be done for you automatically in the back end of this app (to keep you awake, after all!) One last detail: after the raw data gets pulled from Reddit, there will be an initial data cleaning step to drop the Null values and perform profanity checks before displaying the data. ")
st.write("Ready? Go!")

with st.form("reddit_cred"):
    client_id = st.text_input("Reddit Client Id")
    secret_key = st.text_input("Reddit Secret Key")
    username = st.text_input("Reddit User Name")
    password = st.text_input("Reddit Password")

    submitted = st.form_submit_button("Submit")

if submitted:
    time_wanted = datetime(2023, 1, 20, 00, 00, 00, 342380)

    try:
        client_id = client_id
        secret_key = secret_key

        auth = requests.auth.HTTPBasicAuth(client_id, secret_key)
        data = {
            'grant_type': 'password',
            'username': username,
            'password': password
        }

        headers = {'User-Agent': 'MyAPI/0.0.1'}

        res = requests.post('https://www.reddit.com/api/v1/access_token', 
                            auth = auth, 
                            data = data,
                            headers = headers)
        token = res.json()['access_token']

        headers['Authorization'] = f'bearer {token}'    

        st.session_state['reddit'], st.session_state['json_file'] = reddit_data(time_wanted, headers)

        my_bar = st.progress(0, text="Initiating Data Preprocessing")
        time.sleep(3)

        my_bar.progress(40, "Dropping Empty Observations") 
        st.session_state['reddit'] = st.session_state['reddit'].dropna()

        # reddit_data['text'] = reddit_data['text'].apply(apply_censor)
        # reddit['text'] = [profanity.censor(i) for i in reddit['text']]
        time.sleep(3)
        my_bar.progress(80, "Converting pandas dataframe to CSV")

        time.sleep(3)
        my_bar.progress(90, "Generating Previews")
        time.sleep(3)
        my_bar.progress(100, "Job Complete")

        st.write("Curious how the raw data look like? Take a look below to see it for one of the dreams that was just pulled from Reddit. To best understand how JSON works, think of the folder directories in your local computers. Within your Desktop folder, say you have a folder for each class you take. And within each class folder, imagine you have different assignment folders, containing assignments completed. As such JSON divides information in a hierarchical format: the deeper nested values are specific details pertaining to the encompassing information. Please press on the rotated green triangle below to assess the JSON file. This is a good opportunity for you to get familiar with JSON, by the way!")
        st.json(st.session_state['json_file'], expanded= False)

        st.write("Finally, the below is the dataframe based on the JSON file. Note that from the JSON data the app extracts subreddit thread name, the title of the post, the dream, and the date at which the post was made. The analyses taking part in this app exclude any comments that may be made by users following up on a post. ")
        st.dataframe(st.session_state['reddit'].head(30))

        st.write("Why, do you think, one would ever need JSON if dataframes seem so much cleaner? JSON-formatted data and dataframes happen to serve different purposes! While dataframe is the go-to format for presenting data for statistical and machine learning analyses, JSON happens to be the common format for data interchange between applications or between a client and a server. To make this more intuitive, imagine you are building a mobile application for financial trading and the app uses information collected in real time from a certain server that hosts financial data. That data can efficiently be provided/sent from the server in JSON format which, upon further cleaning and preprocessing can be represented in a dataframe and be used in statistical and machine learning analyses that would be used in the app that you are building. So, different formats for different purposes!")
    
        st.info("Next click on the next tab on the left to move on to the Data Cleaning Section!" ,icon="ℹ️")
    except KeyError:
        st.warning("Please enter correct Reddit Credentials", icon="⚠️")