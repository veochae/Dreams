########################################################################################
########################################################################################
######################       Dreams NLP Streamlit Application     ######################
########################################################################################
########################################################################################
######################                Veo Chae                 #########################
########################################################################################
########################################################################################


########################################################################################
#############################       Package Requirements   #############################
########################################################################################
#python native packages
import requests
import re
import os
import glob
import sys
import math
import json
import time
import warnings
import multiprocessing

#streamlit
import spacy_streamlit
import streamlit as st

#common add ons
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import nltk
@st.cache_resource
def nltk_downloads():
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download("punkt")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')    

nltk_downloads()
import spacy
from datetime import datetime, date
from sklearn.feature_extraction.text import CountVectorizer


#plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

#other pacakges
from better_profanity import profanity

#huggingface
from transformers import pipeline

#openai
import openai

#tensorflow
import torchvision
import torch
########################################################################################
#############################       required UDFs     #############################
########################################################################################
warnings.filterwarnings('ignore')

##########profanity filter
def task(index , xx):
    return(index,profanity.censor(xx, "*"))

def multiprocessing_function(text_data):
    st.info("**Data Filtering in Progress**: This Process would take about 2-3 Minutes!")
    try:
        with multiprocessing.Pool(processes=6) as pool:
            res = pool.starmap(task, enumerate(text_data)) 
        res.sort(key=lambda x: x[0])
        final_results = [result[1] for result in res]
    except Exception as e:
        print("exception in worker process", e)
        raise e
    return final_results

##########en-core-sm preload
@st.cache_resource
def load_nlp():
    return spacy.load('en_core_web_sm')

##########wordcloud
def wordcloud(x, lim, collocation_threshold, stopword):
    text = " ".join(x)
    cloud = WordCloud(collocations = False, stopwords = stopword, max_words = lim,min_word_length = 3, collocation_threshold = collocation_threshold).generate(text)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)

###################### dataframe to csv conversion
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

###################### reddit data extraction
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

############### chat-gpt incorporated function
def summarize_dream(prompt, length):
    response = openai.Completion.create(
        engine="text-davinci-003",                  #most advanced version of text related algo in open ai
        prompt=prompt,                              #what is being inputted to gpt
        max_tokens=length,                            #maximum number of words
        n=1,                                        #number of outputs
        stop=None,                                  #stop when
        temperature=0.5,                            #how much "risk" do you want the gpt to take
    )

    text = response.choices[0].text.strip()
    return text

########################################################################################
#############################       introduction page      #############################
########################################################################################

def introduction():
    
    col4, col5, col6 = st.columns([1,8,1])

    with col4:
        st.write("")

    with col5:
        st.title("Analyzing Dreams Using Natural Language Processing")

    with col6:
        st.write("")

    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("./Dream.jpeg")

    with col3:
        st.write("")


########################################################################################
#############################       data collection page      ##########################
########################################################################################

def data_collection():
    st.title("Data Collection")
    st.write("Before anything else, you are going to first acquire the data which later will be analyzed using NLP. For that matter, shortly, you will be collecting data in real time from Reddit, an online environment for sharing and discussing information. Note that Reddit is organized in threads called “subreddits” which essentially are topics, where the discussion actually takes place. As you may have guessed – one such subreddit – in fact the only subreddit that you will use relates to reported dreams. It is a community where users share their dreams seeking interpretations or simply for the sake of sharing.")
    st.write("To collect the data on dreams in real time, posted by the Reddit users, you first need to access the Reddit Application Programming Interface (API). Information on how to do that can be found here. That is a necessary step which fortunately won’t take more than a minute or so … for sure it won’t be too long to put you to sleep before the main event! The below will serve as a guideline for the readers to gain access to the Reddit Developer's account.")
    st.warning("Please refrain from using Chrome for the process below! If you are a Mac user, please try using Safari, and if Windows, try using Edge!")
    st.video("https://youtu.be/k6TD-pOsh8s")
    st.write("Click on this [link](https://www.reddit.com/prefs/apps) to get to the Reddit API OAUTH2 Page!")
    st.write("Now that you have gained access to the Reddit developer's account, you are ready to use the Reddit API in order to gather dreams that will then be used as the data for NLP. The subreddit to be used is r/Dreams, which can be easily searched on search engines for viewing purposes. In the below text boxes, please input your Reddit information in order to collect the dreams. ")

    st.write("The process of Data Collection follows the below details: ")
    st.write("1. Your authentication is granted with correct Client Id, Secret Key, Username, and Password. This implies that Reddit knows who is accessing their database and can identify whether you have access to the data of observance. If you do not input the correct credentials, your requests will be denied.")
    st.write("2. With the correct credentials approved by Reddit, now we start collecting the Dreams. Majority of the major platform APIs prevent users from extracting large quantites of data at once. This is in order to prevent injection of malware viruses into the system, as well as to prevent data mining using a data bot. In order to constrain such possibilities, Reddit has placed a maximum number of data that can be collected at each run of request for data. Thus, to not manually rerun and append data each and every run, the script embeded in this app will take short 'time-off' after each run in order to not be restricted by Reddit data collection regulations. For each run, the amount of collected data will be displayed in the progress bar.")
    st.write("3. Contrary to what users may believe, the raw data that is collected from Reddit is in json format. For clarity, json file is a nested dictionary format, where all infromation is stored like a hierarchical tree, not a dataframe. Thus, we select only portions of the json data that is required for this anlaysis and create a dataframe.")
    st.write("4. After the intial raw data collection process, the embedded script performs initial cleaning on the dataset. This process includes the rudimentary process such as dropping Null values and profanity checks.")
    st.write(" ")
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

            st.write("Ever wondered why one would ever need JSON if dataframes seem so much cleaner? You see, although dataframes are intuitive – their size and the consequent burden on memory can become extremely large as the number of observations or features increase! Further, dataframes typically store various meta data, such as the data type, etc. On the contrary, the JSON format only stores the text values of the data. Therefore, it is a structured word file that can be interpreted in hierarchical fashion when imported into an Integrated Development Environment (IDE). This saves tremendous amount of space when it comes to storing large datasets. And because typically the data in APIs are extremely large, JSON is the go-to format!")
        
            st.info("Next click on the next tab on the left to move on to the Data Cleaning Section!" ,icon="ℹ️")
        except KeyError:
            st.warning("Please enter correct Reddit Credentials", icon="⚠️")

########################################################################################
#############################       data cleaning  page      ###########################
########################################################################################

def data_cleaning():
    st.title("Data Preprocessing")

    try:
        st.write("With the raw dataset in hand, now we move on to the critical stage of analysis: Data Manipulation.")
        st.write("In the dataframe that represents the dreams, each observations (row) represents a unique dream. In general, each unique observation in a collection of texts is referred to as a “document”, while collectively the documents are referred to as a “corpus” or a “text corpus”. With the raw corpus in hand, you are about to embark on an important process that is at the heart of NLP: Data Cleaning. So hold on tight and keep your eyes open – you are about to learn a host of useful tips and tricks.")
        st.write("As one may be aware, different data types require various data cleaning processes. For instance, numeric values may require changing the data type to its correct state, normalization or standardization, and more. Further, categorical variables often need one-hot-encoding or categorical type transformation. In the case of text data, the cleaning process is quite arduous and has various tasks, which are stated below.")
        st.write("**Basic Cleaning** : During this step the text is converted into lower (or upper) case and then stripped off of parts that the data scientist finds unimportant for the downstream NLP tasks. This finding of text within the text is often done using “Regex” which stands for “regular expression”! Regex allows finding of certain patterns that one wishes to identify (and often remove or replace) in a given text. Say the data scientist wishes to eliminate numbers, punctuation, URLs, abbreviations, etc. before moving on to analyzing the text. ")
        st.write("**Tokenization** : Tokenization is the process of segmenting the text into smaller pieces – tokens. Those tokens are usually just unigrams – single words or equivalently the lemmatized or stemmed versions of words if lemmatization/stemming has been applied. To preserve context, text can instead be tokenized into pairs of neighboring words called bigrams. In general, depending on the situation, text can be tokenized into n-grams: collections of neighboring n words! ")
        st.write("**Stopwords Removal** : In writing an English sentence, commonly repeated words such as articles and prepositions often can be eliminated without much loss of information when it comes to NLP. If left untouched, then when analyzing word frequency in text, it is inevitable for these connecting words to be the most prevalent. So, in order to analyze text more meaningfully and efficiently, those “stopwords” are often eliminated as part of cleaning. Notice also how stopword removal helps preserve computer memory, which can easily get out of hands if analyzing large volumes of text carelessly.")
        st.write("**Lemmatization or Stemming** : The purpose of this step is the standardization of the different versions of the same word. For instance, let's say we have words: improve, improving, and improved. All three have the same root, but in a different tense. Therefore, if we try to analyze frequencies of the words in a text, each of the three will count as different words. To prevent this from happening, we can lemmatize or stem the words, to reduce them to a shorter, more standard form. Note that while lemmatization reduces each word to a shorter form (“lemma”), which still is a word in a dictionary, in stemming the resultant shorter version (“stem”) may not be a proper word. In the case of the three words here, those would revert to 'improve'. Again, take a moment to appreciate how making words shorter is going to aid preserve memory which in turn will speed up processing and compute time in downstream tasks. ")
        st.write("Below, once the reader starts the cleaning process, the progress bar will show the different stages in which the data is being processed through. Then, for each of the cleaning steps above, with the reader's choice of dream, the reader will be able to see the direct changes made to the dreams!")
        st.write("Have fun playing with the different data cleaning tasks below! You are about to get into something even more interesting once you are done with this.")
            
        result_dc = st.button("Click to Start Data Preprocessing")
        stopword = nltk.corpus.stopwords.words('english')

        if result_dc:
            st.session_state['result_dc'] = True
        try:
            if st.session_state['result_dc']:

                @st.cache_data
                def preprocess(df):
                    my_bar = st.progress(0, text="Dropping Null Values")
                    time.sleep(2)
                    df= df.dropna()
                    my_bar.progress(10, text = "Transforming Date Time Objects")
                    time.sleep(2)
                    df['date'] = [datetime.fromtimestamp(time) for time in df['date']]
                    my_bar.progress(30, text = "Profanity Censor in Progress")
                    time.sleep(2)
                    # df['text'] = [profanity.censor(i) for i in df['text']]
                    my_bar.progress(50, text = "Calculating Length of each Text")
                    time.sleep(2)
                    #calculating length of each dream
                    df['length'] = [len(j) for j in df['text']]
                    my_bar.progress(70, text = "Getting Semi Dataset")
                    time.sleep(2)
                    # if less than or equal to 5th percentile, assign t_f column False
                    df['t_f'] = [True if j > np.percentile(df['length'], 5) else False for j in df['length']]
                    my_bar.progress(90, text = "Making Deep Copy of Semi")
                    time.sleep(2)
                    #only keep t_f == True rows
                    semi = df.loc[df['t_f'] == True, :].reset_index(drop = True).__deepcopy__()
                    my_bar.progress(100, text = "Complete!")

                    return df, semi
                
                df, semi = preprocess(st.session_state['reddit'])

                st.header("Breakdown of Data Cleaning Process")
                st.info("Type in Keyword you would like to see in the Dream" ,icon="ℹ️")
                st.session_state['keyword'] = " " + st.text_input("Keyword:") + " "
                filtered = semi[semi['text'].str.contains(st.session_state['keyword'])]
                
                if "keyword" in st.session_state.keys():
                    st.dataframe(filtered)
                    st.session_state['filtered'] = filtered
                    
                else:
                    st.dataframe(semi)

                st.session_state['row_n'] = int(st.text_input("Type in Index Number of the Dream you would like to examine"))            
                
                def clean(text):
                    # Remove URLs while preserving punctuation
                    text = re.sub(r'https?://\S+|www\.\S+', '', text)

                    # Remove mentions while preserving punctuation
                    text = re.sub(r"@\S+", '', text)

                    # Remove hashtags while preserving punctuation
                    text = re.sub(r"#\S+", '', text)

                    # Remove standalone numbers while preserving punctuation
                    text = re.sub(r"\b[0-9]+\b", '', text)

                    # Remove newlines
                    text = re.sub(r"\n", '', text)

                    # Replace contractions and special characters
                    text = re.sub("\'m", ' am ', text)
                    text = re.sub("\'re", ' are ', text)
                    text = re.sub("\'d", ' had ', text)
                    text = re.sub("\'s", ' is ', text)
                    text = re.sub("\'ve", ' have ', text)
                    text = re.sub("n't", r' not ', text)
                    text = re.sub(" im ", ' i am ', text)
                    text = re.sub(" iam ", ' i am ', text)
                    text = re.sub(" youre ", ' you are ', text)
                    text = re.sub(" theyre ", ' they are ', text)
                    text = re.sub(" theyve ", ' they have ', text)
                    text = re.sub(" weve ", ' we have ', text)
                    text = re.sub(" isnt ", ' is not ', text)
                    text = re.sub(" arent ", ' are not ', text)
                    text = re.sub(" ur ", ' you are ', text)
                    text = re.sub(" ive ", ' i have ', text)
                    text = re.sub("_", '', text)
                    text = re.sub("\"", '', text)
                    text = re.sub(" bc ", ' because ', text)
                    text = re.sub(" aka ", ' also known as ', text)
                    text = re.sub("√©", 'e', text)  # Encoding error for é. Replace it with e
                    text = re.sub(" bf  ", ' boyfriend ', text)
                    text = re.sub(" gf  ", ' girlfriend ', text)
                    text = re.sub(" btw  ", ' by the way ', text)
                    text = re.sub(" btwn  ", ' between ', text)
                    text = re.sub(r'([a-z])\1{2,}', r'\1', text)  # If the same character is repeated more than twice, remove it to one.
                    text = re.sub(' ctrl ', ' control ', text)
                    text = re.sub(' cuz ', ' because ', text)
                    text = re.sub(' dif ', ' different ', text)
                    text = re.sub(' dm ', ' direct message ', text)
                    text = re.sub(" fav ", ' favorite ', text)
                    text = re.sub(" fave ", ' favorite ', text)
                    text = re.sub(" fml ", " fuck my life ", text)
                    text = re.sub(" hq ", " headquarter ", text)
                    text = re.sub(" hr ", " hours ", text)
                    text = re.sub(" idk ", " i do not know ", text)
                    text = re.sub(" ik ", ' i know ', text)
                    text = re.sub(" lol ", ' laugh out loud ', text)
                    text = re.sub(" u ", ' you ', text)
                    text = re.sub("√¶", 'ae', text)  # Encoding error for áe. Replace it with ae
                    text = re.sub("√® ", 'e', text)   # Encoding error for é. Replace it with e
                    text = re.sub("amp amp", "", text)
                    text = re.sub("tl;dr", "too long did not read", text)
                    text = re.sub("buttfuck", "", text)
                    text = text.strip()
                    return text

                def tokenization(text):
                    text = re.split('\W+', text) #split words by whitespace to tokenize words
                    return text

                def remove_stopwords(text):
                    text = [word for word in text if word not in stopword] #remove stopwords in the nltk stopwords dictionary
                    return text

                def lemmatizer(text):
                    nlp = load_nlp()
                    doc = nlp(" ".join(text))
                    
                    # Create list of tokens from given string
                    tokens = []
                    for token in doc:
                        tokens.append(token)

                    text = [token.lemma_ for token in doc]
                    return text                                  #because lemmatizing keeps the context of words alive

                def vectorization(li):                            #create matrix of words and its respective presence for each dream
                    vectorizer = CountVectorizer()   
                    Xs = vectorizer.fit_transform(li)   
                    X = np.array(Xs.todense())
                    
                    return X

                def get_column_name(li):                          #extract each word so that it will be present in corpus as column names
                    vectorizer = CountVectorizer()   
                    Xs = vectorizer.fit_transform(li)   
                    col_names=vectorizer.get_feature_names_out()
                    col_names = list(col_names)

                    return col_names
                
                @st.cache_data
                def extract_array(df):
                    my_bar = st.progress(0, text="Initializing Text Cleaning")

                    clean_text = df['text'].apply(lambda x:clean(x.lower()))         #first clean the text on lower cased list of dreams

                    my_bar.progress(10, text = "Initial Dreams Cleaning Complete")
                    time.sleep(2)

                    tokenized = clean_text.apply(lambda x: tokenization(x))          #tokenize the cleaned text
                    clean_text = tokenized.apply(lambda x: " ".join(x))              #rejoin the words (just in case white space still present)
                            
                    my_bar.progress(30, text = "Dreams Tokenization Complete")
                    time.sleep(2)

                    x_stopwords = tokenized.apply(lambda x: remove_stopwords(x))     #remove stopwords from tokenized list
                                    
                    my_bar.progress(50, text = "Dreams Stopwords Removal Complete")
                    time.sleep(2)

                    lemmatized = [lemmatizer(x) for x in x_stopwords]
                    
                    my_bar.progress(70, text = "Dreams Lemmatization Complete")
                    time.sleep(2)

                    complete = [" ".join(x) for x in lemmatized]               #rejoin the words so it will look like a sentence
                    mapx = vectorization(complete)                                   #start of mapping to corpus
                    name = get_column_name(complete)
                    mapx = pd.DataFrame(mapx, columns = name)
                    mapx.columns = name
                    my_bar.progress(90, text = "Dreams Corpus Complete")
                    time.sleep(2)
                    my_bar.progress(100, text = "Dreams Text Cleaning Complete")

                    return clean_text, tokenized, x_stopwords, lemmatized, complete, mapx

                clean_text, tokenized, x_stopwords, lemmatized, complete, corpus = extract_array(semi)

                st.session_state['clean_text'] = clean_text
                st.session_state['tokenized'] = tokenized
                st.session_state['x_stopwords'] = x_stopwords
                st.session_state['lemmatized'] = lemmatized
                st.session_state['complete'] = complete
                st.session_state['corpus'] = corpus
                st.session_state['semi'] = semi

                def extract_array_sample(ind):
                    with st.form("Original Text"):
                        st.header("Original Text")
                        st.write(st.session_state['semi']['text'][ind])

                        submit_1 = st.form_submit_button("Continue to Initial Cleaning Process")   
                    
                        if submit_1: 
                            st.session_state['submit_1'] = True

                    if st.session_state['submit_1']:
                        with st.form("Initial Data Cleaning"):
                            st.header("Simple Text Cleaning")
                            st.write(st.session_state['clean_text'][ind])

                            submit_2 = st.form_submit_button("Continue to Tokenization")           
                            if submit_2:
                                st.session_state['submit_2'] = True

                    if st.session_state['submit_2']:
                        with st.form("Tokenization"):
                            st.header("Tokenization")
                            st.write(" , ".join(st.session_state['tokenized'][ind][:-1]))

                            submit_3 = st.form_submit_button("Continue to Stopwords Removal")         
                            if submit_3:
                                st.session_state['submit_3'] = True

                    if st.session_state['submit_3']:         
                        with st.form("Stopwords Removal"):
                            st.header("Removing Stopwords")
                            st.write(" ".join(st.session_state['x_stopwords'][ind]))

                            submit_4 = st.form_submit_button("Continue to Lemmatization")  
                            if submit_4:
                                st.session_state['submit_4'] = True

                    if st.session_state['submit_4']:               
                        with st.form("Lemmatization"):
                            st.header("Lemmatization")
                            st.write(" ".join(st.session_state['lemmatized'][ind]))

                            submit_5 = st.form_submit_button("Click to View the final WordCloud!")
                            if submit_5:
                                st.session_state['submit_5'] = True

                    if st.session_state['submit_5']:
                        with st.container():
                            st.header("Resulting Wordcloud")

                            corpus = st.session_state['corpus']
                            token = st.session_state['lemmatized']     
                            tokenized = [list(set(li)) for li in token]

                            #define term frequency (tf) function
                            def tf(corpus, token_set):
                                tf_dict = {}
                                n = len(token_set)
                                row_dict = corpus

                                for word, count in row_dict.items():
                                    tf_dict[word] = count / float(n)
                                
                                return tf_dict

                            #define inverse data frequency (idf) function
                            def idf(documents):
                                n = len(documents)
                                idf_dict = dict.fromkeys(documents[0].keys(),0)

                                for document in documents:
                                    for word, val in document.items():
                                        if val > 0:
                                            idf_dict[word] += 1
                                    
                                for word, val in idf_dict.items():
                                    idf_dict[word] = math.log(n / float(val))

                                    #if one wants to match the sklearn version of the tfidfvectorizor
                                    #idf_dict[word] = math.log((n+1) / (1+float(val)))+1

                                return idf_dict

                            #define tf-idf function
                            def tf_idf(tf, idf):
                                tf_idf_dict = {}

                                for word, val in tf.items():
                                    tf_idf_dict[word] = val * idf[word]

                                return tf_idf_dict

                            #main function to execute all above
                            def main(corpus, tokenized):
                                my_bar = st.progress(0,"Initializing tf-idf calculation")
                                tf_li = []
                                tf_idf_li = []
                                
                                documents = [corpus.iloc[i,:].to_dict() for i in range(corpus.shape[0])]
                                time.sleep(2)

                                my_bar.progress(35, "Calculating tf")
                                for l, r in enumerate(documents):
                                    tf_temp = tf(r, tokenized[l])
                                    tf_li.append(tf_temp)
                                
                                time.sleep(2)
                                my_bar.progress(70, "Calculating idf")
                                idf_dict = idf(documents)

                                time.sleep(2)
                                my_bar.progress(95, "Calculating tf_idf")
                                for t in tf_li:
                                    tf_idf_li.append(tf_idf(t, idf_dict))
                                

                                my_bar.progress(100, "TF-IDF Calculation Complete. Exporting...")

                                return pd.DataFrame(tf_idf_li)
                            
                            
                            st.session_state['tf_idf_df'] = main(st.session_state['corpus'], st.session_state['lemmatized'])
                            
                            tf_idf_mean = st.session_state['tf_idf_df'].describe().iloc[1,:].tolist()

                            t_f = [False if z < np.mean(np.nonzero(np.array(tf_idf_mean))) else True for z in tf_idf_mean]
                            not_words = [j for e,j in enumerate(st.session_state['tf_idf_df'].columns) if t_f[e] == False]

                            wordcloud(st.session_state['clean_text'], lim=100, stopword = not_words ,collocation_threshold = 10)
                        
                        st.info("Next click on the next tab on the left to move on to the Part of Speech Tagging Section!" ,icon="ℹ️")

                            

                extract_array_sample(st.session_state['row_n'])  
        except:
            st.warning("Please Complete the Previous Stage Before Moving On")

    except:
        st.warning("Please Complete the Previous Stage Before Moving On")
                
########################################################################################
###############       POS Tagging / NER Visualization  page      #######################
########################################################################################

def part_of_speech_tag():
    st.title("Part of Speech Tagging (POS)")
    try:
        st.info(f"Chosen Dream: Dream {st.session_state['row_n']}" ,icon="ℹ️")
        nlp = load_nlp()
        st.write("Part of Speech Tagging (POS) is a classification method, where each word in a sentence is given a particular part of speech depending on the position and context within the sentence structure. The method was first introduced as a measure to reduce the ambiguity of word implications in a sentence for machine translation purposes. In other words, POS Tagging allows for machines to recognize the way in which the word is utilized. For example, the word “run” in the two sentences:")
        st.write("“I like to run” and “I went for a run”")
        st.write("has two separate meanings. The former “run” is a verb that pertains to the action of running. The latter “run” pertains to the activity of running, a noun. However, in the sense of machine learning models, the two usages of “run” in both contexts are not distinguishable causing ambiguity.")
        st.write("So there has to be a way for the machine to understand the different ways the same word is utilized in different contexts! Therefore we introduce the POS Tagging.")        
        result = st.button("Click to Start POS Tagging")

        @st.cache_data
        def pos_preprocess(df):
            tag_dict = {"word" :[], "tag":[]}

            for e,i in enumerate(df):
                sent = nlp(i)
                for j in sent:
                    tag_dict['word'].append(j.text)
                    tag_dict['tag'].append(j.tag_)

            tag_df  = pd.DataFrame(tag_dict)

            return tag_df
        
        complete_load = st.session_state['complete']
        tag_df = pos_preprocess(complete_load)

        if result:
            st.session_state['show'] = True

            cola, colb = st.columns(2)
            with cola:
                st.header("POS Tag List")
                st.dataframe(pd.read_csv("https://gist.githubusercontent.com/veochae/447a8d4c7fa38a9494966e59564d4222/raw/9df88f091d6d1728eb347ee68ee2cdb297c0e5ff/spacy_tag.csv"))
            with colb:
                st.header("What is this Table?")
                st.markdown("The table on the left is the Spacy pacakge defined Part of Speech Tags. Each acronym stands for a particular part of speech, and essentially, each word is tagged with one of the tags in the list on the left!")

            @st.cache_data
            def barplot(x):
                t = np.unique(x, return_counts = True)
                s = np.argsort(t[1])

                x = t[0][s][::-1]
                y = t[1][s][::-1]

                fig6 = px.bar(x = x, 
                            y = y, 
                            labels = dict(x = "Part of Speech", y = 'Count'),
                            title = "Count of Part of Speech in the Entire Corpus") 

                fig6.update_layout(xaxis={'categoryorder':'total descending'})   
                    
                st.plotly_chart(fig6,theme="streamlit", use_container_width=True)    

            with st.container():
                st.write("Next with the full list of POS Tags throughout all the Dreams that we have collected, we plot a barplot to see which Tags were heavily uitilized in the Dreams. As one can see from the barplot, Nouns were mostly utilized since Dreams have objects that have to be described in detail. Then, Adverbs and different tenses of verbs were heavily utilized in describing the Dreamers' actions during the dream.")
                barplot(tag_df['tag'])

        # try:
        if st.session_state['show']:
                st.write("Now that we know that each word can be understood by the machine, how about sentences? Can machines now understand full sentences?")
                st.write("To help ease the understanding of why we need this, we can give Chat-GPT as an example. To the human brain, when we observe the two statements: ")
                st.write("“I use Chat-GPT”, “Do you use Chat-GPT?” ")
                st.write("We already know which one of the two statements is a question. Not only because of the question mark on the second statement, but because it is a sentence that starts with an auxillary ”Do” and a pronoun as the target of asking the question. Obviously, humans do not actively process the part of speech for each and every sentence one encounters, but how about when the machine has to learn sentence structure? Just like the young versions of ourselves first learning how to comprehend the sentence structure, machine has to learn the sentence structures of English as well. Now, we can use the individual POS Tags as a sequence in order to essentially create a formula of sentence structures. With the example above, because")
                st.write("auxillary + pronoun + verb + … ")
                st.write("is the sequential order of POS tags in the given sentence, the machine will now recognize that this sentence is a question.")
                st.write("As such, POS tagging not only helps machines understand the individual usage of singular words, but also provides an even more powerful tool when used on an aggregated level!")
            
                df = st.session_state['semi']

                with st.container():
                    temp = np.str.split(df['text'][st.session_state['row_n']], ".")[0] + "."
                    model = "en_core_web_sm"

                    st.title("POS Taggging Visualization")
                    st.write("By now, you have probably seen how long each dreams are. So to show all the dreams and its POS Visualization, it is hard to comprehend because it is so big! So below, we will show how POS tagging works visually for the frist sentence of the dream that you have chosen!")
                    st.write("Also, the text section right below here is interactive! Please enter any kind of text or sentences you would like to examine and it will adjust the visualization according to your provided sentence. Enjoy!")
                    text = st.text_area("Text to analyze", temp, height=200)
                    doc = spacy_streamlit.process_text(model, text)

                    spacy_streamlit.visualize_parser(doc)

                st.info("Next click on the next tab on the left to move on to the Named Entity Recognition Section!", icon="ℹ️")
    except:
            st.warning("Please Complete the Before Step Afore Starting The Current Stage")    

########################################################################################
#############################       namee entity recognition  page      #################################
########################################################################################

def named_entity_recognition():
    st.title("Named Entity Recognition")
    
    try:
        st.info(f"Chosen Dream: Dream {st.session_state['row_n']}" ,icon="ℹ️")
        if st.session_state['show']:
                st.write("As the next step of translating human language to machine comprehensible context, we go through the named entity recognition. Well first, we have to know what Named Entity is! ")
                st.write("Named Entities are words or collection of words that signify a particular subject in a given text. In essence, the particular subjects would entail names, locations, companies, products, monetary values, percentages, time, etc. The key difference from the POS Tagging to Named Entity Recognition is that it provides more context to the sentence the algorithm is trying to understand. ")
                st.write("For instance, let’s take the example of two sentences below:")
                st.write("“I like Google” and “I like Wellesley”")
                st.write("From the POS tagging, the machine learning algorithm understands that Google and Wellesley are nouns. However, it only recognizes that the two words are nouns, but not what the each word entails. Named Entity Recognition will flag the two words into Company and Location respectively. That way, the machine can now have a contextualized understanding of the sentence that one is a statement about a company, and the counterpart about a location. ")
                st.write("So how is this used in real life you may ask! There are countless possible usages of Named Entity Recognition, but one of the most prominent used cases would be Netflix’s recommendation system. When you watch a show or movie on Netflix, based on the description of the show, Netflix can extract the entities in the description and recommend another entertainment piece that has the most similar entities in its description. Other used cases can be a simpler one where we can summarize a unstructured text data (such as a news article) to a structured format. In other words, instead of reading the entire article, NER allows for extraction of the 5Ws: Who, What, Why, When and Where.")
                st.write("Now, with that being said, let’s try this new technique on the dream that you have chosen from the previous section!")
            
                df = st.session_state['semi']

                with st.container():
                    temp = df['text'][st.session_state['row_n']]
                    model = "en_core_web_sm"

                    st.title("NER Visualization")
                    st.write("Just like the POS Visualization, this NER visualization is also interactive! Type in any sentence, preferably ones with noticeable entities, to see how the visualization interacts with your input!")
                    text = st.text_area("Text to analyze", temp, height=200)
                    doc = spacy_streamlit.process_text(model, text)

                    spacy_streamlit.visualize_ner(doc,
                                                show_table=False
                                                    )
                    
                st.info("Next click on the next tab on the left to move on to the TF-IDF Section!")


    except:
            st.warning("Please Complete the Before Step Afore Starting The Current Stage")    


########################################################################################
#############################       TF-IDF  page      ##################################
########################################################################################
def tf_idf():

    tf_latex = r'\text{TF}(w, d) = \frac{\text{Count of } w \text{ in } d}{\text{Total number of words in } d}'
    idf_latex = r'\text{IDF}(w) = \log\left(\frac{N}{n_w}\right)'
    tf_idf_latex = r'\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)'
    text = r"""\text{Number of Words}: N \\ \text{Number of documents containing } w: n_x"""
    
    st.title("TF-IDF Analysis")
    try:
        st.info(f"Chosen Dream: Dream {st.session_state['row_n']}",icon="ℹ️")    
        st.write("Ever wondered how LinkedIn scans your resume or how Google recommendation works?")
        st.write("Certainly, there are many other advanced methods that take place in both of the tech giants' machine learning methods, but in their core, TF-IDF exists.")
        st.write("TF-IDF stands for Term Frequency and Inverse Document Frequency, and it's a numerical representation used in NLP to understand the importance of words in a document or collection of documents. Let's break it down piece by piece to what TF and IDF each does:")
        st.write("**Term Frequency (TF)**: Term Frequency in the simplest sense measures how often a word appears in a document. It takes the document, and counts how many times each word is appearing in the specific document. The mathematical representation used in this following app is as follows:")
        st.latex(tf_latex)
        st.write("**Inverse Document Frequency (IDF)**: Inverse Document Frequency, unlike the TF, takes all the documents in hand. Not specific to a singular document, but the entire set of documents you are trying to analyze. By providing this measure, we can see which words are less common throughout the documents. So in essence, IDF allows us to distinguish which words were rather specific to each document!")
        st.latex(text)
        st.latex(idf_latex)
        st.write("**TF-IDF**: TF-IDF is the amalgamation of TF and IDF as you can tell by the name! By using the equation below, TF-IDF shows how important a word is in a document in comparison to when used in another document. For instance, when we search for the word **entrepreneurship**, a document pertaining to Babson College will have a higher TF-IDF score for the word in comparison to a document about Olin College, because entrepreneurship is more relevant in the document for Babson!")
        st.latex(tf_idf_latex)
        st.write("Check out this [link](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for more information about the equations!")
        st.write("Now, let's start the below section to explore TF-IDF!")        
        with st.expander(f"Click Here to View the Selected Dream "):
            st.write(f"""{st.session_state['semi']['text'][st.session_state['row_n']]}""")

        result_ti = st.button("Click Here to start TF-IDF")

        if result_ti:
            st.session_state['result_ti'] = True
        try:
            if st.session_state['result_ti']:
                def barplot(tf_idf_df, number_of_words):
                    if len(tf_idf_df.iloc[st.session_state['row_n'],:].tolist()) < number_of_words:
                        number_of_words = len(tf_idf_df.iloc[st.session_state['row_n'],:].tolist())
                    else:
                        pass
                    rendered_dream = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n'],:].sort_values(axis = 0, ascending = False)[:number_of_words]})
                    words = rendered_dream.index.tolist()
                    rendered_dream['words'] = words

                    fig = px.bar(rendered_dream,
                                    x='words', 
                                    y='values', 
                                    title=f"Dream {st.session_state['row_n']} tf-idf score words",
                                    labels = dict(words = "Words", values = 'TF-IDF Score'))
                    st.plotly_chart(fig,theme="streamlit", use_container_width=True)   

                def barplot_2(tf_idf_df, number_of_words, number_of_words2):
                    if len(tf_idf_df.iloc[st.session_state['row_n'],:].tolist()) < number_of_words:
                        number_of_words = len(tf_idf_df.iloc[st.session_state['row_n'],:].tolist())
                    else:
                        pass

                    if len(tf_idf_df.iloc[st.session_state['row_n_2'],:].tolist()) < number_of_words:
                        number_of_words2 = len(tf_idf_df.iloc[st.session_state['row_n_2'],:].tolist())
                    else:
                        pass

                    rendered_dream = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n'],:].sort_values(axis = 0, ascending = False)[:number_of_words]})
                    words = rendered_dream.index.tolist()
                    rendered_dream['words'] = words

                    rendered_dream_2 = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n_2'],:].sort_values(axis = 0, ascending = False)[:number_of_words2]})
                    words_2 = rendered_dream_2.index.tolist()
                    rendered_dream_2['words'] = words_2          

                    fig = make_subplots(rows=1, cols=2)

                    fig.add_trace(go.Bar(x = rendered_dream['words'],
                                        y = rendered_dream['values'],
                                        name = f"Dream {st.session_state['row_n']}"),
                                        row = 1, col = 1)
                    
                    fig.add_trace(go.Bar(x = rendered_dream_2['words'],
                                        y = rendered_dream_2['values'],
                                        name = f"Dream {st.session_state['row_n_2']}"),
                                    row = 1, col = 2)         
                    
                    fig.update_layout(
                                        title="TF-IDF Side by Side Barplot",
                                        xaxis_title="Words",
                                        yaxis_title="TF-IDF Values",
                                        legend_title="Dreams"
                                        # font=dict(
                                        #     family="Courier New, monospace",
                                        #     size=18,
                                        #     color="RebeccaPurple"
                                        # )
                                    )
                        
                    st.plotly_chart(fig,theme="streamlit", use_container_width=True)   

                barplot(tf_idf_df = st.session_state['tf_idf_df'], number_of_words = 10)
                change = 2

                if change == 2:
                    st.success("If you wish to change the **first** Dream or Keyword, please go back to the Data Cleaning Section of the App.")

                    st.info("Choose another dream that you would like to examine" ,icon="ℹ️")
                    st.dataframe(pd.DataFrame(st.session_state['semi']))
            
                    st.session_state['row_n_2'] = int(st.text_input("Second Dream Index:"))

                    try:
                        barplot_2(tf_idf_df = st.session_state['tf_idf_df'], number_of_words = 10, number_of_words2 = 10)

                        col1,col2 = st.columns(2)
                        with col1:
                            with st.expander(f"View Dream {st.session_state['row_n']}"):
                                st.write(f"""{st.session_state['semi']['text'][st.session_state['row_n']]}""")                        
                        with col2:
                            with st.expander(f"View Dream {st.session_state['row_n_2']}"):
                                st.write(f"""{st.session_state['semi']['text'][st.session_state['row_n_2']]}""")
                        
                        st.info("Next click on the next tab on the left to move on to the Dream Summarization and Continuation Section!" ,icon="ℹ️")
                    except:
                        st.warning("Please Input the Second Dream Row Number")
                else: pass
        except:
            st.warning("Please Press to Start!")
    except:
        st.warning("Please Complete the Previous Step Before Moving On")

########################################################################################
#############################       Setup for OpenAI      #################################
######################################################################################## 
def set_up_openai():
    st.title("Setting Up your Open AI API")
    st.video("https://youtu.be/VMjJ4BrYVaE")
    with st.form("open_ai_cred"):
        key_open = st.text_input("OpenAI API Key")
        submitted = st.form_submit_button("Submit")   
        if submitted:
            st.session_state['openai_key'] = key_open
            st.success("Your API Key has been Processed!")

########################################################################################
#############################       Sentiment Analysis      #################################
######################################################################################## 
def sentiment_analysis():
    st.title("Sentiment Analysis")
    try:
        st.write("In recent days, companies ask for more detailed reviews about their products than ever before. Ever wondered why?")
        st.write("It's because using Sentiment Analysis, the companies can start to realize how the customers **feel** about their products!")
        st.write("In the earlier days of the sentiment analysis, it was quite simple. You would classify a piece of text as **positive**, **negative**, or **neutral**. We won't dive into too much detail about how that has been done, but if you are curious, checkout this [link](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)!")
        st.write("Now, coming back to the modern days, the sentiment analysis have evolved into something more specific and granular: **Emotion Analysis**.")
        st.write("Instead of just figuring out whether the customers' reactions were positive or negative, we start to look at multiple emotions such as: fear, joy, happiness, surprise, love, anger, sadness ,etc.")
        st.write("The pretrained model that we use for the purpose of this exercise is from hugging face, using more than 100,000 tweets as its training data. Essentially, all of those tweets were tagged with different emotions. And the neural networks is trained to recognize different speech patterns and words that comprises the emotions that has been prelabelled. Now, having **learned** about human emotions over 100,000 text files, the model can start to predict what emotion the writer of the text is trying to show.")
        st.write("So, without further explanation, let's see what kind of emotion the dream you have chosen shows!")

        with st.form("sentiment_analysis"):
            st.info(f"Chosen Dream: Dream {st.session_state['row_n']}",icon="ℹ️")
            with st.expander(f"Click Here to View the Selected Dream "):
                dream = st.session_state['semi']['text'][st.session_state['row_n']]
                st.write(dream)        
            submitted_sentiment = st.form_submit_button("Let's Begin!")   
        
        if submitted_sentiment:
            try:
                openai.api_key = st.session_state['openai_key']

                try:     
                    summary = summarize_dream("Summarize this dream to less than 280 words from the storyteller's perspective \n" + "Dream: " + dream, length = 280)
                except:
                    st.warning("This Error is either: 1. Do not have enough API balance 2. Not the correct API Key")

                classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
                prediction = classifier(summary)
                emotion = [x['label'] for x in prediction[0]]
                score = [y['score'] for y in prediction[0]]

                st.session_state['emotion'] = emotion[score.index(np.max(score))]

                fig10 = make_subplots(rows=1, cols=1)

                fig10.add_trace(go.Bar(x = emotion,
                                        y = score,
                                        name = f"Dream {st.session_state['row_n']}"))

                fig10.update_layout(
                                    title="Sentiment Classification Results",
                                    xaxis_title="Criteria",
                                    yaxis_title="Sentiment Scores",
                                    legend_title="Dreams"
                                    # font=dict(
                                    #     family="Courier New, monospace",
                                    #     size=18,
                                    #     color="RebeccaPurple"
                                    # )
                                )    

                st.plotly_chart(fig10,theme="streamlit", use_container_width=True) 
            except:
                st.warning("Either OpenAI Key is incorrect or you have chosen inappropriate dream")
    except:
        st.warning("Please Complete the Previous Step Before Moving On")
########################################################################################
#############################       Dream Summarization + Continuation      #################################
######################################################################################## 
        
def summary_continue():
    st.title("Dreams with GPT") 

    try:
        st.write("Now we are at the final stage and the most modern stage of NLP: **Generative Pre-trained Transformers** ")
        st.write("You are probably most familiar with Chat GPT as it is rapidly utilized all over regardless of the industries!")
        st.write("GPTs, most simply put, are neural networks that tries to emulate the human brain. Remember how neural networks work? It connects different nodes to resemble the human neurons and each of their interactions. Now, GPT 3.5 for instance, try to create 175 billion parameters. Although recreating a human brain would take trillions of parameters, GPT is yet the closest computer program that is closest to emulating the human brain.")
        st.write("And as a byproduct of of the textual GPTs, there also is DALL-E, which is image generative AI. DALL·E is like a creative image generator with a touch of AI magic. It can take written descriptions and turn them into unique images. Imagine describing an idea in words, and DALL·E brings it to life as an artwork.")
        st.write("Using the Chat GPT 3.5 Davinci and DALL-E, below we summarize, expand and visualize the dream that you have been observing throughout this app.")

        openai.api_key = st.session_state['openai_key']    

        with st.form("asdf"):
            st.header("Original Text")
            try:
                with st.expander(f"Click Here to View the Selected Dream "):
                    dream = st.session_state['semi']['text'][st.session_state['row_n']]
                    st.write(dream)
            except:
                pass
            dream_submit = st.form_submit_button("Proceed to Summarization and Continuation") 
            if dream_submit:
                st.session_state['dream_submit'] = True

        if st.session_state['dream_submit']: 
            if len(dream) <= 280:
                length = int(np.ceil(len(dream) * 0.3))
            else:
                length = 280     
            # try:     
            summary = summarize_dream("Summarize this dream to less than 280 words from the storyteller's perspective \n" + "Dream: " + dream, length = length)
            # except:
                # st.warning("This Error is either: 1. Do not have enough API balance 2. Not the correct API Key")
            continuation = summarize_dream("Tell me what happens after this story in the first person point of view: \n" + dream, length = 280)

            st.header("Dream Summary")
            st.write(summary)

            st.header("Dream Continuation")
            st.write(continuation)

            st.header("Dream Visualization")

            continued = False
            st.session_state['artist'] = st.selectbox(
                "What artist would you like to emulate?",
                ("Salvador Dali", "Edvard Munch", "Gustav Klimt", "Vincent Van Gogh", "Edward Hopper"),
                index = 0,
                placeholder = "Please select an artist")

            if isinstance(st.session_state['artist'],str):
                continued = True
            
            if continued:
                dalle = summarize_dream("Summarize this dream into one sentence to be inputted into DALLE: \n"+dream, length = 100)
                time.sleep(5)
                response = openai.Image.create(
                            prompt=f"Produce a painting in the style of '{st.session_state['artist']}' resembling '{st.session_state['emotion']}' about the following scenario '{dalle}'",
                            n=1,
                            size="1024x1024")
                
                st.image(response['data'][0]['url'])
                dream_submit = False
            else:
                st.warning("Please select an artist")
    except: 
        st.warning("Please Complete the Previous Step Before Moving On")

########################################################################################
#############################       Data Download      #################################
########################################################################################        

def data_download():
    st.title("Download Datasets")

    try:
        titles = ['clean_text', 'tokenized', 'x_stopwords', 'lemmatized', 'complete']
        
        col1,col2,col3,col4,col5,col6,col7 = st.columns([1,1,1,1,1,1,1])

        for k,context in enumerate(titles):
            if k <= 4:
                x = pd.DataFrame({'title': pd.DataFrame(st.session_state['semi'])['title'],
                    context: st.session_state[context]}).reset_index()
                # x = x.drop("index", axis =1)

                vars()[f'{context}_csv'] = convert_df(x)

                with vars()[f'col{k+1}']:
                    st.download_button(
                    f"{context}",
                    vars()[f'{context}_csv'],
                    f"{context}.csv",
                    "text/csv",
                    key=f'download-csv-{k+1}'
            )
            else: pass

        with col6:
            corpus_csv = convert_df(st.session_state['corpus'])

            st.download_button(
            "corpus",
            corpus_csv,
            "corpus.csv",
            "text/csv",
            key=f'download-csv-7'
    )        
            
        with col7:
            st.download_button(
            "semi_cleaned",
            convert_df(st.session_state['semi']),
            "semi_cleaned.csv",
            "text/csv",
            key=f'download-csv-8'
    )
    except: st.warning("You must complete the Data Cleaning Process to View the Page")

########################################################################################
#############################       About Creators      #################################
########################################################################################        

def about_creators():
    st.title("About the Creators")
    st.success("The app has been ideated and created by Professor Davit Khachatryan and Veo Chae")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Davit Khachatryan")
        st.info("**Email:** dkhachatryan@babson.edu")
    with col2:
        st.header("Dong Hyun (Veo) Chae")
        st.info("**Email:** veochae@gmail.com")



########################################################################################
#############################       sidebar  page      #################################
########################################################################################

page_names_to_funcs = {
    "Introduction": introduction,
    "Data Collection": data_collection,
    "Data Cleaning": data_cleaning,
    "Part of Speech Tagging": part_of_speech_tag,
    "Named Entity Recognition": named_entity_recognition,
    "TF-IDF": tf_idf,
    "OpenAI API Setup": set_up_openai,
    "Sentiment Analysis": sentiment_analysis,
    "Dream Summary and Continuation": summary_continue,
    "Data Download": data_download,
    "About the Creators": about_creators
}

demo_name = st.sidebar.selectbox("Please Select a Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


########################################################################################
#############################       lda  page      #################################
########################################################################################

# def lda():
#gensim
# import gensim
# from gensim.utils import simple_preprocess
# import gensim.corpora as corpora
# from gensim.models.coherencemodel import CoherenceModel

#ldavis
# import pyLDAvis.gensim
# import pyLDAvis
#     st.title("Latency Discriminant Analysis")


#     token = st.session_state['lemmatized']   
#     #put the lemmatized dreams into list
#     tokenized = [li for li in token]

#     # Create Dictionary
#     id2word = corpora.Dictionary(tokenized)

#     # Create Corpus
#     texts = tokenized

#     # Term Document Frequency
#     corpus = [id2word.doc2bow(text) for text in texts]     

#     st.write("Calculating the Optimal Number of Topics for LDA model")
#     try:
#         maximum = int(st.text_input("Choose Maximum Number of Topics of Observance"))

#         @st.cache_data
#         def coherence_tuning(max_topics):
#             # number of topics
#             coherence = []
#             my_bar = st.progress(0, "Start of Coherence Measurement")
#             time.sleep(3)

#             for topic in range(3,max_topics+1):
#                 # Build LDA model
#                 lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                                             id2word=id2word,
#                                                             num_topics=topic, 
#                                                             random_state=100,
#                                                             update_every=1,
#                                                             chunksize=10,
#                                                             passes=2,
#                                                             alpha='auto',
#                                                             per_word_topics=True)

#                 cm = gensim.models.coherencemodel.CoherenceModel(
#                                                                 model=lda_model, 
#                                                                 corpus = corpus, 
#                                                                 coherence='u_mass')  
                
#                 coherence.append(cm.get_coherence())
#                 my_bar.progress((1/(max_topics - 2))*(topic-2) ,f"Model with Topic Count {topic} complete")
#                 time.sleep(1)

            

#             fig = px.line(x=range(3,max_topics+1), 
#                             y=coherence, 
#                             title='Coherence Measure for Each Number of Topic',
#                             labels = dict(x = "Topic Count", y = 'U-Mass Coherence Measure'))
#             st.plotly_chart(fig,theme="streamlit", use_container_width=True)  

#             return min(coherence), coherence.index(min(coherence))

#         minimum, min_indx = coherence_tuning(maximum)

#         st.write(f"The best model with the lowest U-MASS Coherence Measure of {round(minimum,3)} is {3+min_indx} Topics")

#         visual_top = int(st.text_input("Choose the Final Number of Topics for Visualization"))

#         lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=visual_top, 
#                                             random_state=100,
#                                             update_every=1,
#                                             chunksize=10,
#                                             passes=2,
#                                             alpha='auto',                                                
#                                             per_word_topics=True)

#         # pyLDAvis.enable_notebook()
#         vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word,  mds='mmds')

#         pyLDAvis.save_html(vis, 'lda.html')

#         st.title("LDA Model Visualization")
#         from streamlit import components
#         with open('./lda.html', 'r') as f:
#             html_string = f.read()
#         components.v1.html(html_string, width=1700, height=1000, scrolling=True)
    
#     except:
#         print("Input a Valid Number for Number of Topics")


