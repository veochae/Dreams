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

#common add ons
import pandas as pd
import numpy as np
import nltk
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from datetime import datetime, date
from sklearn.feature_extraction.text import CountVectorizer

#streamlit
import spacy_streamlit
import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie

#plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

#gensim
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

#ldavis
# import pyLDAvis.gensim
# import pyLDAvis

#other pacakges
from better_profanity import profanity
from pywsd.utils import lemmatize_sentence

#huggingface
from transformers import pipeline

#openai
import openai

#tensorflow
import torchvision
import torch
########################################################################################
#############################       required functions     #############################
########################################################################################
warnings.filterwarnings('ignore')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('punkt')

###################### lottie file extraction
def load_lottieurl(url):
    r  = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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
                                                    'text': profanity.censor(post['data']['selftext']).replace("*",""),
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
                                                    'text': profanity.censor(post['data']['selftext']).replace("*",""),
                                                    'date': post['data']['created']},index=[0])], ignore_index= True)

            latest_key = post['kind'] + '_' + post['data']['id']

            if req * 15 <= 100:    
                my_bar.progress(req *15, text = f"{df.shape[0]} Dreams Collected")
            else:
                my_bar.progress(100, text = f"{df.shape[0]} Dreams Collected")

            if len(df) >= 985:
                latest = df.tail(1)['date'][df.tail(1)['date'].index[0]]
                print("Data Collection Target Reached")
                print(f'{len(df)} rows collected')
                print(f'latest subreddit date: {datetime.fromtimestamp(latest)}')
                return df, res.json()['data']['children'][1]

    else: 
        print("Date Limit Reached")
        print(f'{len(df)} rows collected')
        print(f'latest subreddit date: {datetime.fromtimestamp(latest)}')
        return df
    

########################################################################################
#############################       introduction page      #############################
########################################################################################

def introduction():
    l1 = load_lottieurl("https://lottie.host/0e7f8667-876c-4470-9509-1938d325dbc7/1z7IT8acx2.json")
    col1, col2 = st.columns([4,1])
    with col1:
        st.title("Analyzing Dreams using NLP")

    with col2:
        st_lottie(l1, key = "l1", height = 100, width = 130)

########################################################################################
#############################       data collection page      ##########################
########################################################################################

def data_collection():
    st.title("Data Collection")
    st.write("In order to first collect data for this analysis, we turn our attention to one of the most prominent social media source Reddit. Reddit has been created in 2005 with the purpose of creating an online environment in which users can freely submit images, posts or links for share, and have threaded discussions on various topics. As one may readily be knowledgeable, Reddit has a unique system called Threads. Threads are essentially topics of discussion, where the discussion actually takes place. Today, Reddit has 2.8 Million threads, being uitilzed as one of the hubs of online discussion.")
    st.write("One of the most interesting threads in Reddit is called 'Dreams'. It is a community where users share their dreams and ask for interpretations or simply for the sake of joy of sharing.")
    st.write("To collect the dreams posted by the Reddit users, we first access the Reddit API. The below will serve as a guideline for the readers to gain access to the Reddit Developer's account.")
    st.write("THIS IS SPACE FOR REDDIT DEV ACCOUNT INFROMATION")
    st.write("THIS IS SPACE FOR REDDIT DEV ACCOUNT INFROMATION")
    st.write("Now that we have gained access to the Reddit developer's account, we access the Reddit API in order to gather Dreams that will be utilized for the analysis. The Reddit thread that we are going to be utilizing is r/Dreams, which can be easily searched on search engines for viewing purposes.")
    st.write("In the below text boxes, please input your Reddit information in order to collect the Dreams. Your personal information nor any of the information will be accessed by the creators nor the streamlit application platform.")

    st.write("The process of Data Collection follows the below details: ")
    st.write("1. Your authentication is granted with correct Client Id, Secret Key, Username, and Password. This implies that Reddit knows who is accessing their database and can identify whether you have access to the data of observance. If you do not input the correct credentials, your requests will be denied.")
    st.write("2. With the correct credentials approved by Reddit, now we start collecting the Dreams. Majority of the major platform APIs prevent users from extracting large quantites of data at once. This is in order to prevent injection of malware viruses into the system, as well as to prevent data mining using a data bot. In order to constrain such possibilities, Reddit has placed a maximum number of data that can be collected at each run of request for data. Thus, to not manually rerun and append data each and every run, the script embeded in this app will take short 'time-off' after each run in order to not be restricted by Reddit data collection regulations. For each run, the amount of collected data will be displayed in the progress bar.")
    st.write("3. Contrary to what users may believe, the raw data that is collected from Reddit is in json format. For clarity, json file is a nested dictionary format, where all infromation is stored like a hierarchical tree, not a dataframe. Thus, we select only portions of the json data that is required for this anlaysis and create a dataframe.")
    st.write("4. After the intial raw data collection process, the embedded script performs initial cleaning on the dataset. This process includes the rudimentary process such as dropping Null values and profanity checks.")

    

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

            reddit, json_file = reddit_data(time_wanted, headers)

            my_bar = st.progress(0, text="Initiating Data Preprocessing")

            my_bar.progress(40, "Dropping Empty Observations") 
            st.session_state['reddit'] = reddit.dropna()

            # reddit['text'] = [profanity.censor(i) for i in reddit['text']]

            my_bar.progress(80, "Converting pandas dataframe to CSV")

            time.sleep(3)
            my_bar.progress(90, "Generating Previews")
            time.sleep(3)
            my_bar.progress(100, "Job Complete")

            st.write("As mentioned in the process of collecting data from Reddit API, the inital format of the data collected is in json. The below is a sample look at what the raw data looks like. To best understand how json works, think of the folder directories in your local computers. Within your Desktop folder, the reader would have a folder for each class the reader takes. And within each class folder, the reader would have different assignment folders, containing assignments completed. As such json divides information in a hierarchical order: the deeper nested values are specific information pertaining to the encompassing information. Please press on the rotated green triangle below to assess the json file.")
            st.json(json_file, expanded= False)

            st.write("Finally, the below is the dataframe of the extracted json file. From the json file, subreddit thread name, the title of the post, the dream, and the date at which the post was made in unicode text is extracted.")
            st.dataframe(reddit.head(30))

            st.write("As one can see, the dataframe is a much cleaner table look at the data that is easier for the readers to injest. With such thought in mind, one might wonder, 'Why do we ever use json file then?'. And yes, there are obvious pros and cons to the different file formats. For instance, as mentioned before, dataframe is extremeley effective when it comes to its structures. The data can be accessed by its row and column index and it is easier to manipulate. However, because of its structured nature, the file size can become exponetially large as the number of observations or features increase. Further, pandas dataframe stores various meta data about the data, such as the data type, index, class, etc. However, the json format only stores the text values of the data. Therefore, it is a structured word file that can be interpreted in hierarchical order when imported into an IDE. This saves tremendous amount of space when it comes to storing large datasets. And because most of the times the data in APIs are extremely large, json format is often chosen.")
        
        except KeyError:
            st.warning("Please enter correct Reddit Credentials", icon="⚠️")

########################################################################################
#############################       data cleaning  page      ###########################
########################################################################################

def data_cleaning():
    st.title("Data Manipulation")
    st.write("With the raw dataset in hand, now we move on to the critical stage of analysis: Data Manipulation.")
    st.write("As one may be aware, different data types require various data cleaning process. For instance, numeric values may require changing the data type to its correct state, normalization or standarization, and more. Further, categorical variables will need one-hot-encoding or categorical type transformation. In the case of text data, the cleaning process is quite arduous and has various versions, which are stated below.")
    st.write("**Basic Cleaning** : Basic cleaning is also known as regex cleaning. Regex are regular expressions that allows you to match the sequence of pattern in string values. For instance, when you are analyzing a dream, the raw data may contain the source data url within its text. Or, it may also contain abbrevaitions or slang words that needs to be corrected for the NLP packages to correctly injest. Thus, in order to correct the texts into ML injestable format, basic cleaning process is required.")
    st.write("**Tokenization** : Tokenization is the process of segmenting the text by words. When the raw data is injested into python, python does not recognize the text by words, but by character. Thus, the tokenization allows python to understand that the cleaning process that we are going through is based on words, not chracters. Thus, we split the sentences by words which is called tokens.")
    st.write("**Stopwords Removal** : In writing a sentence, the most repeated words are prepositions and be verbs that connect words into sentences. Thus, in analyzing word frequency in a sentence, it is inevitable for these connecting words to be the most prevalent. Thus, in order to get a more accurate and unhindered result from our analysis, we remove these stopwords.")
    st.write("**Lemmatization** : Lemmatization is a process of grouping different formats of the same rooted words. For instance, let's say we have words: improve, improving, and improved. All three words have the same root, but in a different tense. Therefore, if we try to analyze frequencies of the words in a sentence, all three will count as a different words. In order to prevent this from happening, we lemmatize the words, reverting the words back to its origin. In the case of the three words here, it will revert back to 'improv'.")
    st.write("**Corpus** : Corpus in its simplest sense is a dictionary. From the dataset that we have gathered in the previous steps, corpus takes all possible distinct words from the dataset and makes it into features. Then, instead of the actual sentences, each observation will now contain how many times the word appears in a sentence.")
    st.write("Below, once the reader starts the cleaning process, the progress bar will show the different stages in which the data is being processed through. Then, by selecting different radio buttons, one will be able to see the different results of each cleaning process.")

    # try:
    result_dc = st.button("Click to Start Data Manipulation")

    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    stopword = nltk.corpus.stopwords.words('english')
    wn = nltk.WordNetLemmatizer()

    if result_dc:
        st.session_state['result_dc'] = True
        # try:
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

        st.dataframe(semi)
        st.session_state['row_n'] = int(st.text_input("Type in Index Number of the Dream you would like to examine"))            

        def clean(text):
            text = re.sub('https?://\S+|www\.\S+', '', text) #replace website urls
            text = re.sub(r"@\S+", '', text) #replace anything that follows @
            text = re.sub(r"#\S+", '', text) #replace anything that follows #
            text = re.sub(r"[0-9]", '', text) #replace numeric
            text = re.sub(r"\n", '', text) #replace new line 
            text = re.sub("\'m", ' am ', text) 
            text = re.sub("\'re", ' are ', text) 
            text = re.sub("\'d", ' had ', text)
            text = re.sub("\'s", ' is ', text)
            text = re.sub("\'ve", ' have ', text)
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
            text = re.sub("√©", 'e', text) #encoding error for é. replace it with e
            text = re.sub(" bf  ", ' boyfriend ', text)
            text = re.sub(" gf  ", ' girlfriend ', text)
            text = re.sub(" btw  ", ' by the way ', text)
            text = re.sub(" btwn  ", ' between ', text)
            text = re.sub(r'([a-z])\1{2,}', r'\1', text) #if the same character is repeated more than twice, remove it to one. (E.A. ahhhhhh --> ah)
            text = re.sub(' ctrl ', ' control ', text)
            text = re.sub(' cuz ', ' because ', text)
            text = re.sub(' dif ', ' different ', text)
            text = re.sub(' dm ', ' direct message ', text)
            text = re.sub("n't", r' not ', text)
            text = re.sub(" fav ", ' favorite ', text)
            text = re.sub(" fave ", ' favorite ', text)
            text = re.sub(" fml ", " fuck my life ", text)
            text = re.sub(" hq ", " headquarter ", text)
            text = re.sub(" hr ", " hours ", text)
            text = re.sub(" idk ",  "i do not know ", text)
            text = re.sub(" ik ", ' i know ', text)
            text = re.sub(" lol ", ' laugh out loud ', text)
            text = re.sub(" u ", ' you ', text)
            text = re.sub("√¶", 'ae', text) #encoding error for áe. replace it with ae
            text = re.sub("√® ", 'e', text) #encoding error for é. replace it with e
            text = text.strip()
            return text
        
        def tokenization(text):
            text = re.split('\W+', text) #split words by whitespace to tokenize words
            return text

        def remove_stopwords(text):
            text = [word for word in text if word not in stopword] #remove stopwords in the nltk stopwords dictionary
            return text

        def lemmatizer(text):
            text = lemmatize_sentence(" ".join(text)) #lemmatize the tokenized words. Lemmatized > Stemming in this case
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
        
        @st.cache_data(experimental_allow_widgets=True)
        def extract_array(df,ind):
            my_bar = st.progress(0, text="Initializing Text Cleaning")
            with st.form("Original Text"):
                st.write(df['text'][ind])

                submit_1 = st.form_submit_button("Continue to Initial Cleaning Process")   
            
                if submit_1: 
                    st.session_state['submit_1'] = True
            time.sleep(2)

            if st.session_state['submit_1']:
                clean_text = df['text'].apply(lambda x:clean(x.lower()))         #first clean the text on lower cased list of dreams
                clean_text.dropna()
                with st.form("Initial Data Cleaning"):
                    st.write(clean_text[ind])

                    submit_2 = st.form_submit_button("Continue to Tokenization")           
                    if submit_2:
                        st.session_state['submit_2'] = True

            if st.session_state['submit_2']:
                my_bar.progress(10, text = "Initial Dreams Cleaning Complete")
                time.sleep(2)

                tokenized = clean_text.apply(lambda x: tokenization(x))          #tokenize the cleaned text
                clean_text = tokenized.apply(lambda x: " ".join(x))              #rejoin the words (just in case white space still present)
                clean_text.dropna()
                tokenized.dropna()
                
                with st.form("Tokenization"):
                    st.write(" , ".join(tokenized[ind]))

                    submit_3 = st.form_submit_button("Continue to Stopwords Removal")         
                    if submit_3:
                        st.session_state['submit_3'] = True

            if st.session_state['submit_3']:         
                my_bar.progress(30, text = "Dreams Tokenization Complete")
                time.sleep(2)

                x_stopwords = tokenized.apply(lambda x: remove_stopwords(x))     #remove stopwords from tokenized list
                x_stopwords.dropna()
                
                with st.form("Stopwords Removal"):
                    st.write(" ".join(x_stopwords[ind]))

                    submit_4 = st.form_submit_button("Continue to Lemmatization")  
                    if submit_4:
                        st.session_state['submit_4'] = True

            if st.session_state['submit_4']:               
                my_bar.progress(50, text = "Dreams Stopwords Removal Complete")
                time.sleep(2)

                lemmatized = x_stopwords.__deepcopy__() 
                lemmatized = [lemmatizer(x) for x in lemmatized]
                
                with st.form("Lemmatization"):
                    st.write(" ".join(lemmatized[ind]))

                    submit_5 = st.form_submit_button("Create Corpus")  
                    if submit_5:
                        st.session_state['submit_5'] = True

            if st.session_state['submit_5']:                  
                my_bar.progress(70, text = "Dreams Lemmatization Complete")
                time.sleep(2)

                complete = lemmatized.apply(lambda x: " ".join(x))               #rejoin the words so it will look like a sentence
                mapx = vectorization(complete)                                   #start of mapping to corpus
                name = get_column_name(complete)
                mapx = pd.DataFrame(mapx, columns = name)
                mapx.columns = name
                my_bar.progress(90, text = "Dreams Corpus Complete")
                time.sleep(2)
                my_bar.progress(100, text = "Dreams Text Cleaning Complete")

                return clean_text, tokenized, x_stopwords, lemmatized, complete, mapx

        clean_text, tokenized, x_stopwords, lemmatized, complete, corpus = extract_array(semi, st.session_state['row_n'])

        titles = ['clean_text', 'tokenized', 'x_stopwords', 'lemmatized', 'complete', 'corpus']

        st.session_state['clean_text'] = clean_text
        st.session_state['tokenized'] = tokenized
        st.session_state['x_stopwords'] = x_stopwords
        st.session_state['lemmatized'] = lemmatized
        st.session_state['complete'] = complete
        st.session_state['corpus'] = corpus
        st.session_state['semi'] = semi

        st.write("Preview of the Different Cleaned Datasets")
        radio = st.radio("Choose the Table you would like to see",
                    ('clean_text', 'tokenized', 'x_stopwords', 'lemmatized', 'complete', 'corpus', 'semi'),
                    horizontal=True)
        
        if radio == "clean_text":
            st.write("From the intial text cleaning, the readers will see that all punctuations are eliminated and all words were transformed to lower case. This is becasue we are trying to segment the sentence by words. For instance, 'hello', 'hello.', and 'Hello' will be recognized as two different words in the eyes of python. Thus, to prevent this from happening, the data has been transformed with the measures mentioned above.")
            st.dataframe(clean_text.head(20))
        
        elif radio == "tokenized":
            st.write("From the tokenization, one will observe that the sentences are now 'tokenized' by each word.")
            st.dataframe(pd.DataFrame(st.session_state['tokenized']).head(20))

        elif radio == "x_stopwords":
            st.write("In comparison to the tokenized version of the data, the readers will see that all the prepositions, conjunctions and various connecting words have been eliminated.")
            st.dataframe(x_stopwords.head(20))

        elif radio == "lemmatized":
            st.write("From the lemmatized dataset in comparison to the x_stopwords dataset, the reader will observe that the words have been reverted back to its original root states.")
            st.dataframe(lemmatized.head(20))

        elif radio == "complete":
            st.write("The below is the completed dataset after the cleaning process. In contrary to the lemmatized version, now each row is back to a sentence format rather than tokenized. ")
            st.dataframe(complete.head(20))

        elif radio == "corpus":
            st.write("The corpus below shows how many times a word appears in each sentence (row). Because there are about 1000 dreams, it is inherent that not all words would be in a sentence, thus showing a lot of zero values.")
            st.dataframe(corpus.head(20))

        elif radio == "semi":
            st.write("The Semi Dataset is for the purpose of the analysis. Because shorter length dreams are often harder to extract information due to the lack of it, we eliminated the dreams that are in the low 5 percentile.")
            st.dataframe(semi.head(20))        
    #     except:
    #         st.warning("Please Complete the Previous Stage Before Moving On")

    # except:
    #     st.warning("Please Complete the Previous Stage Before Moving On")
                
########################################################################################
###############       POS Tagging / NER Visualization  page      #######################
########################################################################################

def part_of_speech_tag():
    st.title("Part of Speech Tagging Visualization")

    nlp = spacy.load("en_core_web_sm")

    st.write(f"In order to understand the sentence compositions in Dreams, we first take a look at the different part of speech that were utilized in the text and their frequencies. Using the `Spacy` pacakge, we go through each lines of Dream and tag each word for its POS. To briefly mention the methodology behind this tagging process, Spacy has a dictionary that has preset tags for each word. Then, depending on the position of the word within each sentence cross refrenced with the Spacy dictionary, the POS is determined. The below is the same result of what POS tagging looks like.")

    nlp = spacy.load("en_core_web_sm")

    try:
        result = st.button("Click to Start POS Tagging")
        if result:
            complete_load = st.session_state['complete']
            st.session_state['show'] = True


            @st.cache_data
            def pos_preprocess(df):
                my_bar = st.progress(0, text="Part of Speech Tagging Initialized")
                tag_dict = {"word" :[], "tag":[]}

                for e,i in enumerate(df):
                    sent = nlp(i)
                    for j in sent:
                        tag_dict['word'].append(j.text)
                        tag_dict['tag'].append(j.tag_)
                        my_bar.progress((1/df.shape[0])*(e+1), text = f"{e+1} acquired POS Tags")


                tag_df  = pd.DataFrame(tag_dict)
                my_bar.progress(100, text = "POS Tagging Complete")

                return tag_df
            
            tag_df = pos_preprocess(complete_load)

            rows = st.columns(2)
            rows[0].markdown("Sample POS Tag")
            rows[0].dataframe(tag_df.head(30))
            rows[1].markdown("POS Tag List")
            rows[1].dataframe(pd.read_csv("https://gist.githubusercontent.com/veochae/447a8d4c7fa38a9494966e59564d4222/raw/9df88f091d6d1728eb347ee68ee2cdb297c0e5ff/spacy_tag.csv"))


            def barplot(x, z="", l = False):
                t = np.unique(x, return_counts = True)
                s = np.argsort(t[1])

                if l == True:
                    x = t[0][s][-z:]
                    y = t[1][s][-z:]
                else:   
                    x = t[0][s]
                    y = t[1][s]

                fig6 = px.bar(x = x, 
                            y = y, 
                            labels = dict(x = "Part of Speech", y = 'Count'),
                            title = "Count of Part of Speech in the Entire Corpus")    
                    
                st.plotly_chart(fig6,theme="streamlit", use_container_width=True)    

            with st.container():
                st.write("Next with the full list of POS Tags throughout all the Dreams that we have collected, we plot a barplot to see which prepositions were heavily uitilized in the Dreams. As one can see from the barplot, Nouns were mostly utilized since Dreams have objects that have to be described in detail. Then, Adverbs and different tenses of verbs were heavily utilized in describing the Dreamers' actions during the dream.")
                barplot(tag_df['tag'])

    except:
        st.warning("Please Complete the Before Step Afore Starting The Current Stage")    
    try:
        if st.session_state['show']:
            
                df = st.session_state['semi']

                st.session_state['keyword'] = st.text_input("Type in Keyword you would like to see in the Dream")
                filtered = df[df['text'].str.contains(st.session_state['keyword'])]
                
                if "keyword" in st.session_state.keys():
                    st.dataframe(filtered)
                    st.session_state['filtered'] = filtered
                    
                else:
                    st.dataframed(df)

                st.session_state['row_n'] = int(st.text_input("Type in Index Number of the Dream you would like to examine"))

                with st.container():
                    temp = df['text'][st.session_state['row_n']]
                    model = "en_core_web_sm"

                    st.title("POS Taggging and NER Visualization")
                    text = st.text_area("Text to analyze", temp, height=200)
                    doc = spacy_streamlit.process_text(model, text)

                    spacy_streamlit.visualize_parser(doc)

                    spacy_streamlit.visualize_ner(doc,
                                                show_table=True
                                                    )


                    # spacy_streamlit.visualize(["en_core_web_sm"], df['text'][row_n])

    except:
            st.warning("Please Complete the Before Step Afore Starting The Current Stage")    

########################################################################################
#############################       lda  page      #################################
########################################################################################

# def lda():
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



########################################################################################
#############################       TF-IDF  page      ##################################
########################################################################################
def tf_idf():
    st.title("TF-IDF Analysis")
    try:
        st.write(f"Your current selected dream index is {st.session_state['row_n']}")

        result_ti = st.button("Click Here to start TF-IDF")

        if result_ti:
            st.session_state['result_ti'] = True
        try:
            if st.session_state['result_ti']:
                corpus = st.session_state['corpus']
                token = st.session_state['tokenized']
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
                @st.cache_data
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
                        tf_idf_temp = tf_idf(t, idf_dict)
                        tf_idf_li.append(tf_idf_temp)

                    time.sleep(2)
                    my_bar.progress(100, "TF-IDF Calculation Complete. Exporting...")
                    return pd.DataFrame(tf_idf_li) , pd.DataFrame(tf_li) , pd.DataFrame(idf_dict, index=[0])
                
                tf_idf_df, tf_df, idf_df= main(corpus, tokenized)

                st.write("Preview")
                radio = st.radio("Choose the Table you would like to see",
                            ('TF-IDF', "TF", "IDF"),
                            horizontal=True)
                
                if radio == "TF-IDF":
                    st.dataframe(tf_idf_df.head())
                
                elif radio == "TF":
                    st.dataframe(tf_df.head())

                elif radio == "IDF":
                    st.dataframe(idf_df.head())  



                def barplot(tf_idf_df, number_of_words):
                    rendered_dream = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n'],:].sort_values(axis = 0, ascending = False)[:number_of_words]})
                    words = rendered_dream.index.tolist()
                    rendered_dream['words'] = words

                    fig = px.bar(rendered_dream,
                                    x='words', 
                                    y='values', 
                                    title=f"Dream {st.session_state['row_n']} tf-idf score words",
                                    labels = dict(words = "Words", values = 'TF-IDF Score'))
                    st.plotly_chart(fig,theme="streamlit", use_container_width=True)   

                barplot(tf_idf_df = tf_idf_df, number_of_words = 10)
                change = 2

                if change == 2:
                    def barplot_2(tf_idf_df, number_of_words):
                        rendered_dream = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n'],:].sort_values(axis = 0, ascending = False)[:number_of_words]})
                        words = rendered_dream.index.tolist()
                        rendered_dream['words'] = words

                        rendered_dream_2 = pd.DataFrame({"values": tf_idf_df.iloc[st.session_state['row_n_2'],:].sort_values(axis = 0, ascending = False)[:number_of_words]})
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

                    st.write(f"Current Keyword is `{st.session_state['keyword']}`")
                    st.dataframe(pd.DataFrame(st.session_state['filtered']))
                    st.write("Choose your Second Dream by row index")
                    try:
                        st.session_state['row_n_2'] = int(st.text_input("Type in Index Number of the Dream you would like to examine"))
                        barplot_2(tf_idf_df = tf_idf_df, number_of_words = 10)
                    except:
                        st.warning("Please Input the Second Dream Row Number")
        except:
            print("")
    except:
        st.warning("Please Complete the Previous Step Before Moving On")
########################################################################################
#############################       Dream Summarization + Continuation      #################################
######################################################################################## 
        
def summary_continue():
    st.title("Dream Summarization and Continuation Using GPT3.5")
    with st.form("open_ai_cred"):
        openai.api_key = st.text_input("OpenAI API Key")

        submitted = st.form_submit_button("Submit")    
    try:
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

        with st.form("asdf"):
            st.header("Original Text")
            try:
                dream = st.session_state['semi']['text'][st.session_state['row_n']]
                st.write(dream)
            except:
                pass
            dream_submit = st.form_submit_button("Proceed to Summarization and Continuation") 
            if dream_submit:
                st.session_state['dream_submit'] = True

        if st.session_state['dream_submit']: 
            if len(dream) <= 280:
                length = len(dream) * 0.6 
            else:
                length = 280     
            try:     
                summary = summarize_dream("Summarize this dream to less than 280 words from the storyteller's perspective \n" + "Dream: " + dream, length = length)
            except:
                st.warning("This Error is either: 1. Do not have enough API balance 2. Not the correct API Key")
            continuation = summarize_dream("Tell me what happens after this story as if you are the storyteller \n" + dream + "\n [insert]", length = 280)

            st.header("Dream Summary")
            st.write(summary)

            classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
            prediction = classifier(summary)
            emotion = [x['label'] for x in prediction[0]]
            score = [y['score'] for y in prediction[0]]

            fig10 = make_subplots(rows=1, cols=1)

            fig10.add_trace(go.Bar(x = emotion,
                                    y = score,
                                    name = f"Dream {1}"))

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


            st.header("Dream Continuation")
            st.write(continuation)

            st.header("Dream Visualization")
            dalle = summarize_dream("As if you are the writer, summarize this dream into one sentence \n"+dream, length = 100)
            st.write(dalle)
            time.sleep(30)
            response = openai.Image.create(
                        prompt="Give me a descriptive image of the statement: " + dalle,
                        n=1,
                        size="1024x1024")
            
            st.image(response['data'][0]['url'])
            dream_submit = False
    except: 
        st.warning("Please Complete the Previous Step Before Moving On")

########################################################################################
#############################       Data Download      #################################
########################################################################################        

def data_download():
    st.title("Download Datasets")

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

########################################################################################
#############################       sidebar  page      #################################
########################################################################################

page_names_to_funcs = {
    "Introduction": introduction,
    "Data Collection": data_collection,
    "Data Cleaning": data_cleaning,
    "POS and NER": part_of_speech_tag,
    "TF-IDF": tf_idf,
    "Dream Summary and Continuation": summary_continue,
    "Data Download": data_download
}

demo_name = st.sidebar.selectbox("Please Select a Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()