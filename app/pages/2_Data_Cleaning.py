import streamlit as st
import spacy
from wordcloud import WordCloud #,STOPWORDS
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

##########en-core-sm preload
@st.cache_resource
def load_nlp():
    return spacy.load('en_core_web_sm')

##########wordcloud
def wordcloud(x, lim):
    text = " ".join(x)
    cloud = WordCloud(collocations = False, max_words = lim, min_word_length = 3).generate(text)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)

# ########################################################################################
# #############################       data cleaning  page      ###########################
# ########################################################################################

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
            st.write("Just from the first look of the wordcloud below, you might be thinking to yourself: 'That was not what I expected!' You might have thought that the words showing up would be dream, sleep, night, etc. -- words that would commonly appear when someone writes about their dream. But to see that on a wordcloud wouldn’t be very illuminating on what dreamers have been dreaming recently, would it? Dreams are interesting for the unique and extraordinary stories they tell. In order to see what unique elements appeared commonly throughout the recently reported dreams, the words were selected based on high TF-IDF scores. For now it's okay if you are not yet familiar with TF-IDF! We will discuss it further in the sections following. But in the meantime, take a look at the wordcloud to see what our dreamers have been dreaming about recently. Dream on!")

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
                wordcloud_words = []
                
                documents = [corpus.iloc[i,:].to_dict() for i in range(corpus.shape[0])]
                time.sleep(2)

                my_bar.progress(35, "Calculating tf")
                t = 0
                for l, r in enumerate(documents):
                    try:
                        tf_temp = tf(r, tokenized[l])
                        tf_li.append(tf_temp)
                        t+=1
                    except:
                        print(t)
                
                time.sleep(2)
                my_bar.progress(70, "Calculating idf")
                idf_dict = idf(documents)

                time.sleep(2)
                my_bar.progress(95, "Calculating tf_idf")
                for t in tf_li:
                    tf_idf_li.append(tf_idf(t, idf_dict))
                    wordcloud_words += sorted(tf_idf_li[-1], key=tf_idf_li[-1].get, reverse=True)[:3]
                
                my_bar.progress(100, "TF-IDF Calculation Complete. Exporting...")

                return pd.DataFrame(tf_idf_li), wordcloud_words
            
            st.session_state['tf_idf_df'],st.session_state['wordcloud_words'] = main(corpus, tokenized)

            wordcloud(st.session_state['wordcloud_words'], lim=100)
        
        st.info("Next click on the next tab on the left to move on to the Part of Speech Tagging Section!" ,icon="ℹ️")


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
        
    stopword = nltk.corpus.stopwords.words('english')

    if 'submit_5' in st.session_state.keys():
        reset = st.button("Click here to reset and choose another keyword")

        if reset:
            del st.session_state['submit_5']
            st.info("Why not? Press the Reset Button One More Time :)")

        else:
            extract_array_sample(st.session_state['row_n'])
                
    else:
        result_dc = st.button("Click to Start Data Preprocessing")

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
                    my_bar.progress(50, text = "Calculating Length of each Text")
                    time.sleep(2)
                    #calculating length of each dream
                    df['length'] = [len(j) for j in df['text']]
                    my_bar.progress(70, text = "Getting Semi Dataset")
                    time.sleep(2)
                    # if less than or equal to 5th percentile, assign t_f column False
                    df['t_f'] = [True if j > np.percentile(df['length'], 10) else False for j in df['length']]
                    my_bar.progress(90, text = "Making Deep Copy of Semi")
                    time.sleep(2)
                    #only keep t_f == True rows
                    semi = df.loc[df['t_f'] == True, :].reset_index(drop = True).__deepcopy__()
                    my_bar.progress(100, text = "Complete!")

                    return df, semi
                
                df, semi = preprocess(st.session_state['reddit'])

                st.header("Breakdown of Data Cleaning Process")
                st.info("Type in Keyword you would like to see in the Dream" ,icon="ℹ️")
                st.session_state['keyword'] = re.escape(st.text_input("Keyword:"))
                filtered = semi[semi['text'].str.contains(fr"\b{st.session_state['keyword']}\b", regex=True)]
                
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
                    text = re.sub("&amp", "", text)
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

                extract_array_sample(st.session_state['row_n'])  
        except:
            st.warning("Please Complete the Previous Stage Before Moving On")

except:
    st.warning("Please Complete the Previous Stage Before Moving On")