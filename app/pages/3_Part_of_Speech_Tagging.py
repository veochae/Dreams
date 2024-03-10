import streamlit as st
import pandas as pd
import numpy as np
#plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from spacy import displacy
import spacy


##########en-core-sm preload
@st.cache_resource
def load_nlp():
    return spacy.load('en_core_web_sm')

# ########################################################################################
# ###############       POS Tagging / NER Visualization  page      #######################
# ########################################################################################


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
    
    if result:
        st.session_state['show'] = True
        st.info("Tagging POS Tags for all Dreams. This may take about 1Minute.")
        complete_load = st.session_state['complete']
        tag_df = pos_preprocess(complete_load)

        cola, colb = st.columns(2)
        with cola:
            st.header("POS Tag List")
            st.dataframe(pd.read_csv("https://gist.githubusercontent.com/veochae/447a8d4c7fa38a9494966e59564d4222/raw/9df88f091d6d1728eb347ee68ee2cdb297c0e5ff/spacy_tag.csv"))
        with colb:
            st.header("What is this Table?")
            st.markdown("The table on the left shows various tags with which words from a corpus will get tagged with as a result of the POS tagging process. These are standard tags used in POS tagging and are not corpus-specific.")

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
            st.write("We already know which one of the two statements is a question. Not only because of the question mark on the second statement, but because it is a sentence that starts with an auxiliary ”Do” and a pronoun as the target of asking the question. Obviously, humans do not actively process the part of speech for each and every sentence one encounters, but how about when the machine has to learn sentence structure? Just like the young versions of ourselves first learning how to comprehend the sentence structure, machine has to learn the sentence structures of English as well. Now, we can use the individual POS Tags as a sequence in order to essentially create a formula of sentence structures. With the example above, because")
            st.write("auxiliary + pronoun + verb + … ")
            st.write("is the sequential order of POS tags in the given sentence, the machine will now recognize that this sentence is a question.")
            st.write("As such, POS tagging not only helps machines understand the individual usage of singular words, but also provides an even more powerful tool when used on an aggregated level!")
        
            df = st.session_state['semi']

            with st.container():
                temp = np.str.split(df['text'][st.session_state['row_n']], ".")[0] + "."
                model = "en_core_web_sm"

                st.title("POS Taggging Visualization")
                st.write("To illustrate how POS tagging works in action, for the interest of space we will only tag the first sentence of the dream that you have chosen at the start. As you can see, that first sentence is shown in the text window below, followed by a visualization illustrating the POS tags. Note that this interface is fully interactive, so after you review the results from POS tagging for the first sentence in the below display, feel free to type in any text (perhaps the first sentence from your own past dream?;) and see how it gets tagged and visualized.")
                text = st.text_area("Text to analyze", temp, height=200)
                # doc = spacy_streamlit.process_text(model, text)

                # spacy_streamlit.visualize_parser(doc)

                nlp = load_nlp()

                doc = nlp(text)

                for token in doc:
                    if token.dep_ != "anything":
                        token.dep_ = ""

            c = st.container()
            svg = displacy.render(doc, style='dep', jupyter=False, options={'distance': 90})
            c.image(svg, use_column_width='auto')

            st.info("Next click on the next tab on the left to move on to the Named Entity Recognition Section!", icon="ℹ️")
except:
    st.warning("Please Complete the Before Step Afore Starting The Current Stage")    
        