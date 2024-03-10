import streamlit as st
import requests
#plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
#huggingface
from transformers import pipeline, set_seed




def query(payload, API_URL, headers):
    response = requests.post(API_URL, headers, json=payload)
    return response.json()

def summarize_dream(api_key, prompt):
    API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
    headers = {"Authorization": f"Bearer {api_key}"}

    output = query({
        "inputs": prompt,
    }, API_URL, headers)

    try:
        return output[0]['summary_text']
    except Exception as e:
        st.write(e)


########################################################################################
#############################       Sentiment Analysis      #################################
######################################################################################## 
st.title("Sentiment Analysis")
try:
    st.write("While in the earlier days of ecommerce online sellers would be satisfied with any reviews of the products and services they were offering, in recent years companies have shifted into asking for more detailed reviews. Why? Because using Sentiment Analysis companies can learn how the customers feel about their products, and the more detailed the review the higher the chances of accurately uncovering customer feelings!")
    st.write("In the earlier days of the Sentiment Analysis, it was quite simple. One would classify a piece of text as positive, negative, or neutral. We won't dive into too much detail about how that has been done, but if you are curious, checkout this [link](https://huggingface.co/blog/sentiment-analysis-python)! Coming back to the modern days, Sentiment Analysis has evolved into something more specific and granular: Emotion Analysis")
    st.write("Instead of uncovering whether the customers' reactions were positive or negative, one starts to look at multiple (and more granular) feelings such as: fear, joy, happiness, surprise, love, anger, sadness, etc.")
    st.write("The pretrained model that we use for the purpose of this exercise is from Hugging Face – a popular platform for [NLP](https://huggingface.co/). In particular, using over 100,000 labeled tweets as its training data, neural networks were trained to recognize different speech patterns that lend themselves to various feelings and emotions. Having learned about human emotion based on training data, the model can now be applied to any piece of text – including text on reported dreams – to predict emotion!")
    st.write("So, without further ado, let's uncover the emotions present in the dream that you selected at the outset!")

    with st.form("sentiment_analysis"):
        st.info(f"Chosen Dream: Dream {st.session_state['row_n']}",icon="ℹ️")
        with st.expander(f"Click Here to View the Selected Dream "):
            dream = st.session_state['semi']['text'][st.session_state['row_n']]
            st.write(dream)        
        submitted_sentiment = st.form_submit_button("Let's Begin!")   
    
    if submitted_sentiment:
        try:     
            summary = summarize_dream(st.session_state['hugging_face_key'],dream)
            st.session_state['summary'] = summary
        except Exception as e:
            st.warning("This Error is either: 1. The model has not been loaded yet 2. Not the correct API Key")

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
    st.warning("Please Complete the Previous Step Before Moving On")