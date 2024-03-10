import streamlit as st
import requests
#huggingface
from transformers import pipeline, set_seed
import io
from PIL import Image


############### hugging face incorporated function
def query(payload, API_URL, headers):
    response = requests.post(API_URL, headers, json=payload)
    return response.json()

def query_image(payload, API_URL, headers):
    response = requests.post(API_URL, headers, json=payload)
    return response.content

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


def exapnd_dream(prompt):
    generator = pipeline('text-generation', model='openai-gpt')
    set_seed(42)
    length = len(prompt)//4
    end = generator(prompt, max_length=length*2, num_return_sequences=1)
    return end[0]['generated_text']


def text_to_image(api_key, artist, prompt, emotion):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_key}"}

    image_bytes = query_image({
        "inputs": f"In style of {artist} depicting emotion of {emotion} paint:[{prompt}]",
    }, API_URL, headers)
    # You can access the image with PIL.Image for example

    image = Image.open(io.BytesIO(image_bytes)) 

    return image   

########################################################################################
#############################       Dream Summarization + Continuation      #################################
######################################################################################## 
        
st.title("Dreams with GPT") 

try:
    st.write("Now we are ready to jump right into the heart of advanced, state-of-the-art AI: Generative Pre-trained Transformers (GPT) and DALL·E. You are probably most familiar with Chat GPT as it has quickly diffused into multiple applications and industries.")
    st.write("On a very high level, GPT is a large language model that utilizes neural networks (among other things) and is used for various language-related tasks from machine translation to text summarization and continuation. DALL·E on the other hand is a generative AI technology for creating images. You can think of DALL·E as a creative image generator powered by the recent advances in generative AI and large language models. It can take written descriptions and turn them into unique images. Just describe an idea in words, and DALL·E brings it to life as an artwork.")
    st.write("Using the Chat GPT 3.5 Davinci and DALL-E, below we summarize, expand and visualize the dream that you have been observing throughout this app.")

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

        st.header("Dream Summary")
        st.session_state['summary'] = summarize_dream(st.session_state['hugging_face_key'],dream)
        st.write(st.session_state['summary'])

        st.header("Dream Continuation")
        st.session_state['continuation'] = exapnd_dream(st.session_state['summary'])
        start_point = len(st.session_state['summary'])
        st.session_state['continuation'] = st.session_state['continuation'][start_point+1:]
        st.write(st.session_state['continuation'])

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
            response = text_to_image(st.session_state['hugging_face_key'], st.session_state['artist'], st.session_state['summary'], st.session_state['emotion'])
            
            st.image(response)
            dream_submit = False
        else:
            st.warning("Please select an artist")
except: 
    st.warning("Please Complete the Previous Step Before Moving On")