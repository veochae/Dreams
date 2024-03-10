import streamlit as st

########################################################################################
#############################       Setup for HuggingFace      #################################
######################################################################################## 
def set_up_openai():
    st.title("Setting Up your Hugging Face API")
    st.video("https://youtu.be/VMjJ4BrYVaE")
    with st.form("hugging_face_cred"):
        key_open = st.text_input("Hugging Face API Key")
        submitted = st.form_submit_button("Submit")   
        if submitted:
            st.session_state['hugging_face_key'] = key_open
            st.success("Your API Key has been Processed!")