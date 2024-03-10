import streamlit as st
#plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd



########################################################################################
#############################       TF-IDF  page      ##################################
########################################################################################
tf_latex = r'\text{TF}(w, d) = \frac{\text{Count of } w \text{ in } d}{\text{Total number of words in } d}'
idf_latex = r'\text{IDF}(w) = \ln\left(\frac{N}{n_w}\right)'
tf_idf_latex = r'\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)'
text = r"""\text{Number of Words}: N \\ \text{Number of documents containing } w: n_w"""

st.title("TF-IDF Analysis")
try:
    st.info(f"Chosen Dream: Dream {st.session_state['row_n']}",icon="ℹ️")    
    st.write("Suppose we now want to understand which words are specific to a given dream, say the dream that you have selected at the start. Similarly, if a marketing analyst is analyzing product reviews, they may want to understand which words are particular to a given review. The same way, a hiring manager analyzing resumes may find it useful to know which skills or qualifications are specific to a given applicant based on the resume. For all these tasks and many more a tool referred to as TF-IDF can come in handy.")
    st.write("TF-IDF stands for Term Frequency - Inverse Document Frequency, and it is a numerical representation used in NLP to understand how specific a given word is to a given document. Let's now study the building blocks of this metric.")
    st.write("**Term Frequency (TF)**: Term Frequency in the simplest sense measures how often a word appears in a document. It takes the document, and counts how many times each word is appearing in the specific document. Although there are several variations for TF, here is the formula used in this app:")
    st.latex(tf_latex)
    st.write("**Inverse Document Frequency (IDF)**: Inverse Document Frequency, unlike the TF, takes into consideration all the documents making up the corpus. The purpose of IDF is to measure how common (equivalently, how rare) a word is across the entire corpus. Again, there are a few variations for IDF, and in the current app the following version is used:")
    st.latex(text)
    st.latex(idf_latex)
    st.write("**TF-IDF**: TF-IDF is the amalgamation of TF and IDF as you can tell by the name! In essence TF-IDF for a given term used in a certain document is simply the term frequency of the word scaled or adjusted according to how common the word is across the entire corpus. For a word that appears frequently in the document in question, but also happens to appear commonly throughout the entire corpus, it’s frequency will be scaled down, thus resulting in a relatively lower TF-IDF score. Conversely, a word appearing frequently in a document but being rare across the corpus will have its frequency adjusted/scaled upwards thus resulting in relative higher TF-IDF score. For instance, when we search for the word **entrepreneurship**, a document pertaining to Babson College will have a higher TF-IDF score for the word in comparison to a document about Olin College, because entrepreneurship is more relevant in the document for Babson!")
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
    except Exception as e:
        st.warning("Please Press to Start!")
except Exception as e:
    st.warning("Please Complete the Previous Step Before Moving On")