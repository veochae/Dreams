import streamlit as st
import spacy_streamlit


########################################################################################
#############################       namee entity recognition  page      #################################
########################################################################################

st.title("Named Entity Recognition")

try:
    st.info(f"Chosen Dream: Dream {st.session_state['row_n']}" ,icon="ℹ️")
    if st.session_state['show']:
            st.write("As the next step of translating human language to machine comprehensible context, we go through the named entity recognition. Well first, we have to know what Named Entity is! ")
            st.write("One can think of a named entity as a label that would be assigned to various sections of a text. But compared to POS tagging, in Named Entity Recognition (NER) what gets identified are instances of names, locations, companies, products, monetary values, percentages, time, etc. So, Named Entity Recognition is yet another step, in addition to POS tagging, to provide further context about text to the machine.")
            st.write("For instance, let’s take the example of two sentences below:")
            st.write("“I like Google” and “I like Wellesley”")
            st.write("As you should know by now, POS tags are going to mark both Google and Wellesley as nouns. As useful as that may seem, POS tagging has not provided any further information regarding what each of those two nouns represents. Named Entity Recognition will flag the two words into Company and Location respectively. That way, the machine will now have a more detailed information behind each word in the sentence. In particular,  one (Google) will get comprehended as a company, and the counterpart (Wellesley) as a location!")
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

