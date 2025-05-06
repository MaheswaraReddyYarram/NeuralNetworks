import streamlit as st
import streamlit.web.cli as stcli

if __name__ == '__main__':
    st.set_page_config(page_title="Streamlit Demo")
    st.title('Simple streamlit demo app')
    st.write('Welcome to your first streamlit app')

    # upload required file
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            st.write(f"File{uploaded_file} uploaded successfully")

    #question
    query_text = st.text_input("Enter your question", value="Enter your question here")
    st.write("The question is: ", query_text)

    # combine above
    with st.form(key = 'qa_form', clear_on_submit=True, border=False):
        openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_files and query_text))
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("please upload a file and enter your question")



