import streamlit as st
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import os
import google.generativeai as genai



# Globals
client = MilvusClient("milvus_demo1.db")
coll_name = "legalverse"
emb_dim = 384

model = SentenceTransformer('all-MiniLM-L6-v2')

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCiLVasyxqKMP3qMcJLXdqu33XyTNDqK1M'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

gemini_model = genai.GenerativeModel('gemini-1.5-flash')


def generate_answer_gemini(query, model, client, coll_name, genai, topk = 2):
    # query = "What is the First Schedule of the Patents Rules 2003?"
    query_vectors = model.encode([query])

    res = client.search(
        collection_name=coll_name,  # target collection
        data=query_vectors,  # query vectors
        limit=topk,  # number of returned entities
        output_fields=["id", "text"],  # specifies fields to be returned
    )

    # pretext
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer. If the context is not relevant, return "No relevant passages found."
    Make sure your answers are as explanatory as possible.
    \nNow use the following context items to answer the user query:
    \nRelevant passages: {context}
    User query: {query}\n
    Answer in 200 words
    """


    concatenated_text = "";
    for x in res[0]:
        concatenated_text += x['entity']['text'] + "\n"

    base_prompt = base_prompt.format(query=query, context=concatenated_text)

    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    response = gemini_model.generate_content(base_prompt)
    print(response.text)



# --------------------------------- UI ---------------------------------
# Set the page layout to centered
st.set_page_config(layout="centered")

# Title and styling
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat Application</h1>", unsafe_allow_html=True)
st.markdown("<style> .stButton>button { background-color: #4CAF50; color: white; } </style>", unsafe_allow_html=True)

# Persistent chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.markdown("<div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    st.markdown(f"<div style='padding: 8px; margin: 5px 0; border-radius: 10px; background-color: {'#DCF8C6' if message['is_user'] else '#ECECEC'};'>{message['message']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input area at the bottom
user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here...")


# st.session_state.clear()
if st.button("Send") and user_input:
    st.session_state.chat_history.append({"message": user_input, "is_user": True})
    st.session_state.chat_history.append({"message": f"Echo: {generate_answer_gemini(user_input, model, client, coll_name, genai)}", "is_user": False})
    user_input = ""  # Clear the input field
    st.rerun()  # Rerun the app to display the new messages


