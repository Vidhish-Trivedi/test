import streamlit as st
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import os
import google.generativeai as genai

client = MilvusClient("milvus_demo1.db")
coll_name = "legalverse"
emb_dim = 384

model = SentenceTransformer('all-MiniLM-L6-v2')

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCiLVasyxqKMP3qMcJLXdqu33XyTNDqK1M'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def generate_answer(query, model, client, collection, genai, topk=2, chat_history=None):
    """
    Generate a response using the Gemini model based on relevant search results from Milvus.
    Incorporates previous chat history for more contextual answers.
    """
    query_vector = model.encode([query])

    search_results = client.search(
        collection_name=collection, 
        data=query_vector,  
        limit=topk, 
        output_fields=["id", "text"], 
    )

    # Base prompt for the Gemini model
    base_prompt = """
    Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer. If the context is not relevant, then answer according to yourself in Indian context."
    Make sure your answers are as explanatory as possible.
    \nNow use the following context items to answer the user query:
    \nRelevant passages: {context}
    User query: {query}\n
    Answer in 300 words
    """

    # Concatenate retrieved text into one context block
    concatenated_context = "\n".join([x['entity']['text'] for x in search_results[0]])

    # Convert chat history to a readable string format
    history_text = ""
    if chat_history:
        for message in chat_history:
            speaker = "User" if message['is_user'] else "Assistant"
            history_text += f"{speaker}: {message['message']}\n"

    # Format the prompt with the context, chat history, and query
    formatted_prompt = base_prompt.format(
        context=concatenated_context,
        chat_history=history_text,
        query=query
    )

    # Generate a response from the Gemini model
    gemini_response = gemini_model.generate_content(formatted_prompt)
    
    return gemini_response.text

# --------------------------------- UI ---------------------------------
st.set_page_config(layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat Application</h1>", unsafe_allow_html=True)
st.markdown("<style> .stButton>button { background-color: #4CAF50; color: white; } </style>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("<div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    st.markdown(f"<div style='padding: 8px; margin: 5px 0; border-radius: 10px; background-color: {'#DCF8C6' if message['is_user'] else '#ECECEC'};'>{message['message']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here...")

if st.button("Send") and user_input:
    # Add the user's message to chat history
    st.session_state.chat_history.append({"message": user_input, "is_user": True})

    # Generate the Gemini model's response with chat history
    gemini_response = generate_answer(
        query=user_input, 
        model=model, 
        client=client, 
        collection=coll_name, 
        genai=genai, 
        chat_history=st.session_state.chat_history
    )

    st.session_state.chat_history.append({"message": gemini_response, "is_user": False})

    user_input = ""
    st.rerun()
