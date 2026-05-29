import streamlit as st
from agent import real_estate_agent
import uuid

st.set_page_config('UAE Real Estate AI', page_icon='🏘️')

st.title('UAE Real Estate AI Advisor 🇦🇪')
st.markdown("Your intelligent guide to Dubai and Abu Dhabi property valuations.")

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if prompt := st.chat_input('Ask me about a property'):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
    with st.spinner("Analyzing UAE market data...."):
        user_input={'user_query':prompt}
        final_state=real_estate_agent.invoke(user_input,{'configurable':{"thread_id":st.session_state.thread_id}})
        ai_response=final_state['final_response']
    with st.chat_message('assistant'):
        st.markdown(ai_response)
        st.session_state.messages.append({'role':'assistant','content':ai_response})