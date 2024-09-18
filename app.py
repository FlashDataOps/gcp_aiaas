import streamlit as st
from HelloWorld import main

# Define a function to process the user input
def get_chatbot_response(user_input):
    # For demonstration purposes, we return a simple response
    # You can replace this with actual chatbot logic or a call to a machine learning model
    if user_input:
        return main(query=user_input)
    else:
        return "Please enter a valid input."

# Streamlit App Layout
st.title("MontyBot Cloud Edition")

st.write("Type something to chat with Monty:")

# Text input box for user input
user_input = st.text_input("You:", "")

# Display chatbot response when input is provided
if st.button("Send"):
    if user_input:
        response = get_chatbot_response(user_input)
        st.write(f"Bot: {response}")
    else:
        st.write("You haven't typed anything!")

# You can add more features or chatbot logic in the `get_chatbot_response` function