import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL context for Mac/Legacy systems
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK setup
nltk.data.path.append(os.path.abspath("nltk_data"))
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CACHING THE MODEL TRAINING ---
# We use @st.cache_resource so the model doesn't retrain on every user interaction
@st.cache_resource
def load_and_train_model():
    file_path = os.path.abspath("./patterns.json")
    
    # Check if file exists to prevent crash
    if not os.path.exists(file_path):
        st.error(f"Error: 'patterns.json' not found at {file_path}. Please ensure the file exists.")
        st.stop()

    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Handle both list format and dictionary format {"intents": [...]}
    if isinstance(data, dict) and 'intents' in data:
        intents = data['intents']
    else:
        intents = data

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    clf = LogisticRegression(random_state=0, max_iter=10000)

    tags = []
    patterns = []
    
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)

    if not patterns:
        st.error("Error: No patterns found in JSON to train the model.")
        st.stop()

    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
    
    return vectorizer, clf, intents

# Load model and data
vectorizer, clf, intents = load_and_train_model()

def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm not sure I understand."

def main():
    st.title("WanderStay, a premium hotel finder of INDIA")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # --- HOME MENU ---
    if choice == "Home":
        st.write("Welcome to WanderStay, your one stop solution for finding famous hotels in INDIA. Please kindly search for hotels in state wise or UT wise format like:- 'Recommend some best-rated hotels to rent in Goa', and then press Enter to start the conversation.")

        # Check if chat_log exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Using a static key to prevent input focus loss
        user_input = st.text_input("You:", key="user_input_main")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key="chatbot_response_area")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great JOURNEY ahead!")
                st.stop()

    # --- CONVERSATION HISTORY MENU ---
    elif choice == "Conversation History":
        st.header("Conversation History")
        
        # Check if file exists before reading
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)  # Skip header safely
                
                for row in csv_reader:
                    if len(row) >= 3: # Ensure row has data
                        st.text(f"User: {row[0]}")
                        st.text(f"Chatbot: {row[1]}")
                        st.text(f"Timestamp: {row[2]}")
                        st.markdown("---")
        else:
            st.write("No conversation history found.")

    # --- ABOUT MENU ---
    elif choice == "About":
        st.write("The main goal of making WanderStay is to assist people in India by serving as a personalized, intelligent travel assistant for finding hotels across states and union territories")

        st.subheader("Overview of WanderStay:")
        st.write("""
        Wanderstay is divided into various parts:
        \n1. It allows users to search for hotels in a specific state, city, or union territory by simply entering their destination.
        \n2. It categorizes hotels based on region, making it easier for users to navigate their options.
        \n3. For each state or union territory, WanderStay can provide curated lists of iconic hotels with cultural or historical value (e.g., palaces in Rajasthan), eco-friendly stays or rural homestays, modern business hotels in metro cities like Mumbai, Delhi, or Bengaluru.
        \n4. It also promotes sustainable and eco-friendly accommodations for environmentally conscious travelers.
        """)

        st.subheader("Dataset of WanderStay")
        st.write("""
        The dataset used in WanderStay is a collection of labelled patterns and entities. The data is stored in a list.
        - Patterns: The intent of the user input (e.g. "greeting", "Hotels", "JOURNEY")
        - Entities: The entities extracted from user input (e.g. "Hi", "Show some hotels to stay in for vacations in Goa", "Famous hotels to stay in Gujarat")
        - Text: The user input text.
        """)

        st.subheader("Streamlit WanderStay's Interface:")
        st.write("WanderStay's interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("By offering tailored services and an easy-to-use interface, WanderStay ensures travelers across India find the best accommodation options, saving time, money, and effort. It serves as a one-stop solution for planning stays during vacations, work trips, or even pilgrimages.\n\nDEAR customers, please be ensured that WanderStay will be operational soon across the GLOBE")

if __name__ == '__main__':
    main()
