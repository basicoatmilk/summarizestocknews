import streamlit as st
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
import time

# Initialize tokenizer and model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = TFPegasusForConditionalGeneration.from_pretrained(model_name)
sentiment = pipeline("sentiment-analysis")

# Function to scrape and process articles
def scrape_and_process(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all('p')
    text = [res.text for res in results]
    article = ' '.join(text)
    return article

# Function to summarize articles
@st.cache_data()
def summarize(article):
    input_ids = tokenizer.encode(article, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=300, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

# Streamlit app
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    .input-section {
        margin-bottom: 30px;
    }

    .button {
        background-color: #007bff;
        color: #ffffff;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .button:hover {
        background-color: #0056b3;
    }

    .summary-container {
        margin-bottom: 30px;
    }

    .summary {
        padding: 15px;
        margin-bottom: 20px;
        white-space: pre-wrap;
        background-color:#008000;
        border-radius: 5px;
    }

    .sentiment-container {
        margin-bottom: 30px;
    }

    .sentiment {
        padding: 15px;
        margin-bottom: 20px;
        white-space: pre-wrap;
        background-color: #008000;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Stock and Cryptocurrency News</h1>", unsafe_allow_html=True)

# Input for URL
url_input = st.text_input("Enter the URL of the news article")

# Get news summary and sentiment analysis after a delay
if st.button("Get Summary and Sentiment"):
    st.info("Please wait while we fetch and analyze the news...")
    time.sleep(3)

    article = scrape_and_process(url_input)
    summary = summarize(article)
    sentiment_result = sentiment(summary)

    # Displaying the results
    st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary'><b>Summary:</b><br/>{summary}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sentiment-container'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sentiment'><b>Sentiment Analysis:</b><br/>{sentiment_result}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
