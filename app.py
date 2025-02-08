import streamlit as st
import pandas as pd
import numpy as np
import nltk
import jieba
import jieba.posseg as pseg
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import requests
from io import BytesIO
from snownlp import SnowNLP
from textblob import TextBlob
import re
import kiwipiepy
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from docx import Document  
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mplfonts import use_font
use_font('Noto Serif CJK SC')

stopwords = Stopwords()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
# Load stopwords
cn_stopwords_url = "https://raw.githubusercontent.com/CharyHong/Stopwords/main/stopwords_cn.txt"
en_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"

cn_stopwords = requests.get(cn_stopwords_url).text.splitlines()
en_stopwords = requests.get(en_stopwords_url).text.splitlines()
# Function to generate and display the word cloud
def generate_wordcloud(frequency_data):
    # Join the words and frequencies to form the text for the word cloud
    word_freq = {row['words']: row['frequency'] for index, row in frequency_data.iterrows()}

    # Generate the word cloud using the word frequency data
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()  # Display the word cloud in Streamlit

# Function to read content from .docx file
def read_docx(file):
    doc = Document(file)
    content = []
    for para in doc.paragraphs:
        content.append(para.text)
    return '\n'.join(content)

# Function to clean text
def clean_text(text, language):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    URL_REGEX = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)
    if language == "EN":
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
        text = text.lower()
    elif language == "ZH":
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
        text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
        # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
        text = text.replace("转发微博", "")       # 去除无意义的词语
        text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
        text = re.sub(r'[^\u4e00-\u9fff]+', ' ',text)
    else:
        text=text
    return text.strip()

# Function to perform word segmentation and POS tagging
def segment_and_tag(text, language):
    if language == "ZH":
        words = pseg.cut(text)
        result = [(word, flag) for word, flag in words if word not in cn_stopwords]
    elif language == "EN":
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        result = [(word, tag) for word, tag in tagged_words if word not in en_stopwords]
    else:
        kiwi = Kiwi()
        tokens = kiwi.tokenize(text, stopwords=stopwords)
        # Construct the result in the desired format (Token and its attributes)
        result = [(token.form, token.tag) for token in tokens]
    return result

# Function to calculate word frequency
def calculate_frequency(tagged_words):
    freq_dict = {}
    for word, tag in tagged_words:
        if word in freq_dict:
            freq_dict[word]['frequency'] += 1
        else:
            freq_dict[word] = {'word_tag': tag, 'frequency': 1}
    return freq_dict
# Function to calculate sentiment score
def get_sentiment_score(text, language):
    if language == "ZH":
        return SnowNLP(text).sentiments
    elif language == "EN":
        return TextBlob(text).sentiment.polarity
    else:
        return 0
st.title("Text Processing App")

language = st.sidebar.selectbox("Select Language", ["ZH", "EN","KR"])

calculate_sentiment = st.sidebar.checkbox("Calculate Sentiment Score")

uploaded_file = st.file_uploader("Upload a TXT or DOCX file", type=["txt", "docx"])
st.divider()
with st.expander("A Powerful Text Process App !"):
    st.markdown("""
        ## Support Language : ZH(汉语)、 EN(英语)、 KR(韩语)
        ### Word Frequency 
        ### Text Segmentation
        ### POS Tagging
        """,    
            unsafe_allow_html=True)
if uploaded_file:
    try:
        if ".docx" in uploaded_file.name[-5:]  :  # Check for DOCX file
            text = read_docx(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
    else:
        # Read the file line by line and create a DataFrame
        lines = text.splitlines()
        df = pd.DataFrame(lines, columns=["content"])

        df=df.drop_duplicates(subset=['content'])
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(how='all')
        # Apply cleaning and segmentation to each line
        df['cleaned_content'] = df['content'].apply(lambda x: clean_text(x, language))
        df['posTag_content'] = df['cleaned_content'].apply(lambda x: " ".join([f"{word}/{tag}" for word, tag in segment_and_tag(x, language)]))
        if calculate_sentiment:
            df['sentiment_score'] = df['content'].apply(lambda x: get_sentiment_score(x, language))
        tagged_words=[tuple(item.split('/')) for sublist in df['posTag_content'].str.split(' ').tolist() for item in sublist]
        dd=list(filter(lambda x:len(x)>1,tagged_words))
        freq_dict = calculate_frequency(dd)
        sheet2_data = pd.DataFrame([
            {"words": word, "word_tag": data["word_tag"], "frequency": data["frequency"]}
            for word, data in freq_dict.items()
        ])
        # Displaying the top 10 rows of df and sheet2_data
        st.subheader("Processed Content Data (Top 10 rows)")
        st.dataframe(df.head(10))

        st.subheader("Word Frequency Data (Top 10 rows)")
        st.dataframe(sheet2_data.head(10))
        # Generate and display the word cloud
        st.subheader("Word Cloud")
        generate_wordcloud(sheet2_data)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="content", index=False)
            sheet2_data.to_excel(writer, sheet_name="frequency", index=False)
        output.seek(0)

        st.download_button(
            label="Download Results",
            data=output,
            file_name=f"{language}text_processing_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
