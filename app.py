import streamlit as st
import pandas as pd
import nltk
import jieba
import jieba.posseg as pseg
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import requests
from io import BytesIO
import re
from snownlp import SnowNLP
from textblob import TextBlob
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
# Load stopwords
cn_stopwords_url = "https://raw.githubusercontent.com/CharyHong/Stopwords/main/stopwords_cn.txt"
en_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"

cn_stopwords = requests.get(cn_stopwords_url).text.splitlines()
en_stopwords = requests.get(en_stopwords_url).text.splitlines()

# Function to clean text
def clean_text(text, language):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    if language == "英文":
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = text.lower()
    else:
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
        text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
        # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
        URL_REGEX = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            re.IGNORECASE)
        text = re.sub(URL_REGEX, "", text)       # 去除网址
        text = text.replace("转发微博", "")       # 去除无意义的词语
        text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
        text = re.sub(r'[^\u4e00-\u9fff]+', ' ',text)
    return text.strip()

# Function to perform word segmentation and POS tagging
def segment_and_tag(text, language):
    if language == "中文":
        words = pseg.cut(text)
        result = [(word, flag) for word, flag in words if word not in cn_stopwords]
    else:
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        result = [(word, tag) for word, tag in tagged_words if word not in en_stopwords]
    return result

# Function to calculate word frequency
def calculate_frequency(tagged_words):
    freq_dict = {}
    print(tagged_words)
    for word, tag in tagged_words:
        if word in freq_dict:
            freq_dict[word]['frequency'] += 1
        else:
            freq_dict[word] = {'word_tag': tag, 'frequency': 1}
    return freq_dict
# Function to calculate sentiment score
def get_sentiment_score(text, language):
    if language == "中文":
        return SnowNLP(text).sentiments
    else:
        return TextBlob(text).sentiment.polarity
    
st.title("Text Processing App")

language = st.sidebar.selectbox("Select Language", ["中文", "英文"])

calculate_sentiment = st.sidebar.checkbox("Calculate Sentiment Score")

uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

if uploaded_file:
    try:
        text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The uploaded file is not UTF-8 encoded. Please upload a UTF-8 encoded TXT file.")
    else:
        # Read the file line by line and create a DataFrame
        lines = text.splitlines()
        df = pd.DataFrame(lines, columns=["content"])

        df=df.drop_duplicates(subset=['content'])
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
