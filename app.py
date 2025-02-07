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
    for word, tag in tagged_words:
        if word in freq_dict:
            freq_dict[word]['frequency'] += 1
        else:
            freq_dict[word] = {'word_tag': tag, 'frequency': 1}
    return freq_dict

st.title("Text Processing App")

language = st.sidebar.selectbox("Select Language", ["中文", "英文"])

uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8", errors='ignore')
    cleaned_text = clean_text(text, language)

    tagged_words = segment_and_tag(cleaned_text, language)

    sheet1_data = pd.DataFrame({
        "content": [text],
        "cleaned_content": [cleaned_text],
        "posTag_content": [" ".join(f"{word}/{tag}" for word, tag in tagged_words)]
    })

    freq_dict = calculate_frequency(tagged_words)
    sheet2_data = pd.DataFrame([
        {"words": word, "word_tag": data["word_tag"], "frequency": data["frequency"]}
        for word, data in freq_dict.items()
    ])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet1_data.to_excel(writer, sheet_name="content", index=False)
        sheet2_data.to_excel(writer, sheet_name="frequency", index=False)
    output.seek(0)

    st.download_button(
        label="Download Results",
        data=output,
        file_name=f"{language}text_processing_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
