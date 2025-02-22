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
from wordcloud import WordCloud
from mplfonts import use_font
import matplotlib.pyplot as plt

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
FONT_PATH="/home/adminuser/venv/lib/python3.12/site-packages/mplfonts/fonts/SourceHanMonoSC-Regular.otf"
# Function to generate and display the word cloud
@st.cache_data
def generate_wordcloud(frequency_data):
    # Join the words and frequencies to form the text for the word cloud
    word_freq = {row['words']: row['frequency'] for index, row in frequency_data.iterrows()}

    # Generate the word cloud using the word frequency data
    wordcloud = WordCloud(width=800,
                          height=400,
                          font_path=FONT_PATH,
                          background_color='white').generate_from_frequencies(word_freq)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Display the word cloud using matplotlib
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # Turn off axis
    st.pyplot(fig)  # Display the word cloud in Streamlit
@st.cache_data 
def generate_excel(df, sheet2_data,number_control,lenth_control):
    sheet2_data['words'] = sheet2_data['words'].str.strip()
    sheet2_data['words']=sheet2_data['words'].astype(str)
    if number_control: sheet2_data = sheet2_data[~sheet2_data['words'].str.match(r'^[\d.]+$')]  # 匹配数字和小数
    if lenth_control: sheet2_data = sheet2_data[sheet2_data['words'].str.len() >= 2]
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="content", index=False)
        
        sheet2_data.to_excel(writer, sheet_name="frequency", index=False)
    output.seek(0)
    return output
def generate_png(frequency_data):
    # Save the word cloud as a PNG file with transparent background
    word_freq = {row['words']: row['frequency'] for index, row in frequency_data.iterrows()}
    img_buffer = BytesIO()
    wordcloud = WordCloud(background_color = None,
                          mode='RGBA',
                          width = 800, 
                          height = 600,
                          margin = 3,
                          font_path=FONT_PATH,
                          max_font_size=100,scale=10).generate_from_frequencies(word_freq)
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer
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
number_control = st.sidebar.checkbox("Filter Number Keyword",value=True)
lenth_control = st.sidebar.checkbox("Filter Lenth lt 2",value=True)

uploaded_file = st.file_uploader("Upload a TXT or DOCX file", type=["txt", "docx"])
# Display the selected language as a message
# Show the selected language in red
st.markdown(f"<h3 style='color:red;'>Current Selected Language: {language}</h3>", unsafe_allow_html=True)

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
        st.dataframe(df.head(10),use_container_width=True)

        st.subheader("Word Frequency Data (Top 10 rows)")
        st.dataframe(sheet2_data.head(10),use_container_width=True)
        # Generate and display the word cloud
        st.subheader("Word Cloud")
        generate_wordcloud(sheet2_data)
        # Add a download button to allow users to download the PNG image
        st.subheader("Download Word Cloud as PNG")
        png_data=sheet2_data.sort_values(by='words',ascending=False).iloc[:40,:]
        
        img_buffer = generate_png(png_data)  # Generate PNG only when the user clicks the button
        st.download_button(
            label="Download Word Cloud as PNG",
            data=img_buffer,
            file_name=f"{language}wordcloud.png",
            mime="image/png"
        )
        output = BytesIO()
        excel_data = generate_excel(df, sheet2_data,number_control,lenth_control)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"{language}text_processing_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
