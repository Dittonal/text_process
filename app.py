import streamlit as st
import pandas as pd
import jieba
import jieba.posseg as pseg
from io import BytesIO

# Function to clean text
def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

# Function to perform word segmentation and POS tagging
def segment_and_tag(text):
    words = pseg.cut(text)
    result = [(word, flag) for word, flag in words]
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

# Streamlit UI
st.title("Text Processing App")

# Sidebar language selection
language = st.sidebar.selectbox("Select Language", ["中文", "英文"])

# File uploader
uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8", errors='ignore')
    cleaned_text = clean_text(text)

    # Perform segmentation and POS tagging
    tagged_words = segment_and_tag(cleaned_text)

    # Data for sheet1
    sheet1_data = pd.DataFrame({
        "content": [text],
        "cleaned_content": [cleaned_text],
        "posTag_content": [" ".join([f"{word}/{tag}" for word, tag in tagged_words])]
    })

    # Data for sheet2
    freq_dict = calculate_frequency(tagged_words)
    sheet2_data = pd.DataFrame([
        {"words": word, "word_tag": data["word_tag"], "frequency": data["frequency"]}
        for word, data in freq_dict.items()
    ])

    # Button to download Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet1_data.to_excel(writer, sheet_name="content", index=False)
        sheet2_data.to_excel(writer, sheet_name="frequency", index=False)
    output.seek(0)

    st.download_button(
        label="Download Results",
        data=output,
        file_name="text_processing_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
