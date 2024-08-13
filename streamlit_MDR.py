# 导入需要的库
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
label_columns = ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational', 'Emotional']

output = './downloaded_file.zip'
model_save_path = './content/saved_model/'
if not os.path.exists(model_save_path):
    import gdown
    import zipfile
    file_id = '1gp1P74uVeRFDuNf5P6-Av-0PjG10HqWA'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    # 读取压缩文件
    file=zipfile.ZipFile(output)
    file.extractall('./')
    # 关闭文件流
    file.close()
    os.remove(output)

if "model" not in st.session_state:
    with st.spinner('model load...'):
        # 加载保存的模型和 tokenizer
        st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(model_save_path)
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_save_path)
loaded_model = st.session_state["model"]
loaded_tokenizer = st.session_state["tokenizer"]

st.write(os.listdir('./'))
test_txt = st.text_area("请输入文本", None)
if st.button('predict'):
    # Tokenize新数据
    new_data = [test_txt]
    inputs = loaded_tokenizer(new_data, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    # 模型推理
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    # 通过阈值得到二进制预测
    preds = (logits > 0).int()
        st.write([v for _, v in zip(preds[0].tolist(), label_columns) if _ == 1])
