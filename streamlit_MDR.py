# 导入需要的库
import streamlit as st
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

# with st.spinner('please wait...'):
#     a = os.system("wget https://api.hostize.com/files/c97bBhCf0r/download/file.zip & unzip file.zip")
#     print(a)
label_columns = ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational', 'Emotional']
if not os.path.exists('./saved_model/model.safetensors'):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir='proj', num_labels=len(label_columns))
    model.save_pretrained('./saved_model/')
st.write(os.listdir('./saved_model/'))
# with st.spinner('model load...'):
#     model_save_path = "./content/saved_model"  # 指定保存路径
#     # 加载保存的模型和 tokenizer
#     loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
#     loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
#     # Tokenize新数据
#     new_data = ["Example text to classify"]
#     inputs = loaded_tokenizer(new_data, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
#     # 模型推理
#     outputs = loaded_model(**inputs)
#     logits = outputs.logits
#     # 通过阈值得到二进制预测
#     preds = (logits > 0).int()
#     st.write(preds)
    
