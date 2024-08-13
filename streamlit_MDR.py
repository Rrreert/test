# 导入需要的库
import streamlit as st
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

label_columns = ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational', 'Emotional']
import gdown

# 替换为你的文件 ID
file_id = '1-RPL_mDqOJ1uIe-NF06ZvlVa5hRhV0Mo'
url = f'https://drive.google.com/uc?id={file_id}'

# 替换为你希望保存的文件名
output = './downloaded_file.zip'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# if not os.path.exists('./saved_model/model.safetensors'):
#     model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir='proj', num_labels=len(label_columns))
#     model.save_pretrained('./saved_model/')
# import zipfile

# # 读取压缩文件
# file=zipfile.ZipFile(output)
# file.extractall('./')
# # 关闭文件流
# file.close()


st.write(os.listdir('./content/saved_model/'))
# test_txt = "I don‚Äôt understand how I‚Äôm feeling and all I can describe it as is numbness but it‚Äôs past that point and I‚Äôve felt like this for a long time, I feel like I don‚Äôt belong to this life like it isn‚Äôt for me. I can‚Äôt see myself in any career, my own family my own little life I can‚Äôt see it , I‚Äôm so disconnected from social interaction I don‚Äôt leave my house much and the sad thing is as much as I hate it I don‚Äôt want to change it I have no motivation I‚Äôm so tired to the point I don‚Äôt see a point on living when I‚Äôm so tired I can‚Äôt do daily life like everyone. What is the point to this life? How do you really find happiness I feel nothing I get the occasional anger and I‚Äôm always Irritated but besides that I feel nothing and I hate it I can‚Äôt cry I can‚Äôt laugh I can‚Äôt feel anything"
# with st.spinner('model load...'):
#     model_save_path = "./saved_model"  # 指定保存路径
#     # 加载保存的模型和 tokenizer
#     loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
#     loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
#     # Tokenize新数据
#     new_data = [test_txt]
#     inputs = loaded_tokenizer(new_data, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
#     # 模型推理
#     outputs = loaded_model(**inputs)
#     logits = outputs.logits
#     # 通过阈值得到二进制预测
#     preds = (logits > 0).int()
#     st.write(preds)
    
