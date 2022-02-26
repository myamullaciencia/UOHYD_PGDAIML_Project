
import numpy as np
import pandas as pd
import numpy as np
import time
import streamlit as st
import altair as alt
import pickle
import os, requests

my_classifier_url = r'https://github.com/myamullaciencia/UOHYD_PGDAIML_Project/blob/main/sms_email_classifier_xgb.pkl?raw=true'
my_text_url = r'https://github.com/myamullaciencia/UOHYD_PGDAIML_Project/blob/main/sms_email_tfidf_vect_xgb.pkl?raw=true'

my_cwd = os.getcwd()

xgb_model_file = os.path.join(my_cwd,'sms_email_classifier_xgb.pkl')
text_vect_file = os.path.join(my_cwd,'sms_email_tfidf_vect_xgb.pkl')

my_model_resp = requests.get(my_classifier_url)
my_text_resp = requests.get(my_text_url)

with open('sms_email_tfidf_vect_xgb.pkl', 'wb') as fopen:
        fopen.write(my_text_resp.content)

with open('sms_email_classifier_xgb.pkl', 'wb') as fopen:
        fopen.write(my_model_resp.content)

with open(xgb_model_file, 'rb') as file:
	XGB_model_classifier = pickle.load(file)
 
with open(text_vect_file, 'rb') as file:
	text_tfidf_vectorizer = pickle.load(file) 

def model_predictor(text):
	txt_vec = text_tfidf_vectorizer.transform(text)
	txt_class = int(XGB_model_classifier.predict(txt_vec)[0])
	return txt_class

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="E-Mail and SMS Messaging Classifiation", page_icon="🔍", layout="wide")
hide_streamlit_style = """
            <style>
            footer {
	        visibility: hidden;
	            }
            footer:after {
	            content:'developed by Mallesham Yamulla'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;
                    }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.header("E-Mail and SMS Messaging Classifiation")

with st.form("Message classification"):
	txt_msg = st.text_input("Enter a text message to check it is a Spam or Legit?")
	submitted = st.form_submit_button("Submit")
	if submitted and len(txt_msg)>=32:
		msg_cls = model_predictor([txt_msg])
		if msg_cls==1:
			st.markdown(f'<h2 style="color:#de2d26;font-size:18px;">{"Spam !!!"}</h2>', unsafe_allow_html=True)
		else:
			st.markdown(f'<h2 style="color:#31a354;font-size:18px;">{"Clean !!!"}</h2>', unsafe_allow_html=True)
	elif submitted and len(txt_msg)>=1 and len(txt_msg)<32:
		st.markdown(f'<h1 style="color:#de2d26;font-size:18px;">{"Short messages cant be classified and its length should be more than 30 !!!"}</h1>', unsafe_allow_html=True)
	elif submitted and len(txt_msg)==0:
		st.markdown(f'<h1 style="color:#de2d26;font-size:18px;">{"Please Enter a text message !!!"}</h1>', unsafe_allow_html=True)
	else:
		pass
