import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
        page_title="BDS_Bikes",
        page_icon="bike",
        layout="wide"
    )

# Settings
st.markdown(""" <style> .main {padding-top: 0rem; padding-bottom: 0rem;padding-left:17rem;padding-right:13rem;} </style> """, unsafe_allow_html=True)
st.markdown("""
    <style>
    p {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
     st.image('assets/images/cyclop.png')
     st.write(" ")
     st.write(" ")
     menu = option_menu(None, ["Home", "Exploration des données", "Modelisation","Méthodologie","Interprétation","Prédiction","Conclusions & Perspectives"], 
         icons=['house', "search", "graph-up","graph-up-arrow","search",'bicycle',"card-checklist"], 
         menu_icon="cast", default_index=0,styles={
        "container": {"font-family": "Arial"},
        "icon": {"font-size": "20px"},
        "nav-link-selected": {"background-color": "#2596be"}
         })
     st.write(" ")
     st.write(" ")
     st.info("""Projet DS - Promotion Bootcamp Avril 2023

Participants :

- David Serruya

- Jean-Nicolas Lamarre

- Yves Le Nouveau""")

if menu=="Home":
    with open("views/home.py", "r", encoding="utf-8") as file:
        exec(file.read())

elif menu=="Exploration des données":
    with open("views/exploration.py", "r", encoding="utf-8") as file:
        exec(file.read())

elif menu=="Modelisation":
    with open("views/modelisation.py", "r", encoding="utf-8") as file:
        exec(file.read()) 

elif menu=="Méthodologie":
    with open("views/methodologie.py", "r", encoding="utf-8") as file:
        exec(file.read())

elif menu=="Conclusions & Perspectives":
    with open("views/conclusion.py", "r", encoding="utf-8") as file:
        exec(file.read())

elif menu=="Prédiction":
    with open("views/prediction.py", "r", encoding="utf-8") as file:
        exec(file.read())

elif menu=="Interprétation":
    with open("views/interpretation.py", "r", encoding="utf-8") as file:
        exec(file.read())





