import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from datetime import date

st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Evaluez votre vélo','Test sur les données'])

# Fonction pour charger tous les modèles et les scalers
def load_models(model_name_list):
    model_dict  ={}
    scaler_dict ={}
    for model_name in model_name_list:
            model_dict[model_name] = joblib.load('assets/modeles/'+model_name+'.pkl')
            scaler_dict[model_name] = joblib.load('assets/modeles/'+model_name+'_scaler.pkl')
    return model_dict,scaler_dict

# Liste des noms de modèle
model_list = ['elasticnet','linearsvr','linearsvrtx','gradientboostingregressor','stackingregressor']
# Charger tous les modèles et scalers
model_dict,scaler_dict = load_models(model_list)

with tab1:
    # Chargement du dataset pour accéder à la liste des Marques, Catégories, Etat Vélo
    df = pd.read_csv('assets/dataset/data_ML.csv')
    liste_marques = sorted(list(df['Marque'].unique()))

    col1, col2 = st.columns([1,1])
    with col1:
        with st.form("my_form"):
            st.markdown("<h1 style='text-align: center;padding-top:0rem'>Evaluez votre vélo</h1>", unsafe_allow_html=True)
            marque=st.selectbox('Marque',liste_marques)
            catégorie = st.selectbox('Catégorie:',('Route','Aero / CLM','Gravel / CX','Piste','Trail / Enduro','Cross-country','Descente',
                                                'Dirt / Jump','Urbain','Hybrides','VTC','VTT','Loisir','Fatbike','Speedbike','Pliant'))
            electrique = st.radio("Electrique ?", ('Oui', 'Non'),horizontal=True)
            etatvelo = st.selectbox('État vélo:',('Comme Neuf','Très Bon État','Bon État','Usé'))
            annee=st.selectbox('Millésime:',[2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013],help="Le millésime d'un vélo peut être différent de son année d'achat. Par exemple les vélos de millésime 2023 sont en vente depuis Septembre 2022.")
            prixorigine=st.number_input('Prix achat vélo:',min_value=0, max_value=20000, step=1, format="%d", value=1500)
            # Every form must have a submit button.
            submitted = st.form_submit_button("Prédire")

    # Fonction pour calculer la prédiction de prix d'un vélo à partir des inputs et d'un modèle fitté
    def price_predict_inputs(model,scaler,marque,catégorie,electrique,etatvelo,annee,prixorigine,tcroix=False):
        bike_dict = {var:0 for var in model.feature_names_in_}
        var = 'Marque_'+marque
        if var in bike_dict.keys():
            bike_dict[var]=1
        var = 'Categorie_'+catégorie
        if var in bike_dict.keys():
            bike_dict[var]=1
        bike_dict['Etat_'+etatvelo]=1
        if electrique=='Oui':
            bike_dict['Elec_True']=1
        bike_dict['PrixOrigine']=prixorigine
        anciennete = max(0,((date.today() - date(annee,9,15)).days / 365))
        bike_dict['Anciennete']=anciennete

        if tcroix:
            var = 't_Marque_'+marque
            if var in bike_dict.keys():
                bike_dict[var]=anciennete
            var = 't_Categorie_'+catégorie
            if var in bike_dict.keys():
                bike_dict[var]=anciennete
            bike_dict['t_Etat_'+etatvelo]=1
            if electrique=='Oui':
                bike_dict['t_Elec_True']=anciennete

        bike_item = pd.DataFrame([bike_dict])
        model_var = model.feature_names_in_
        bike_item = bike_item[model_var]
        scaler_var = scaler.feature_names_in_
        bike_item.loc[:,scaler_var] = scaler.transform(bike_item[scaler_var])
        prediction = model.predict(bike_item)[0]
    
        return prediction

    # # Calcul des prédictions de tous les modèles sur les inputs
    with col2:
        if submitted:
            pred_inputs_perc = {}
            pred_inputs_eur = {}
            for model_name in model_list:
                if len(model_name)<11:
                    pred_inputs_perc[model_name]=price_predict_inputs(model_dict[model_name],scaler_dict[model_name],marque,catégorie,electrique,etatvelo,annee,prixorigine,tcroix=False)
                    pred_inputs_eur[model_name]=(1+pred_inputs_perc[model_name])*prixorigine
                else:
                    pred_inputs_perc[model_name]=price_predict_inputs(model_dict[model_name],scaler_dict[model_name],marque,catégorie,electrique,etatvelo,annee,prixorigine,tcroix=True)
                    pred_inputs_eur[model_name]=(1+pred_inputs_perc[model_name])*prixorigine         
            df_pred = pd.DataFrame({'Dépréciation %': list(pred_inputs_perc.values()), 'Prix EUR': list(pred_inputs_eur.values())}, index=list(pred_inputs_perc.keys()))
            df_pred['Dépréciation %'] = df_pred['Dépréciation %'].map(lambda x: '{:.1%}'.format(x))
            df_pred['Prix EUR'] = df_pred['Prix EUR'].map(lambda x: round(x))
            st.markdown("<h1 style='text-align: center;padding-top:0rem'>Prix estimés</h1>", unsafe_allow_html=True)
            st.dataframe(df_pred)

with tab2:
    # Import et preprocess des données
    features = pd.read_csv('assets/dataset/features_preprocessed.csv')
    target = pd.read_csv('assets/dataset/target.csv')
    target = target[["Depreciation%"]]
    features = features[features.columns[1:]]

    # Fonction pour calculer la prédiction de prix d'un item du dataset pour un modèle
    def price_predict_item(i,model,scaler):
        features_model = features[model.feature_names_in_]
        item = features_model.iloc[i].to_frame().T
        scaler_var = scaler.feature_names_in_
        item.loc[:,scaler_var] = scaler.transform(item[scaler_var])
        prediction = model.predict(item)[0]
        return prediction

    # Fonction pour afficher un item du dataset
    def shape_item(i):
        item = features.iloc[i].to_frame().T
        #item_non_nul=item.loc[:, (item != 0).any()]
        item_non_nul = item.loc[:, ~item.columns.str.startswith('t') & (item != 0).any()]
        return item_non_nul

    with st.form("my_form_2"):
        st.markdown("<h1 style='text-align: center;padding-top:0rem'>Choisissez un élément</h1>", unsafe_allow_html=True)
        n_item=st.number_input('Indice (0 à 4500):',min_value=0, max_value=4500, step=1, format="%d")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Prédiction")

    if submitted:
        pred_item_perc = {}
        pred_item_eur = {}
        item = shape_item(int(n_item))
        pred_item_perc['Réalité'] = target.iloc[n_item,0]
        pred_item_eur['Réalité'] = (1+pred_item_perc['Réalité'])*float(item['PrixOrigine'])
        for model_name in model_list:
            pred_item_perc[model_name]=price_predict_item(n_item,model_dict[model_name],scaler_dict[model_name])
            pred_item_eur[model_name]=(1+pred_item_perc[model_name])*float(item['PrixOrigine'])
        df_pred_item = pd.DataFrame({'Dépréciation %': list(pred_item_perc.values()), 'Prix EUR': list(pred_item_eur.values())}, index=list(pred_item_perc.keys()))
        df_pred_item['Dépréciation %'] = df_pred_item['Dépréciation %'].map(lambda x: '{:.1%}'.format(x))
        df_pred_item['Prix EUR'] = df_pred_item['Prix EUR'].map(lambda x: round(x))
        st.markdown("<h1 style='text-align: center;padding-top:0rem'>Prix estimés</h1>", unsafe_allow_html=True)
        st.dataframe(item)
        st.dataframe(df_pred_item)
        
            