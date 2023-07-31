import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import shapiro, spearmanr, kendalltau
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import chi2_contingency


df=pd.read_csv('assets/dataset/dataset_finale.csv')
df2=pd.read_csv('assets/dataset/data_ML.csv')

ligne=df2.shape[0]
col=df2.shape[1]


st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Dataset','Variables','Conclusions'])

with tab1:

    st.markdown('<h2 style="font-family:optima;color:#2596be;">Jeu de données</h2>', unsafe_allow_html=True)

    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">1. Source</h3>', unsafe_allow_html=True)

    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Il n’existe pas de source de données de transactions effectives</strong>, 
    en revanche quelques sites présentent une longue liste d’annonces de vélo à vendre avec en plus du prix demandé de nombreuses caractéristiques répertoriées (marque, modèle, année de fabrication…) . 
    Les deux principaux sites sont <strong>LeBonCoin et Troc-Vélo</strong>. Troc-Vélo étant un site spécialisé pour les vélos et le matériel cycliste, 
    il constitue le meilleur choix comme source de données par <strong>l’utilisation de Webscraping</strong> : les données y sont mieux structurées, 
    plus complètes et le taux de d’annonces frauduleuses y est nettement plus bas.</p>""", unsafe_allow_html=True)

    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">2. Période</h3>', unsafe_allow_html=True)

    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Un premier jeu de données obtenues par Webscraping de Troc-Vélo en <strong>Janvier 2023</strong> est directement utilisable . 
    Il comporte environ <strong>4500 annonces</strong> présentes sur le site en Janvier 2023, mises en ligne <strong>au maximum 1,5 mois</strong> auparavant et suffisamment renseignées pour être exploitables.
    </p>""", unsafe_allow_html=True)

    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">3. Propriété des données</h3>', unsafe_allow_html=True)

    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le jeu de données est la propriété de la société <strong>CyclOp Vélos</strong>.
    </p>""", unsafe_allow_html=True)

    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">4. Exploration des données</h3>', unsafe_allow_html=True)

    st.markdown(f"""<p style="font-family:Arial,sans-serif;text-align: justify;">
    Langage utilisé : <strong>Python</strong>
    <br>Librairies utilisées : <strong>Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn</strong>
    <br><br><strong>Taille du DataFrame</strong>
    <br> {ligne} lignes x {col} colonnes</p>""", unsafe_allow_html=True)

    st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Les variables</strong></p>""" ,unsafe_allow_html=True)
    
    
    st.markdown(""" 
    |  N° colonne |	Nom de la colonne |	Type | 
    | --------- | --------- | --------- |
    | 1 | IdAnnonce : ID de l'annonce | int64  |
    | 2 | Nom : Titre de l'annonce  | object |
    | 3  | Annee : Année d'achat du vélo  | int64  |
    | 4  | Prix : Prix de revente du vélo | float64  |
    | 5 | PrixOrigine : Prix d'achat du vélo  | float64  |
    | 6  | TypePrix : Prix négociable ou ferme  | object |
    | 7  | Marque : Marque du vélo  | object  |
    | 8  | Catégorie : Catégorie du vélo | object  |
    | 9  | JourIntegration : Jour d'intégration de l'annonce dans le dataset  | object  |
    | 10  | EtatVelo : L'état du vélo au moment de l'annonce sous la forme numérique  | int64  |
    | 11  | FlagElectrique : Si le vélo est électrique  | int64  |
    | 12  | EtatVeloLabel : L'état du vélo au moment de l'annonce sous la forme de label  | object  |
    """ ,unsafe_allow_html=True)

    st.markdown('<br><h3 style="font-family:optima;color:#00c4cc;">5. Ajout de variables</h3>', unsafe_allow_html=True)

    st.markdown(f"""<p style="font-family:Arial,sans-serif;text-align: justify;">
    Pour répondre à nos besoins, nous avons crée <strong>3 nouvelles variables: </strong>
    <br><strong>- Depreciation%</strong>: La dépréciation relative fait référence à la diminution de la valeur d'un vélo par rapport à sa valeur initiale, exprimée en pourcentage.Il s'agit potentiellement de notre variable cible (Partie Modèlisation).
    <br><strong>- DepreciationEUR</strong>: La dépréciation en euro fait référence à la diminution de la valeur d'un vélo par rapport à sa valeur initiale, exprimée en euro.Il s'agit potentiellement de notre variable cible (Partie Modèlisation).              
    <br><strong>- Anciennete</strong> : L'ancienneté tend à approximer l'usure du vélo.""", unsafe_allow_html=True)

    st.markdown('<br><h3 style="font-family:optima;color:#00c4cc;">6. Supression de variables</h3>', unsafe_allow_html=True)

    st.markdown(f"""<p style="font-family:Arial,sans-serif;text-align: justify;">Pour différentes raisons, nous avons décidé de ne pas garder ces variables pour la suite de nos travaux:<br>
      - Variable(s) non pertinente pour les modèles: <strong>IdAnnonce</strong>, <strong>Nom</strong>, <strong>TypePrix</strong><br>
      - Variable(s) de référence pour la création des nouvelles variables : <strong>Prix</strong>, <strong>JourIntegration</strong>, <strong>Annee</strong><br>
      - Variable(s) remplacé(s): <strong>EtatVelo par EtatVeloLabel</strong>""", unsafe_allow_html=True)

    st.markdown('<br><h3 style="font-family:optima;color:#00c4cc;">7. Dataset Final</h3>', unsafe_allow_html=True)

    st.markdown(f"""<p style="font-family:Arial,sans-serif;">Voici un aperçu du dataset final:""", unsafe_allow_html=True)

    df_finale = pd.read_csv('assets/dataset/dataset_finale.csv')

    st.dataframe(df_finale.head(10))

with tab2:
    option = st.selectbox('',('Anciennete', 'EtatVeloLabel', 'FlagElectrique','Categorie','Marque','PrixOrigine','DepreciationEUR','Depreciation%'),index=0)
    st.markdown('<h2 style="font-family:optima;color:#2596be;">Définition</h2>', unsafe_allow_html=True)
    if option=='Categorie' or option=='EtatVeloLabel' or option=='FlagElectrique' or option=='Marque':
           if option=='Categorie':
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable Categorie dans notre dataset indique la catégorie à laquelle chaque vélo appartient. 
                Cette information peut être pertinente pour évaluer la dépréciation des vélos, car les différents types de vélos ont des caractéristiques 
                et des utilisations distinctes qui peuvent influencer leur valeur à long terme.</p><br>""", unsafe_allow_html=True)
           elif option=='Marque':
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable Marque dans notre dataset indique la marque associée à chaque vélo. 
                Cette variable peut jouer un rôle important dans la détermination de la dépréciation d'un vélo, 
                car la marque peut influencer sa qualité, sa durabilité, sa disponibilité et sa réputation sur le marché</p><br>""", unsafe_allow_html=True)  
           elif option=='EtatVeloLabel':
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable EtatVelo dans notre dataset renseigne l'état de vétusté de chaque vélo, fourni par l'annonceur. 
                Cette variable est essentielle pour évaluer la dépréciation d'un vélo, car elle influence directement sa valeur marchande sur le marché.
                Ainsi, la variable EtatVelo constitue notre meilleur indicateur pour déterminer l'état général du vélo dans notre analyse.</p><br>""", unsafe_allow_html=True)
           else: 
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable FlagElectriqueFinal dans notre dataset indique si le vélo possède une assistance électrique. 
                Les vélos électriques ont un prix plus élevé en raison de la technologie intégrée, et leur valeur peut être influencée par l'évolution du marché, l'obsolescence technologique et l'état de la batterie. 
                Il est donc important de prendre en compte cette distinction pour évaluer la dépréciation des vélos. </p><br>""", unsafe_allow_html=True)

           st.markdown('<h2 style="font-family:optima;color:#2596be;">Exploration variable</h2>', unsafe_allow_html=True)

           col1, col2 = st.columns(2)
           with col1:
                type='Qualitative'
                NB_modalite=df[option].nunique()
                modalite=df[option].unique()
                na=df[option].isna().sum()
                taux_na=(df[option].isna().sum())/(len(df.index))
                st.markdown("""<br>""", unsafe_allow_html=True)    
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Type de variable: </strong>{}</p>""".format(type), unsafe_allow_html=True)                
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Nombre de modalités: </strong>{}</p>""".format(NB_modalite), unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Les modalités: </strong>{}</p>""".format(modalite), unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Nombre de valeurs manquantes: </strong>{}</p>""".format(na), unsafe_allow_html=True)  
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Pourcentage de valeurs manquantes: </strong>{} %</p>""".format(taux_na), unsafe_allow_html=True)               

           with col2:      
                fig, ax = plt.subplots(figsize=(20, 12))
                sns.countplot(x=option, data=df, ax=ax)
                plt.title("Répartition des vélos en fonction de "+str(option))

                total = len(df[option])
                for p in ax.patches:
                   percentage = f'{100 * p.get_height() / total:.1f}%\n'
                   x = p.get_x() + p.get_width() / 2
                   y = p.get_height()
                   ax.annotate(percentage, (x, y), ha='center', va='center')

                plt.tight_layout()
                st.pyplot(fig)
            
           st.markdown('<h2 style="font-family:optima;color:#2596be;">Test Statistique</h2>', unsafe_allow_html=True) 

           if option!='FlagElectrique':
                st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Kruskal-Wallis (test d\'association)</h3>', unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                • H0 : Les distributions de tous les groupes sont égales.<br>
                • H1 : Au moins une des distributions des groupes est différente des autres.</p>""", unsafe_allow_html=True)

                conditions = st.checkbox('Conditions d\'utilisation')

                if conditions:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les observations sont indépendantes.<br>
                           • Les données peuvent être ordonnées (au moins ordinale).<br>
                           • Plus de deux modalités sur une variable.</p>""", unsafe_allow_html=True) 

                
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong><br>""", unsafe_allow_html=True)
                fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                sns.boxplot(x=option, y='DepreciationEUR', data=df, ax=axs[0])
                axs[0].set_xticklabels(df[option].unique(), rotation=45)
                axs[0].set_title('DepreciationEUR en fonction de la variable '+str(option))
                sns.boxplot(x=option, y='Depreciation%', data=df, ax=axs[1])
                axs[1].set_xticklabels(df[option].unique(), rotation=45)
                axs[1].set_title('Depreciation% en fonction de la variable '+str(option))
                st.pyplot(fig)

                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)
                etats = df[option].unique()
                data1 = [df[df[option] == etat]['DepreciationEUR'] for etat in etats]
                statistic1, pvalue1 = kruskal(*data1)
                data2 = [df[df[option] == etat]['Depreciation%'] for etat in etats]
                statistic2, pvalue2 = kruskal(*data2)
                df_stats = pd.DataFrame({'Variable Cible':['DepreciationEUR','Depreciation%'],'Statistique du test de Kruskal-wallis': [statistic1,statistic2], 'P-value': [pvalue1,pvalue2]})
                df_stats = df_stats.astype(str)
                st.dataframe(df_stats, height=100)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La P-value est inférieure à 0.05 : au moins une des distributions de Categorie est différente des autres.<br>
                Par contre, ce test n'identifie pas où cette dominance se produit ni pour combien de paires de groupes la dominance s'obtient.<br>""", unsafe_allow_html=True)


           else:
               st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Mann-Withney (test d\'association)</h3>', unsafe_allow_html=True) 
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                • H0 : Les distributions des deux groupes sont égales.<br>
                • H1 : Les distributions des deux groupes sont différentes.</p>""", unsafe_allow_html=True)

               conditions = st.checkbox('Conditions d\'utilisation')

               if conditions:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les observations sont indépendantes.<br>
                           • Les données peuvent être ordonnées (au moins ordinale).<br>
                           • Seulement deux modalités sur une variable.</p>""", unsafe_allow_html=True) 
               
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong><br>""", unsafe_allow_html=True)
               fig, axs = plt.subplots(1, 2, figsize=(15, 5))
               sns.boxplot(x=option, y='DepreciationEUR', data=df, ax=axs[0])
               axs[0].set_xticklabels(df[option].unique(), rotation=45)
               axs[0].set_title('DepreciationEUR en fonction de la variable '+str(option))
               sns.boxplot(x=option, y='Depreciation%', data=df, ax=axs[1])
               axs[1].set_xticklabels(df[option].unique(), rotation=45)
               axs[1].set_title('Depreciation% en fonction de la variable '+str(option))
               st.pyplot(fig)

               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)
               etats = df[option].unique()
               data1 = [df[df[option] == etat]['DepreciationEUR'] for etat in etats]
               statistic1, pvalue1 = mannwhitneyu(*data1)
               data2 = [df[df[option] == etat]['Depreciation%'] for etat in etats]
               statistic2, pvalue2 = mannwhitneyu(*data2)
               df_stats = pd.DataFrame({'Variable Cible':['DepreciationEUR','Depreciation%'],'Statistique du test de Mann-Whitney': [statistic1,statistic2], 'P-value': [pvalue1,pvalue2]})
               df_stats = df_stats.astype(str)
               st.dataframe(df_stats, height=100)
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La P-value est inférieure à 0.05 : au moins une des distributions de Categorie est différente des autres.<br>
               Par contre, ce test n'identifie pas où cette dominance se produit ni pour combien de paires de groupes la dominance s'obtient.<br>""", unsafe_allow_html=True)
                

    else:

           if option=='Anciennete':
                
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les variables explicatives utilisées en priorité par les experts métier pour estimer la dépréciation d'un vélo d'occasion par rapport à son prix d'origine sont:<br><br>
                            - <strong>L'usure</strong><br>
                            -<strong> Le millésime</strong> qui reflète l'obsolescence éventuelle de certaines technologies.<br><br>
                            L'usure peut-être approximée par le kilométrage, qui lui-même peut-être approximé par l'ancienneté au sens de la différence entre la date de mise en annonce vélo et la date de sa mise en annonce.
                            La création de cette varaible nous permettra donc de capturer les effets d'usure et de vieillissement des technologies.""", unsafe_allow_html=True)
                st.markdown('<h2 style="font-family:optima;color:#2596be;">Création de la variable</h2>', unsafe_allow_html=True) 
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous allons définir la variable ancienneté à partir des variables <strong>JourIntegration</strong> et <strong>Annee</strong> du dataset. <br>
                Le jour d'intégration est notre meilleur proxy pour connaitre la date de mise en annonce et l'année correspondant au millésime est notre meilleur proxy pour la date de mise en service du vélo.""", unsafe_allow_html=True)  
                st.markdown('<h2 style="font-family:optima;color:#2596be;">Formule</h2>', unsafe_allow_html=True) 
                st.latex(r'''df['Anciennete'] = ((df['JourIntegration'] - pd.to_datetime(df["Annee"].astype(str) + "-09-15")).dt.days / 365).apply(lambda x: max(0, x))''')
              
           elif option=='Depreciation%':
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La dépréciation relative fait référence à la diminution de la valeur d'un vélo par rapport à sa valeur initiale, exprimée en pourcentage.Il s'agit potentiellement de notre variable cible (Partie Modèlisation).<p><br>""", unsafe_allow_html=True)  
                st.markdown('<h2 style="font-family:optima;color:#2596be;">Formule</h2>', unsafe_allow_html=True) 
                st.latex(r'''df["DepreciationRel"] = (df["Prix"]/df["PrixOrigine"])-1''')
           elif option=='DepreciationEUR':
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La dépréciation en euro fait référence à la diminution de la valeur d'un vélo par rapport à sa valeur initiale, exprimée en euro.Il s'agit potentiellement de notre variable cible (Partie Modèlisation).</p><br>""", unsafe_allow_html=True)
                st.markdown('<h2 style="font-family:optima;color:#2596be;">Formule</h2>', unsafe_allow_html=True) 
                st.latex(r'''df["DepreciationEUR"] = df["Prix"] - df["PrixOrigine"]''')
           st.markdown('<h2 style="font-family:optima;color:#2596be;">Exploration variable</h2>', unsafe_allow_html=True)    

           col_2_1, col_2_2 = st.columns(2)
           with col_2_1:
                type='Quantitative'
                max=df[option].max()
                min=df[option].min()
                na=df[option].isna().sum()
                taux_na=(df[option].isna().sum())/(len(df.index))
                normale="La variable n\'est pas normalement distribuée (Test Shapiro)"
                st.markdown("""<br>""", unsafe_allow_html=True)    
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Type de variable: </strong>{}</p>""".format(type), unsafe_allow_html=True)                
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Valeur minimale: </strong>{}</p>""".format(min), unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Valeur maximale: </strong>{}</p>""".format(max), unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Nombre de valeurs manquantes: </strong>{}</p>""".format(na), unsafe_allow_html=True)  
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Pourcentage de valeurs manquantes: </strong>{} %</p>""".format(taux_na), unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;"><strong>Normalité: </strong>{}</p>""".format(normale), unsafe_allow_html=True)                 

           with col_2_2:        
               plt.figure(figsize=(20, 12))
               sns.histplot(df[option], kde=True)
               plt.xlabel(option)
               plt.ylabel('Fréquence')
               plt.title("Histogramme et courbe de densité pour la variable " + str(option))
               st.pyplot(plt)
            

           if option == 'Anciennete':
               st.markdown('<h2 style="font-family:optima;color:#2596be;">Test Statistique</h2>', unsafe_allow_html=True)  
               st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Spearman & Kendall (test de corrélation)</h3>', unsafe_allow_html=True) 
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                • H0 : Il n'y a pas de <strong>corrélation monotone</strong> entre les deux variables.<br>
                • H1 : Il y a une corrélation monotone entre les deux variables.</p>""", unsafe_allow_html=True) 

               conditions = st.checkbox('Conditions d\'utilisation')

               if conditions:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les deux variables sont quantitatives ou ordinales.<br>
                           • Les données n'ont pas besoin d'être normalement distribuées.<br>
                           • La relation entre les variables peut être monotone (linéaire ou non linéaire).</p>""", unsafe_allow_html=True)                       
               
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong><br>""", unsafe_allow_html=True)
               fig, ax = plt.subplots(2, 1, figsize=(15, 8))
               sns.scatterplot(x='Anciennete', y='DepreciationEUR', data=df, ax=ax[0], s=10)
               sns.regplot(x='Anciennete', y='DepreciationEUR', data=df, ax=ax[0], scatter=False, line_kws={'color':'k'})
               ax[0].set_ylim(-10000, 300)
               ax[0].set_xlim(None) 
               sns.scatterplot(x='Anciennete', y='Depreciation%', data=df, ax=ax[1], s=10)
               sns.regplot(x='Anciennete', y='Depreciation%', data=df, ax=ax[1], scatter=False, line_kws={'color':'k'})
               ax[1].set_ylim(-0.9, 0.15)
               ax[1].set_xlim(None)  
               st.pyplot(fig)

               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)

               interpretation = st.checkbox('Interprétation des résultats')

               if interpretation:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Interprétation des résultats</strong><br>
                           • Si le coefficient est proche de 1 (un seuil à 70% pour Kendall et 60% pour Spearman), il y a une corrélation positive entre les variables.<br>
                           • Si le coefficient est proche de -1 (un seuil à 70% pour Kendall et 60% pour Spearman), il y a une corrélation négative entre les variables.<br>
                           • Si le coefficient est proche de 0, il n'y a pas de corrélation monotone entre les variables.</p>""", unsafe_allow_html=True)
                    
               coeff_spearmanEUR, p_value_spearmanEUR = spearmanr(df['DepreciationEUR'], df['Anciennete'])
               coeff_spearmanREL,p_value_spearmanREL=spearmanr(df['Depreciation%'],df['Anciennete'])
               coeff_kendallEUR,p_value_kendallEUR=kendalltau(df['DepreciationEUR'],df['Anciennete'])
               coeff_kendallREL,p_value_kendallREL=kendalltau(df['Depreciation%'],df['Anciennete'])
               df_stats = pd.DataFrame({'Variable Cible':['DepreciationEUR','Depreciation%','DepreciationEUR','Depreciation%'],'Statistique du test de Spearman & Kendall': [coeff_spearmanEUR,coeff_spearmanREL,coeff_kendallEUR,coeff_kendallREL], 'P-value': [p_value_spearmanEUR,p_value_spearmanREL,p_value_kendallEUR,p_value_kendallREL]})
               df_stats = df_stats.astype(str)
               st.dataframe(df_stats, height=200)
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Spearman</strong><br> La P-value est inférieure à 0.05 dans les deux cas : <strong>il y a une corrélation entre les deux variables</strong>.<br>
                               La statistique du test est en-dessous de 0 dans les deux cas : <strong>il y a une corrélation négative entre les deux variables</strong>.
                               La corrélation est plus prononcée avec <strong>la dépréciation relative</strong>.<br>""", unsafe_allow_html=True)
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Kendall</strong><br> La P-value est inférieure à 0.05 dans les deux cas : <strong>il y a une corrélation entre les deux variables</strong>.<br>
                               La statistique du test est en-dessous de 0 dans les deux cas : <strong>il y a une corrélation négative entre les deux variables</strong>.
                               La corrélation est plus prononcée avec <strong>la dépréciation relative</strong>.<br>""", unsafe_allow_html=True)



with tab3:
    
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Dans cette section, nous allons effectuer un récapitulatif des variables et du jeu de données final qui sera utilisé pour entraîner nos modèles. Il servira de base solide pour la suite de notre analyse et de notre processus de modélisation. <br>""", unsafe_allow_html=True)                

    st.markdown('<h2 style="font-family:optima;color:#2596be;">Tableau Récapitulatif</h2>', unsafe_allow_html=True)

    st.markdown(""" 
    | 	Caractéristiques |	Résultats | 
    |  --------- | --------- |
    | Nombre de variables| 8 variables  |
    | Nombre d'annonces | 4531 lignes  |    
    | Liste des variables| Categorie, Marque, EtatVeloLabel, FlagElectrique, PrixOrigine, Anciennete, DepreciationEUR, Depreciation% |
    | Liste des variables qualititatives| Categorie, Marque, EtatVeloLabel, FlagElectrique|
    | Liste des variables quantitatives| PrixOrigine, Anciennete, DepreciationEUR, Depreciation% |
    | Variables cibles potentielles| DepreciationEUR, Depreciation% |
    | Variables distribuées de manière équilibrée  | FlagElectrique,EtatVeloLabel  |
    | Variables distribuées de manière inéquilibrée   | Marque, Categorie |
    | Variables suivant une loi normale | D'après le test de Shapiro, aucune variable semble suivre une loi normale |
    """ ,unsafe_allow_html=True)

    st.markdown('<h2 style="font-family:optima;color:#2596be;"><br>Statistiques globales</h2>', unsafe_allow_html=True)
    option = st.selectbox('',('Variables Continues/Variables Cibles','Variables Catégorielles/Variables Cibles','Variables Catégorielles/Variables Catégorielles'),index=0)
    
    if option=='Variables Continues/Variables Cibles':
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les variables continues sont : Anciennete</p>""", unsafe_allow_html=True) 
               st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Spearman & Kendall (test de corrélation)</h3>', unsafe_allow_html=True) 
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                • H0 : Il n'y a pas de <strong>corrélation monotone</strong> entre les deux variables.<br>
                • H1 : Il y a une corrélation monotone entre les deux variables.</p>""", unsafe_allow_html=True) 

               conditions2 = st.checkbox('Conditions d\'utilisation ')

               if conditions2:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les deux variables sont quantitatives ou ordinales.<br>
                           • Les données n'ont pas besoin d'être normalement distribuées.<br>
                           • La relation entre les variables peut être monotone (linéaire ou non linéaire).</p>""", unsafe_allow_html=True)                       
               
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphiques</strong><br>""", unsafe_allow_html=True)
               fig, ax = plt.subplots(2, 1, figsize=(15, 8))
               sns.scatterplot(x='Anciennete', y='DepreciationEUR', data=df, ax=ax[0], s=10)
               sns.regplot(x='Anciennete', y='DepreciationEUR', data=df, ax=ax[0], scatter=False, line_kws={'color':'k'})
               ax[0].set_ylim(-10000, 300)
               ax[0].set_xlim(None) 
               sns.scatterplot(x='Anciennete', y='Depreciation%', data=df, ax=ax[1], s=10)
               sns.regplot(x='Anciennete', y='Depreciation%', data=df, ax=ax[1], scatter=False, line_kws={'color':'k'})
               ax[1].set_ylim(-0.9, 0.15)
               ax[1].set_xlim(None)  
               st.pyplot(fig)

               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)

               interpretation2 = st.checkbox('Interprétation des résultats ')

               if interpretation2:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Interprétation des résultats</strong><br>
                           • Si le coefficient est proche de 1 (un seuil à 70% pour Kendall et 60% pour Spearman), il y a une corrélation positive entre les variables.<br>
                           • Si le coefficient est proche de -1 (un seuil à 70% pour Kendall et 60% pour Spearman), il y a une corrélation négative entre les variables.<br>
                           • Si le coefficient est proche de 0, il n'y a pas de corrélation monotone entre les variables.</p>""", unsafe_allow_html=True)
                    
               coeff_spearmanEUR, p_value_spearmanEUR = spearmanr(df['DepreciationEUR'], df['Anciennete'])
               coeff_spearmanREL,p_value_spearmanREL=spearmanr(df['Depreciation%'],df['Anciennete'])
               coeff_kendallEUR,p_value_kendallEUR=kendalltau(df['DepreciationEUR'],df['Anciennete'])
               coeff_kendallREL,p_value_kendallREL=kendalltau(df['Depreciation%'],df['Anciennete'])
               df_stats = pd.DataFrame({'Variable Cible':['DepreciationEUR','Depreciation%','DepreciationEUR','Depreciation%'],'Statistique du test de Spearman & Kendall': [coeff_spearmanEUR,coeff_spearmanREL,coeff_kendallEUR,coeff_kendallREL], 'P-value': [p_value_spearmanEUR,p_value_spearmanREL,p_value_kendallEUR,p_value_kendallREL]})
               df_stats = df_stats.astype(str)
               st.dataframe(df_stats, height=200)
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Spearman</strong><br> La P-value est inférieure à 0.05 dans les deux cas : <strong>il y a une corrélation entre les deux variables</strong>.<br>
                               La statistique du test est en-dessous de 0 dans les deux cas : <strong>il y a une corrélation négative entre les deux variables</strong>.
                               La corrélation est plus prononcée avec <strong>la dépréciation relative</strong>.<br>""", unsafe_allow_html=True)
               st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Kendall</strong><br> La P-value est inférieure à 0.05 dans les deux cas : <strong>il y a une corrélation entre les deux variables</strong>.<br>
                               La statistique du test est en-dessous de 0 dans les deux cas : <strong>il y a une corrélation négative entre les deux variables</strong>.
                               La corrélation est plus prononcée avec <strong>la dépréciation relative</strong>.<br>""", unsafe_allow_html=True)
   
    if option=='Variables Catégorielles/Variables Cibles':
                var_cat=['Categorie','Marque','EtatVeloLabel','FlagElectrique']
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les variables catégorielles sont : Categorie, Marque, EtatVeloLabel, FlagElectrique.</p>""", unsafe_allow_html=True) 
                st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Kruskal-Wallis ou Mann-Withney (test d\'association)</h3>', unsafe_allow_html=True)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                • H0 : Les distributions de tous les groupes sont égales.<br>
                • H1 : Au moins une des distributions des groupes est différente des autres.</p>""", unsafe_allow_html=True)

                conditions3 = st.checkbox('Conditions d\'utilisation  ')

                if conditions3:
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les observations sont indépendantes.<br>
                           • Les données peuvent être ordonnées (au moins ordinale).<br>
                           • Plus de deux modalités sur une variable pour Kruskal-Wallis et deux modalités pour Mann-Withney.</p>""", unsafe_allow_html=True) 
                
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)
                
                @st.cache_data
                def stat(var_cat):
                  stat=[]
                  pval=[]
                  for i in var_cat:
                     if i=='FlagElectrique':
                          etats = df[i].unique()
                          data1 = [df[df[i] == etat]['DepreciationEUR'] for etat in etats]
                          statistic1, pvalue1 = mannwhitneyu(*data1)
                          data2 = [df[df[i] == etat]['Depreciation%'] for etat in etats]
                          statistic2, pvalue2 = mannwhitneyu(*data2)
                          stat.append(statistic1)
                          stat.append(statistic2)
                          pval.append(pvalue1)
                          pval.append(pvalue2)

                     else:
                          etats = df[i].unique()
                          data1 = [df[df[i] == etat]['DepreciationEUR'] for etat in etats]
                          statistic1, pvalue1 = kruskal(*data1)
                          data2 = [df[df[i] == etat]['Depreciation%'] for etat in etats]
                          statistic2, pvalue2 = kruskal(*data2) 
                          stat.append(statistic1)
                          stat.append(statistic2)
                          pval.append(pvalue1)
                          pval.append(pvalue2)                       
                  df_stats = pd.DataFrame({'Variable':['FlagElectrique','FlagElectrique','EtatVeloLabel','EtatVeloLabel','Categorie','Categorie','Marque','Marque'],
                                              'Variable Cible':['DepreciationEUR','Depreciation%','DepreciationEUR','Depreciation%','DepreciationEUR','Depreciation%','DepreciationEUR','Depreciation%'],
                                              'Test':['Mann-Whitney','Mann-Whitney','Kruskal-wallis','Kruskal-wallis','Kruskal-wallis','Kruskal-wallis','Kruskal-wallis','Kruskal-wallis'],
                                              'Statistique du test': stat, 'P-value': pval})
                  return df_stats


                df_stats=stat(['FlagElectrique','EtatVeloLabel','Categorie','Marque'])
                df_stats = df_stats.astype(str)
                st.dataframe(df_stats)
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La P-value est inférieure à 0.05 pour toutes les variables : au moins une des distributions de chaque variable est différente des autres.<br>
                Par contre, ces tests n'identifient pas où cette dominance se produit ni pour combien de paires de groupes la dominance s'obtient.<br>""", unsafe_allow_html=True)                

    if option=='Variables Catégorielles/Variables Catégorielles':
                
                import numpy as np
                @st.cache_data
                def all_result(cat_var,df_final):
                     df_chi2 = pd.DataFrame(index=cat_var, columns=cat_var)
                     df_cramer_v = pd.DataFrame(index=cat_var, columns=cat_var)
                     for i, column1 in enumerate(cat_var):
                               for j, column2 in enumerate(cat_var):
                                 if i != j:
                                     ct = pd.crosstab(df_final[column1], df_final[column2])
                                     chi2, p, dof, expected = chi2_contingency(ct)
                                     cramer_v = np.sqrt(chi2 / (df_final.shape[0] * (np.min(ct.shape) - 1)))
                                     df_chi2.loc[column1, column2] = p
                                     df_cramer_v.loc[column1, column2] = cramer_v
                     return df_chi2.astype(str), df_cramer_v.astype(float)
                
                df_chi2, df_cramer_v=all_result(['FlagElectrique','EtatVeloLabel','Categorie','Marque'],df)
                #st.write(df['FlagElectrique'])
                var_cat=['Categorie','Marque','EtatVeloLabel','FlagElectrique']
                st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les variables catégorielles sont : Categorie, Marque, EtatVeloLabel, FlagElectrique.</p>""", unsafe_allow_html=True) 
                st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Test de Chi-carré (chi2) & Coefficient de Cramer-V (test d\'association)</h3>', unsafe_allow_html=True)
                radio = st.radio("",('Chi-carré', 'Cramer-V'))
                if radio=="Chi-carré":
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Hypothèses</strong><br>
                     • H0: Il n'y a pas d'association entre les deux variables qualitatives.<br>
                     • H1: Il y a une association entre les deux variables qualitatives.</p>""", unsafe_allow_html=True)
                     conditions4 = st.checkbox('Conditions d\'utilisation  ')
                     if conditions4:
                         st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les deux variables sont qualitatives (catégoriques).<br>
                           • Les effectifs attendus sont supérieurs à 5 pour chaque croissement de variable.</p>""", unsafe_allow_html=True) 
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)
                     st.dataframe(df_chi2,width=1000)
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La P-value est inférieure à 0.05 dans tous les cas : il y a donc une association à toutes les combinaisons.
                      Il faut maintenant calculer le coefficient de Cramer-V pour mesurer la force d'association.</p>""", unsafe_allow_html=True) 
                else:
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Interprétation</strong><br>
                     Le coefficient de Cramér-V mesure la force de l'association entre deux variables qualitatives. Il varie entre 0 (pas d'association) et 1 (association parfaite).
                     Plus la valeur de Cramér-V est proche de 1 (supérieur à 0.60), plus l'association entre les deux variables est forte.""", unsafe_allow_html=True)                     
                     conditions5 = st.checkbox('Conditions d\'utilisation  ')
                     if conditions5:
                         st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Conditions d'utilisation</strong><br>
                           • Les deux variables sont qualitatives (catégoriques).<br>
                           • Le test du chi-carré a déjà été effectué et a montré une association significative entre les deux variables.<br>""", unsafe_allow_html=True)                 
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong><br>""", unsafe_allow_html=True)
                     
                     fig, ax = plt.subplots(figsize=(4, 2))
                     heatmap = sns.heatmap(df_cramer_v, cmap='Reds', annot=True, fmt=".2f", linewidths=0.5, linecolor='black', ax=ax,
                      annot_kws={"size": 5}, cbar_kws={"shrink": 1})
                     heatmap.set_facecolor('lightgray')
                     for _, spine in heatmap.spines.items():
                        spine.set_visible(True)
                        spine.set_color('black')

                     heatmap.set_xticklabels(heatmap.get_xticklabels(), size=5)
                     heatmap.set_yticklabels(heatmap.get_yticklabels(), size=5)
                     st.pyplot(fig)
                     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La valeur de Cramer-V est inférieure à 0.6 dans chacun des cas : aucune association n'est considérée comme forte.
                      Dans notre dataset, Les plus fortes associations sont entre : Categorie et FlagElectrique (0.58) et Marque et FlagElectrique (0.46).</p>""", unsafe_allow_html=True) 
                      
