import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(['Choix variable cible','Modèles linéaires','Modèles ensemblistes', 'Récapitulatif'])

scores_ene = pd.read_pickle("assets/tables/scores_ene.pkl")
scores_ene_q10 = pd.read_pickle("assets/tables/scores_ene_q10.pkl")

scores_enr = pd.read_pickle("assets/tables/scores_enr.pkl")
scores_enr_q10 = pd.read_pickle("assets/tables/scores_enr_q10.pkl")

scores_skb_mf_q10 = pd.read_pickle("assets/tables/scores_skb_mf_q10.pkl")
scores_skb_mm_q10 = pd.read_pickle("assets/tables/scores_skb_mm_q10.pkl")

scores_svr = pd.read_pickle("assets/tables/scores_svr.pkl")
scores_svr_q10 = pd.read_pickle("assets/tables/scores_svr_q10.pkl")

scores_gbr = pd.read_pickle("assets/tables/scores_gbr.pkl")
scores_gbr_q10 = pd.read_pickle("assets/tables/scores_gbr_q10.pkl")

scores_skr = pd.read_pickle("assets/tables/scores_gbr.pkl")
scores_skr_q10 = pd.read_pickle("assets/tables/scores_gbr_q10.pkl")

quantiles = [0,1695.0,2499.0,3000.0,3624.0,4469.0,5109.0,5999.0,6999.0,8500.0,50000]

with tab1:
     st.markdown('<h3 style="font-family:optima;color:#2596be;">Problème</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable cible évidente devrait être le prix auquel ces vélos d’occasion sont mis en vente. 
     Pourtant la pratique des professionnels du vélo et plus généralement des acteurs professionnels du commerce des objets de seconde main est différente : ces différents acteurs 
     raisonnent généralement sur la <strong>dépréciation relative</strong> du vélo, et évaluent par expérience les différentes caractéristiques qui déterminent cette dépréciation relative.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons donc testé deux variables cibles concurrentes :
     <br>- la dépréciation en euros (i.e. prix d'occasion - prix d'origine)
     <br>- la dépréciation relative en % (i.e. prix d'occasion/prix d'origine - 1)</p>"""
     , unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Pour ce faire, nous avons entraîné un modèle linéaire de type <strong>ElasticNet</strong> pour 
     chacune des variables cibles, en optimisant les hyperparamètres <i>l1_ratio</i> et <i>alpha</i> par validation croisée en 5 blocs 
     sur les données d'entraînement et comparé les résultats obtenus. Plus de détails sont donnés dans la section suivante.</p>"""
     , unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultats</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Métriques en utilisant la dépréciation en euros :</p>"""
     , unsafe_allow_html=True)

     st.dataframe(scores_ene, width=300)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Métriques en utilisant la dépréciation relative en % :</p>"""
     , unsafe_allow_html=True)

     st.dataframe(scores_enr, width=300)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les scores sont nettement supérieurs pour la dépréciation en euros. Cependant, les autres métriques 
     ne sont pas à la même échelle et sont donc difficilement comparables. Pour ce faire, nous avons transformé les prédictions de chaque modèle à une échelle comparable à celle de 
     l’autre modèle, et nous avons étudié les différentes erreurs moyennes par quantile de prix d’origine des vélos.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;margin-bottom:0px;">Comparatif des erreurs moyennes :</p>"""
     , unsafe_allow_html=True)

     option = st.selectbox('',('DepreciationE', 'Depreciation%'),index=0)
     if option=='DepreciationE' or option=='Depreciation%':
          if option=='DepreciationE':
               fig,ax = plt.subplots(2, 1, figsize=(10,6))
               
               ax[0].plot(scores_ene_q10['RMSE(E)'],label='Modèle EUR')
               ax[0].plot(scores_enr_q10['RMSE(E)'],label='Modèle %')
               ax[0].set_ylim(0,1500)
               ax[0].set_title("RMSE en fonction du prix d'origine du vélo (sur les données de test)")
               ax[0].set_xticks(range(len(quantiles)-1))
               xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
               ax[0].set_xticklabels(xticklabels)
               ax[0].set_xlabel('PrixOrigine')
               ax[0].set_ylabel('Depreciation en euros')
               ax[0].legend()
               ax[1].plot(scores_ene_q10['MAE(E)'],label='Modèle EUR')
               ax[1].plot(scores_enr_q10['MAE(E)'],label='Modèle %')
               ax[1].set_ylim(0,1500)
               ax[1].set_title("MAE en fonction du prix d'origine du vélo (sur les données de test)")
               ax[1].set_xticks(range(len(quantiles)-1))
               ax[1].set_xticklabels(xticklabels)
               ax[1].set_xlabel('PrixOrigine')
               ax[1].set_ylabel('Depreciation en euros')
               ax[1].legend()
               
               plt.tight_layout()
               st.pyplot(fig)
          
          elif option=='Depreciation%':
               fig,ax = plt.subplots(2,1,figsize=(10,6))

               ax[0].plot(scores_ene_q10['RMSE(%)'],label='Modèle EUR')
               ax[0].plot(scores_enr_q10['RMSE(%)'],label='Modèle %')
               ax[0].set_ylim(0,0.7)
               ax[0].set_title("RMSE en fonction du prix d'origine du vélo (sur les données de test)")
               ax[0].set_xticks(range(len(quantiles)-1))
               xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
               ax[0].set_xticklabels(xticklabels)
               ax[0].set_xlabel('PrixOrigine')
               ax[0].set_ylabel('Depreciation relative en %')
               ax[0].legend()
               ax[1].plot(scores_ene_q10['MAE(%)'],label='Modèle EUR')
               ax[1].plot(scores_enr_q10['MAE(%)'],label='Modèle %')
               ax[1].set_ylim(0,0.7)
               ax[1].set_title("MAE en fonction du prix d'origine du vélo (sur les données de test)")
               ax[1].set_xticks(range(len(quantiles)-1))
               ax[1].set_xticklabels(xticklabels)
               ax[1].set_xlabel('PrixOrigine')
               ax[1].set_ylabel('Depreciation relative en %')
               ax[1].legend()

               plt.tight_layout()
               st.pyplot(fig)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les erreurs moyennes obtenues avec la dépréciation relative en % sont similaires à celles obtenues 
     avec la dépréciation en euros pour les quantiles de prix d’origine médians, mais elles sont nettement inférieures pour les quantiles extrêmes (prix faibles et prix élevés). 
     Les prédictions basées sur la dépréciation relative sont plus précises, que ce soit pour prédire une dépréciation en euros ou une dépréciation relative en %.</p>"""
     , unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#2596be;">Choix de la variable cible</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous privilégierons la <strong>dépréciation relative</strong> dans la suite du projet malgré des scores inférieurs.</p>"""
     , unsafe_allow_html=True)


with tab2:
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les modèles linéaires considérés dans notre étude :
     <br>- ElasticNet
     <br>- SelectKBest
     <br>- LinearSVR</p>"""
     , unsafe_allow_html=True)
     
     modeleLineaire = st.selectbox('',('ElasticNet', 'SelectKBest', 'LinearSVR'),index=0)

     if modeleLineaire=='ElasticNet':
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Introduction</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Au départ, nous avons cherché à utiliser un modèle de régression linéaire sur notre jeu de données. 
          Cependant, la quantité considérable de variables nous a amené à consider une régression avec régularisation comme <i>ElasticNet</i> pour obtenir une solution dans laquelle toutes 
          les variables corrélées pertinentes pour la prédiction sont sélectionnées et reçoivent un poids identique.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le modèle <i>ElasticNet</i> a été testé afin de limiter le risque de sur-apprentissage dû au grand nombre 
          de coefficients à estimer (128). Ce modèle est une régression linéaire dont la fonction de perte intègre une pénalité liée aux normes L1 et L2 de l’ensemble des coefficients. Nous 
          avons optimisé les hyperparamètres <i>l1_ratio</i> (poids de la pénalité L1 par rapport à celle L2) et <i>alpha</i> (poids global des pénalités) à l’aide d’une validation croisée 
          en 5 blocs sur les données d'entraînement.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les hyperparamètres optimaux obtenus sont :
          <br>- <i>l1_ratio</i> (0,1)
          <br>- <i>alpha</i> (0,001)</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultats</h3>', unsafe_allow_html=True)

          st.dataframe(scores_enr, width=300)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
          , unsafe_allow_html=True)
          
          fig,ax = plt.subplots(2,1,figsize=(10,8))
          
          ax[0].plot(scores_enr_q10['RMSE(E)'],label='RMSE')
          ax[0].plot(scores_enr_q10['MAE(E)'],label='MAE')
          ax[0].set_ylim(0,1500)
          ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[0].set_xticks(range(len(quantiles)-1))
          xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
          ax[0].set_xticklabels(xticklabels)
          ax[0].set_xlabel('PrixOrigine')
          ax[0].set_ylabel('Depreciation en euros')
          ax[0].legend()
          ax[1].plot(scores_enr_q10['RMSE(%)'],label='RMSE')
          ax[1].plot(scores_enr_q10['MAE(%)'],label='MAE')
          ax[1].set_ylim(0,0.3)
          ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[1].set_xticks(range(len(quantiles)-1))
          ax[1].set_xticklabels(xticklabels)
          ax[1].set_xlabel('PrixOrigine')
          ax[1].set_ylabel('Depreciation relative en %')
          ax[1].legend()
          
          plt.tight_layout()
          st.pyplot(fig)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Conclusions</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">En conclusion, les résultats sont corrects avec une MAE sous les 10%. Peut-on faire mieux ? Cette approche 
          a aussi le mérite d'avoir une sélection de variables réduites et une stabilité des résultats vérifiée. Les résultats ne montrent pas de signe marqué de sur-apprentissage malgré le nombre 
          de coefficients estimés. L'approche <i>ElasticNet</i> sera notre référence dans la suite de l'étude.</p>"""
          , unsafe_allow_html=True)

          
     
     if modeleLineaire=='SelectKBest':
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Introduction</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Au départ, nous avons cherché à utiliser un modèle de régression linéaire sur notre jeu de données. 
          Cependant, la quantité considérable de variables avec un nombre d'occurrences faible a donné des résultats aberrants dans certains cas. Nous avons alors examiner d'autres 
          approches pour améliorer la performance du modèle et la répétabilité des résultats.</p>"""
          , unsafe_allow_html=True)
          
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Pour sélectionner les variables les plus pertinentes pour notre modèle de régression linéaire, 
          nous avons utilisé la méthode <i>SelectKBest</i>. Cela nous permet de choisir les K meilleures variables en fonction de leur relation avec la variable cible. Nous avons 
          optimisé notre modèle avec deux métriques différentes : <i>f_regression</i> et <i>mutual_info_regression</i>. La première métrique mesure la relation linéaire entre chaque 
          variable indépendante et la variable cible, tandis que la seconde métrique capture les dépendances non linéaires.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Notre approche avec les deux métriques a été similaire. Nous avons d'abord cherché à identifier 
          l'intervalle optimal du nombre de variables qui produisait des résultats fiables et cohérents, en éliminant les variables entrainant les resultats aberrants. Nous avons ensuite déterminé le nombre 
          idéal de variables (K) qui offrait les meilleurs scores pour chacune des métriques. Cela nous a permis de sélectionner un ensemble pertinent de variables, améliorant 
          ainsi la stabilité des résultats obtenus.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultats</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Notre modèle a été optimisé avec deux métriques différentes :
          <br>- <i>f_regression</i>
          <br>- <i>mutual_info_regression</i></p>"""
          , unsafe_allow_html=True)

          option = st.selectbox('',('f_regression', 'mutual_info_regression'),index=0)
          if option=='f_regression' or option=='mutual_info_regression':
               if option=='f_regression':
                    
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Sélection des variables (37) :</p>"""
                    , unsafe_allow_html=True)
                    st.write("""'Anciennete', 'PrixOrigine', 'Etat_Bon État', 'Etat_Comme Neuf','Etat_Très Bon État', 'Etat_Usé', 'Marque_Argon 18', 'Marque_Bianchi','Marque_Bulls', 
                    'Marque_B’Twin', 'Marque_CBT Italia', 'Marque_Cube', 'Marque_Devinci', 'Marque_Gitane', 'Marque_Haibike', 'Marque_Heroin', 'Marque_Ibis', 'Marque_Lapierre', 
                    'Marque_Look', 'Marque_Matra','Marque_Megamo', 'Marque_Moustache', 'Marque_Orange', 'Marque_Orbea', 'Marque_Polygon', 'Marque_Specialized', 'Marque_Time', 
                    'Marque_Yeti', 'Categorie_Aero / CLM', 'Categorie_Cross-country', 'Categorie_Descente', 'Categorie_Gravel / CX', 'Categorie_Route', 'Categorie_Trail / Enduro', 
                    'Categorie_VTT', 'Elec_False', 'Elec_True'""")
                    
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
                    , unsafe_allow_html=True)
                    
                    fig,ax = plt.subplots(2,1,figsize=(10,8))
                    
                    ax[0].plot(scores_skb_mf_q10['RMSE(E)'],label='RMSE')
                    ax[0].plot(scores_skb_mf_q10['MAE(E)'],label='MAE')
                    ax[0].set_ylim(0,1500)
                    ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
                    ax[0].set_xticks(range(len(quantiles)-1))
                    xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
                    ax[0].set_xticklabels(xticklabels)
                    ax[0].set_xlabel('PrixOrigine')
                    ax[0].set_ylabel('Depreciation en euros')
                    ax[0].legend()
                    ax[1].plot(scores_skb_mf_q10['RMSE(%)'],label='RMSE')
                    ax[1].plot(scores_skb_mf_q10['MAE(%)'],label='MAE')
                    ax[1].set_ylim(0,0.3)
                    ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
                    ax[1].set_xticks(range(len(quantiles)-1))
                    ax[1].set_xticklabels(xticklabels)
                    ax[1].set_xlabel('PrixOrigine')
                    ax[1].set_ylabel('Depreciation relative en %')
                    ax[1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)
               
               elif option=='mutual_info_regression':
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Sélection des variables (31) :</p>"""
                    , unsafe_allow_html=True)
                    st.write("""'Anciennete', 'PrixOrigine', 'Etat_Bon État', 'Etat_Comme Neuf', 'Etat_Très Bon État', 'Etat_Usé', 'Marque_Batavus', 'Marque_Bianchi', 'Marque_Bulls', 
                    'Marque_Cannondale', 'Marque_Gazelle', 'Marque_Giant', 'Marque_Gitane', 'Marque_Kona', 'Marque_Moustache', 'Marque_Orbea', 'Marque_Polygon', 'Marque_Rossignol', 
                    'Marque_Specialized', 'Marque_Stevens', 'Marque_Sunn', 'Marque_Van Rysel', 'Marque_Wilier Triestina', 'Categorie_Aero / CLM', 'Categorie_Cross-country', 
                    'Categorie_Trail / Enduro', 'Categorie_Urbain', 'Categorie_VTC', 'Categorie_VTT', 'Elec_False', 'Elec_True'""")
                    
                    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
                    , unsafe_allow_html=True)

                    fig,ax = plt.subplots(2,1,figsize=(10,8))

                    ax[0].plot(scores_skb_mm_q10['RMSE(E)'],label='RMSE')
                    ax[0].plot(scores_skb_mm_q10['MAE(E)'],label='MAE')
                    ax[0].set_ylim(0,1500)
                    ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
                    ax[0].set_xticks(range(len(quantiles)-1))
                    xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
                    ax[0].set_xticklabels(xticklabels)
                    ax[0].set_xlabel('PrixOrigine')
                    ax[0].set_ylabel('Depreciation en euros')
                    ax[0].legend()
                    ax[1].plot(scores_skb_mm_q10['RMSE(%)'],label='RMSE')
                    ax[1].plot(scores_skb_mm_q10['MAE(%)'],label='MAE')
                    ax[1].set_ylim(0,0.3)
                    ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
                    ax[1].set_xticks(range(len(quantiles)-1))
                    ax[1].set_xticklabels(xticklabels)
                    ax[1].set_xlabel('PrixOrigine')
                    ax[1].set_ylabel('Depreciation relative en %')
                    ax[1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Conclusions</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">En conclusion, l'approche avec la métrique <i>f_regression</i> offre des résultats très similaires 
          voire meilleurs qu'avec la métrique <i>mutual_info_regression</i>. Cette approche a aussi le mérite d'avoir un temps d'exécution très court, une sélection de variables réduites 
          à moins de 40 variables et une stabilité des résultats vérifiée. Les résultats ne montrent pas de signe marqué de sur-apprentissage malgré le nombre de coefficients estimés.</p>"""
          , unsafe_allow_html=True)

     if modeleLineaire=='LinearSVR':
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Introduction</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les meilleurs scores obtenus avec les modèles linéaires (<i>ElasticNet</i> & <i>SelectKBest</i>) restant 
          assez faibles, nous avons voulu testé si en acceptant une certaine imprécision irréductible dans les prédictions il était possible d’améliorer le score global et de pouvoir formuler 
          une prédiction dans une marge d’erreur acceptable pour un plus grand nombre d’observations.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’approche SVM appliquée à la régression linéaire (<i>LinearSVR</i>) permet d'entraîner le modèle sur un 
          sous-ensemble des données correspondant aux observations pour lesquelles la précision est inférieure à un seuil fixé à l’avance (<i>Epsilon</i>). Les observations pour lesquelles 
          les prédictions sont dans une marge d’erreur acceptable sont ignorées dans l’estimation du modèle.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons choisi d’estimer le modèle avec une fonction de perte de type L1 afin de minimiser l’effet des 
          larges erreurs de prédiction (par rapport à la norme L2). Afin de trouver le meilleur modèle, nous avons fait varier l'hyperparamètre <i>Epsilon</i> sur une  plage de valeurs en lien 
          avec la MAE obtenue avec les autres modèles, soit de 3% à 13%, et pour chaque valeur d'<i>Epsilon</i> nous avons optimisé l’hyperparamètre de pénalité <i>C</i> par une grille de recherche 
          sur les données d'entraînement. Nous cherchions à la fois à minimiser la MAE et à maximiser la proportion d’observations dont l’erreur de prédiction était 
          inférieure en valeur absolue à <i>Epsilon</i>.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les hyperparamètres optimaux obtenus sont :
          <br>- <i>C</i> (3)
          <br>- <i>Epsilon</i> (5.0%)</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultats</h3>', unsafe_allow_html=True)

          st.dataframe(scores_svr, width=300)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Autre métrique (% within Epsilon) :
          <br>- 38.5% (Train)
          <br>- 36.4% (Test)</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
          , unsafe_allow_html=True)

          fig,ax = plt.subplots(2,1,figsize=(10,8))

          ax[0].plot(scores_svr_q10['RMSE(E)'],label='RMSE')
          ax[0].plot(scores_svr_q10['MAE(E)'],label='MAE')
          ax[0].set_ylim(0,1500)
          ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[0].set_xticks(range(len(quantiles)-1))
          xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
          ax[0].set_xticklabels(xticklabels)
          ax[0].set_xlabel('PrixOrigine')
          ax[0].set_ylabel('Depreciation en euros')
          ax[0].legend()
          ax[1].plot(scores_svr_q10['RMSE(%)'],label='RMSE')
          ax[1].plot(scores_svr_q10['MAE(%)'],label='MAE')
          ax[1].set_ylim(0,0.3)
          ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[1].set_xticks(range(len(quantiles)-1))
          ax[1].set_xticklabels(xticklabels)
          ax[1].set_xlabel('PrixOrigine')
          ax[1].set_ylabel('Depreciation relative en %')
          ax[1].legend()
          
          plt.tight_layout()
          st.pyplot(fig)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Conclusions</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">En conclusion, les résultats obtenus sont comparables à ceux des autres modèles linéaires testés 
          et ne montrent pas de signe marqué de sur-apprentissage malgré le nombre de coefficients estimés. L'approche <i>LinearSVR</i> n'apporte pas de valeur particulière.</p>"""
          , unsafe_allow_html=True)

with tab3:
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’objectif de cette partie est de vérifier si en autorisant des interactions non linéaires et en 
     combinant les résultats de plusieurs modèles pour former les prédictions nous pouvons dépasser ces seuils de performance. Pour ce faire, nous avons testé deux modèles de type 
     “ensembliste”, de deux sous-catégories différentes : l’un de type Boosting, et l'autre de type Stacking.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les modèles ensemblistes considérés dans notre étude :
     <br>- GradientBoostingRegressor
     <br>- StackingRegressor</p>"""
     , unsafe_allow_html=True)

     modeleEnsemble = st.selectbox('',('GradientBoostingRegressor', 'StackingRegressor'),index=0)

     if modeleEnsemble=='GradientBoostingRegressor':
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Introduction</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le fonctionnement du GradientBoostingRegressor est de construire successivement une série 
          d’arbres de décision qui visent chacun à prédire les résidus de l’arbre précédent. La prédiction finale du modèle est alors la somme des prédictions de ces différents 
          arbres. On pourrait s’attendre à ce que les arbres successifs puissent capturer les relations locales entre certaines zones de variables explicatives et la dépréciation 
          relative qui seraient difficilement décrites par une relation linéaire unique.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons privilégié la métrique <i>neg_mean_absolute_error</i> pour estimer les différents 
          modèles candidats et les comparer en validation croisée. Cette dernière correspond à notre objectif : améliorer l’erreur de prédiction moyenne sans accorder trop de poids 
          aux larges erreurs de prédiction des outliers.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons filtré les données en fonction du nombre d'observations par classe et optimisé les 
          hyperparamètres à l'aide d'une validation croisée en 5 blocs sur les données d'entraînement.</p>"""
          , unsafe_allow_html=True)    

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La filtration optimale obtenue est :
          <br>- seuil de filtration (10)</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les hyperparamètres optimaux obtenus sont :
          <br>- taux d'apprentissage du modèle (0.1)
          <br>- nombre d'estimateurs successifs (200)
          <br>- profondeur maximum de chaque arbre (2)</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultats</h3>', unsafe_allow_html=True)

          st.dataframe(scores_gbr, width=300)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
          , unsafe_allow_html=True)

          fig,ax = plt.subplots(2,1,figsize=(10,8))

          ax[0].plot(scores_gbr_q10['RMSE(E)'],label='RMSE')
          ax[0].plot(scores_gbr_q10['MAE(E)'],label='MAE')
          ax[0].set_ylim(0,1500)
          ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[0].set_xticks(range(len(quantiles)-1))
          xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
          ax[0].set_xticklabels(xticklabels)
          ax[0].set_xlabel('PrixOrigine')
          ax[0].set_ylabel('Depreciation en euros')
          ax[0].legend()
          ax[1].plot(scores_gbr_q10['RMSE(%)'],label='RMSE')
          ax[1].plot(scores_gbr_q10['MAE(%)'],label='MAE')
          ax[1].set_ylim(0,0.3)
          ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[1].set_xticks(range(len(quantiles)-1))
          ax[1].set_xticklabels(xticklabels)
          ax[1].set_xlabel('PrixOrigine')
          ax[1].set_ylabel('Depreciation relative en %')
          ax[1].legend()

          plt.tight_layout()
          st.pyplot(fig)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Conclusions</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">
          En conclusion, les résultats obtenus sont comparables à ceux des autres modèles linéaires, et cela malgré une plus grande complexité, la présence d’un sur-apprentissage 
          et une moindre interprétabilité. L'approche <i>GradientBoosingRegressor</i> n'apporte pas de valeur particulière.</p>"""
          , unsafe_allow_html=True)

     if modeleEnsemble=='StackingRegressor':
          st.markdown('<h3 style="font-family:optima;color:#2596be;">Introduction</h3>', unsafe_allow_html=True)
          
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons voulu testé une méthode radicalement différente d’agréger différents modèles pour 
          former la prédiction finale par l’utilisation de <i>StackingRegressor</i>. Plusieurs estimateurs de base forment chacun et de façon indépendante une prédiction, et ces prédictions 
          de base sont ensuite agrégées par un autre algorithme (de type régression) pour former la prédiction finale du modèle.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Modélisation</h3>', unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons choisi d’utiliser trois estimateurs de base : deux estimateurs linéaires avec régularisation, 
          <i>RidgeCV</i> et <i>LassoCV</i>, et un estimateur d’interpolation locale <i>KNeighborsRegressor</i>. Ces trois estimateurs sont ensuite agrégés avec <i>GradientBoostingRegressor</i> 
          combinant différents arbres de décision.</p>"""
          , unsafe_allow_html=True)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"> Aucun des modèles n'a été optimisé. Le seul objectif était de tester des combinaisons avec cette méthode. 
          En effet, les performances obtenues étaient assez similaires aux modéles linéaires, avec toutefois une marque nette d'un certain sur-apprentissage.</p>"""
          , unsafe_allow_html=True)

          st.markdown('<h3 style="font-family:optima;color:#2596be;">Résultat</h3>', unsafe_allow_html=True)

          st.dataframe(scores_skr, width=300)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes :</p>"""
          , unsafe_allow_html=True)

          fig,ax = plt.subplots(2,1,figsize=(10,8))

          ax[0].plot(scores_skr_q10['RMSE(E)'],label='RMSE')
          ax[0].plot(scores_skr_q10['MAE(E)'],label='MAE')
          ax[0].set_ylim(0,1500)
          ax[0].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[0].set_xticks(range(len(quantiles)-1))
          xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
          ax[0].set_xticklabels(xticklabels)
          ax[0].set_xlabel('PrixOrigine')
          ax[0].set_ylabel('Depreciation en euros')
          ax[0].legend()
          ax[1].plot(scores_skr_q10['RMSE(%)'],label='RMSE')
          ax[1].plot(scores_skr_q10['MAE(%)'],label='MAE')
          ax[1].set_ylim(0,0.3)
          ax[1].set_title("RMSE/MAE en fonction du prix d'origine du vélo (sur les données de test)")
          ax[1].set_xticks(range(len(quantiles)-1))
          ax[1].set_xticklabels(xticklabels)
          ax[1].set_xlabel('PrixOrigine')
          ax[1].set_ylabel('Depreciation relative en %')
          ax[1].legend()

          plt.tight_layout()
          st.pyplot(fig)

          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">
          En conclusion, les résultats obtenus sont comparables à ceux des autres modèles linéaires, et cela malgré une plus grande complexité, la présence d’un sur-apprentissage 
          et une moindre interprétabilité. L'approche <i>StackingRegressor</i> n'apporte pas de valeur particulière.</p>"""
          , unsafe_allow_html=True)

with tab4:
     st.markdown('<h3 style="font-family:optima;color:#2596be;">1. Choix de la variable cible</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La variable cible évidente devrait être le <b>Prix</b> du vélo.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Mais nous avons testé deux variables cibles concurrentes :
     <br>- la dépréciation en euros (i.e. prix d'occasion - prix d'origine)
     <br>- la dépréciation relative en % (i.e. prix d'occasion/prix d'origine - 1)</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Modélisation avec <i>ElasticNet</i></p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les scores sont nettement supérieurs pour la dépréciation en euros :
     <br>- 75,4% vs. 61,2% (Train)
     <br>- 76,8% vs. 59,3% (Test)</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;margin-bottom:0px;">Comparatif des erreurs moyennes sur les données de test :</p>"""
     , unsafe_allow_html=True)

     fig,ax = plt.subplots(1, 2, figsize=(10,4))
               
     ax[0].plot(scores_ene_q10['RMSE(E)'],label='Modèle EUR')
     ax[0].plot(scores_enr_q10['RMSE(E)'],label='Modèle %')
     ax[0].set_ylim(0,1500)
     ax[0].set_title("RMSE en fonction du prix d'origine du vélo")
     ax[0].set_xticks(range(len(quantiles)-1))
     xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
     ax[0].set_xticklabels(xticklabels, rotation=45)
     ax[0].set_xlabel('PrixOrigine')
     ax[0].set_ylabel('Depreciation en euros')
     ax[0].legend()

     ax[1].plot(scores_ene_q10['MAE(E)'],label='Modèle EUR')
     ax[1].plot(scores_enr_q10['MAE(E)'],label='Modèle %')
     ax[1].set_ylim(0,1500)
     ax[1].set_title("MAE en fonction du prix d'origine du vélo")
     ax[1].set_xticks(range(len(quantiles)-1))
     ax[1].set_xticklabels(xticklabels, rotation=45)
     ax[1].set_xlabel('PrixOrigine')
     ax[1].set_ylabel('Depreciation en euros')
     ax[1].legend()
     
     plt.tight_layout()
     st.pyplot(fig)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><u>Observations :</u>
     <br>- les erreurs moyennes obtenues sont similaires pour les quantiles de prix d’origine médians,
     <br>- mais ces erreurs sont nettement inférieures pour les quantiles extrêmes (prix faibles et prix élevés).</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les prédictions basées sur la dépréciation relative sont plus précises.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Variable cible choisie : <b><u>dépréciation relative en %</u></b></p>"""
     , unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#2596be;">2. Modèles linéaires</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les modèles linéaires considérés dans notre étude :
     <br>- ElasticNet
     <br>- SelectKBest
     <br>- LinearSVR</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><u>Objectifs :</u>
     <br>- limiter le risque de sur-apprentissage dû au nombre de coefficients à estimer (128)
     <br>- améliorer la performance du modèle (ElasticNet : MAE de l'ordre de 8,5%)
     <br>- stabilité des résultats (traitement des scores aberrants)</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><u>Problèmes rencontrés :</u> <b>scores aberrants</b> sur certains modèles
     <br>- identification des variables (nombre d'occurrences faible)
     <br>- traitement des données sous la forme d'un filtre</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes sur les données de test :</p>"""
     , unsafe_allow_html=True)

     fig,ax = plt.subplots(1,2,figsize=(10,4))
          
     ax[0].plot(scores_enr_q10['RMSE(%)'],label='ElasticNet')
     ax[0].plot(scores_skb_mf_q10['RMSE(%)'],label='SelectKBest')
     ax[0].plot(scores_svr_q10['RMSE(%)'],label='LinearSVR')
     ax[0].set_ylim(0,0.3)
     ax[0].set_title("RMSE en fonction du prix d'origine du vélo")
     ax[0].set_xticks(range(len(quantiles)-1))
     xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
     ax[0].set_xticklabels(xticklabels, rotation=45)
     ax[0].set_xlabel('PrixOrigine')
     ax[0].set_ylabel('Depreciation relative en %')
     ax[0].legend()
     ax[1].plot(scores_enr_q10['MAE(%)'],label='ElasticNet')
     ax[1].plot(scores_skb_mf_q10['MAE(%)'],label='SelectKBest')
     ax[1].plot(scores_svr_q10['MAE(%)'],label='LinearSVR')
     ax[1].set_ylim(0,0.3)
     ax[1].set_title("MAE en fonction du prix d'origine du vélo")
     ax[1].set_xticks(range(len(quantiles)-1))
     ax[1].set_xticklabels(xticklabels, rotation=45)
     ax[1].set_xlabel('PrixOrigine')
     ax[1].set_ylabel('Depreciation relative en %')
     ax[1].legend()
     
     plt.tight_layout()
     st.pyplot(fig)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><b><u>Conclusions :</u></b>
     <br>- résultats similaires entre les différents modèles
     <br>- pas de signe de sur-apprentissage</p>"""
     , unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#2596be;">2. Modèles ensemblistes</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><u>Avantages :</u>
     <br>- autoriser des interactions non-linéaires
     <br>- combiner les résultats de différents modèles</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les modèles ensemblistes considérés dans notre étude :
     <br>- GradientBoostingRegressor
     <br>- StackingRegressor</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><u>Objectifs :</u>
     <br>- limiter le risque de sur-apprentissage dû au nombre de coefficients à estimer (128)
     <br>- améliorer la performance du modèle (ElasticNet : MAE de l'ordre de 8,5%)
     <br>- stabilité des résultats (traitement des scores aberrants)</p>"""
     , unsafe_allow_html=True)
     
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Comparatif des erreurs moyennes sur les données de test :</p>"""
     , unsafe_allow_html=True)

     fig,ax = plt.subplots(1,2,figsize=(10,4))
     
     ax[0].plot(scores_enr_q10['RMSE(%)'],label='ElasticNet')
     ax[0].plot(scores_gbr_q10['RMSE(%)'],label='GradientBoostingR')
     ax[0].plot(scores_skr_q10['RMSE(%)'],label='StackingR')
     ax[0].set_ylim(0,0.3)
     ax[0].set_title("RMSE en fonction du prix d'origine du vélo")
     ax[0].set_xticks(range(len(quantiles)-1))
     xticklabels = [format(label, ',.0f') for label in quantiles[1:]]
     ax[0].set_xticklabels(xticklabels, rotation=45)
     ax[0].set_xlabel('PrixOrigine')
     ax[0].set_ylabel('Depreciation relative en %')
     ax[0].legend()
     ax[1].plot(scores_enr_q10['MAE(%)'],label='ElasticNet')
     ax[1].plot(scores_gbr_q10['MAE(%)'],label='GradientBoostingR')
     ax[1].plot(scores_skr_q10['MAE(%)'],label='StackingR')
     ax[1].set_ylim(0,0.3)
     ax[1].set_title("MAE en fonction du prix d'origine du vélo")
     ax[1].set_xticks(range(len(quantiles)-1))
     ax[1].set_xticklabels(xticklabels, rotation=45)
     ax[1].set_xlabel('PrixOrigine')
     ax[1].set_ylabel('Depreciation relative en %')
     ax[1].legend()
     
     plt.tight_layout()
     st.pyplot(fig)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><b><u>Conclusions :</u></b>
     <br>- résultats similaires entre les différents modèles
     <br>- présence de sur-apprentissage
     <br>- moindre interprétabilité</p>"""
     , unsafe_allow_html=True)