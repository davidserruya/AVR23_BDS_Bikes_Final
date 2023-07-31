import streamlit as st

st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Récapitulatif resultats','Bilan','Pistes d\'amélioration'])

with tab1:
     st.write('Récapitulatif resultats')

     st.markdown(""" 
      | 	Modèle| Catégorie | Techniques d’optimisation | Description | R² Score | RMSE Score | MAE Score|  Conclusion |
      |  --------- | --------- |--------- | --------- |--------- | --------- |--------- | --------- |
      | SelectKBest | Linéaire | métriques: f_regression & mutual_info_regression | Régression linéaire avec utilisation de SelectKBest pour déterminer le nombre de variables à sélectionner| 58,7% | 11% | 8,6% | Score proche des autres modèles testés avec un ensemble de variables beaucoup plus faible. |
      | ElasticNet | Linéaire | Cross validation sur l1_ratio et alpha | Régression linéaire avec régularisation par les normes L1 et L2 des coefficients| 59,3% | 10,9% | 8,5%| Bon niveau de score par rapport à l’ensemble des modèles testés. |
      | LinearSVR | Linéaire | Grid search sur la valeur optimale de C pour différentes valeurs d’Epsilon | Approche à vecteur support avec pénalité L1 des résidus en dehors de la marge d’erreur acceptable | 59,4% | 10,9% | 8,6%| Score comparable aux autres modèles. L’approche SVM n’apporte pas de valeur particulière. |
      | GradientBoostingRegressor| Ensemble | Cross validation sur les paramètres    filtering__hurdle, learning_rate, n_estimators, max_depth | Agrégation d’une série d’arbres d’estimation par GradientBoosting | 59,7% | 10,8% |8,5% | Résultats identiques aux modèles linéaires. Signe de sur apprentissage. Pas de valeur ajoutée spécifique.  |
      | StackingRegressor| Ensemble | Essai non systématique de différents jeux d’hyperparamètres | Agrégation de prédictions d’estimateurs de base à l’aide d’un algorithme final de type Regressor | 59,3% | 10,8% |8,5% | Résultats identiques aux modèles linéaires. Signe de sur apprentissage. Pas de valeur ajoutée spécifique.  |
       """ ,unsafe_allow_html=True)

with tab2:
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Bilan Final</h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’objectif du projet était d'identifier s'il existait des variables qui expliquent le prix des vélos d’occasion, 
     de pouvoir formuler une prédiction des prix d’occasion à partir de ces variables 
     et d’examiner de façon quantitative si la pratique et la compréhension des acteurs professionnels étaient justifiées.<strong> Ces objectifs ont été atteints en grande partie.</strong></p>""", unsafe_allow_html=True)
     
     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Objectif 1: Identification des variables déterminant les prix de revente des vélos d\'occasion</h3>', unsafe_allow_html=True)
     with st.expander("Développer"):
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">L'analyse des données et l'estimation de différents modèles ont confirmé que l'ancienneté, l'état du vélo, le prix d'origine et le type de 
          propulsion (électrique ou musculaire) sont les principales variables déterminant le prix des vélos d'occasion. 
          Nous avons également constaté que la prédiction des prix était plus précise en utilisant la dépréciation relative plutôt que la dépréciation absolue, confortant en cela la pratique des différents acteurs du marché des vélos d’occasion </p>""", unsafe_allow_html=True)
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">L'ajout de certaines marques et catégories en tant que variables supplémentaires améliore légèrement la qualité des modèles, 
          mais il est important de limiter le risque de sur-apprentissage en utilisant des techniques de régularisation. </p>""", unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Objectif 2: Création de modèles transparents et interpretables</h3>', unsafe_allow_html=True)
     with st.expander("Développer"):
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">L'utilisation de variables croisées entre l'ancienneté et les variables binaires (état du vélo, type de propulsion, marques, catégories) 
          n'améliore pas la précision des prévisions, mais permet de construire un modèle complet et interprétable, en accord avec l'expérience des professionnels du marché des vélos d'occasion. 
          Cela permet aussi de quantifier de manière fiable les décotes initiales et les dépréciations annuelles utilisées par les professionnels pour estimer les différents types de vélos.</p>""", unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Objectif 3: Création d\'une côte dynamique</h3>', unsafe_allow_html=True)
     with st.expander("Développer"):
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;"> Nous avons créé un modèle de cotation des vélos d'occasion, qui permet 
          d'estimer n'importe quel vélo en relation avec les annonces en ligne en Janvier 2023. Ce modèle pourra être régulièrement actualisé en l'entrainant sur un jeu de données scrappées à partir des annonces récemment mises en ligne. Parmi l’ensemble des modèles utilisés (linéaires et non linéaires) aucun ne semble se détacher de façon significative en termes de performances. 
          Tous réalisent le même seuil maximum de performance d’une MAE sur la dépréciation relative aux environs de 8,50% et d’un coefficient de détermination de 60% (sur les données hors apprentissage). 
          Les modèles sans régularisation semblent en revanche tous sur déterminés quand les nombreuses variables Marques (et plus encore avec l’emploi des variables croisées) sont employées.</p>""", unsafe_allow_html=True)

     st.markdown('<h2 style="font-family:optima;color:#2596be;">Limites des nos travaux </h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ce niveau indépassable de performance dans les prévisions, qui reste très bas, est relativement insatisfaisant. 
     Il pourrait sans doute être amélioré par l’emploi de méthodes et techniques de Machine Learning que nous n’avons pas employé dans ce court projet (onglet <i>Pistes d'amélioration</i>), 
     mais doit aussi tenir en partie à la nature intrinsèque des données utilisées: </p>""", unsafe_allow_html=True)   
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">• Les données sont toutes <strong>déclaratives</strong> et donc sujettes à erreur, mauvaise information (onglet <i>Pistes d'amélioration</i>).
     <br> <br>• La variable cible est construite à partir des prix de vente demandés et <strong>non des prix des transactions réelles</strong> (inaccessibles). La création d’un registre national des transactions sur les vélos d’occasion (comme il en existe un pour les voitures d’occasion) permettrait certainement de resserrer les erreurs de prévision.
     <br> <br>• Le marché des vélos d’occasion est encore <strong>jeune, dispersé, et en manque d’informations partagées</strong> par tous les acteurs particuliers. A mesure que ce marché mature et que des références communes de prix se développent (à l’instar de ce rapide essai), la variabilité des prix annoncés et des prix de transaction ne pourra que décroitre.</p>""", unsafe_allow_html=True)    


with tab3:
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Pistes d\'amélioration</h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Cette étude a des limitations évidentes dont certaines tiennent au cadre et aux contraintes dans lesquelles elle s’inscrivait. 
     Les directions dans lesquelles les analyses pourraient être approfondies, avec un objectif réaliste d’améliorer les résultats sont multiples. 
     <br><strong>Nous en proposons deux en particulier:</strong> </p>""", unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">1. Introduire une dimension temporelle</h3>', unsafe_allow_html=True)
     with st.expander("Développer"):
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">     Les données étudiées proviennent d’un lot d’annonces scrappées en Janvier 2023, homogènes dans le temps (annonces mises en ligne entre fin Novembre 2022 et début Janvier 2023). 
          <br>Les professionnels s’accordent pourtant sur le fait que le marché des vélos d’occasion montre des caractéristiques saisonnières : plus grande activité au printemps, moindre en hiver avec un pic court vers la période de Noël.</p>""", unsafe_allow_html=True)
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">Le prix des vélos d’occasion a également fortement progressé (par rapport à celui des vélos neufs) 
          pendant la crise du Covid en raison des tensions sur la disponibilité des vélos neufs. Il serait par conséquent intéressant d’introduire une dimension temporelle à l’étude dans une prochaine étape.</p>""", unsafe_allow_html=True)
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">Cela nécessiterait de scrapper régulièrement les annonces pour produire d’autres jeux de données (tous les mois par exemple) et d’:<br>
          • Estimer le même modèle sur les jeux de données successifs pour étudier comment les paramètres de ce modèle évoluent dans le temps.
          <br>• Intégrer une approche de séries temporelles à l’analyse par variables homogènes dans le temps pour essayer de prévoir les éventuelles tendances structurelles et saisonnières du marché.</p>""", unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">2. Utiliser les données non structurées</h3>', unsafe_allow_html=True)
     with st.expander("Développer"):
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">La deuxième piste d’amélioration que nous proposons tient au type de données utilisées : 
          actuellement seule une partie des informations disponibles dans les annonces ont été exploitées pour construire le jeu de variables explicatives; les données structurées telles le millésime, la marque, l’état du vélo… 
          Les données non structurées ont été ignorées.</p>""", unsafe_allow_html=True)
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;font-size:16px;">Hors à la fois les photos du vélo (ou des composants du vélo) généralement attachées à l’annonce, ainsi que le texte libre de la rubrique “Description” 
          pourraient également être exploitées par le biais de méthodes de Deep Learning (NLP et Computer vision) pour:
          <br>• Filtrer les annonces pour améliorer la qualité des données (repérage des vélos ayant fait l’objet d’améliorations, exclusion des annonces frauduleuses, exclusion des annonces erronées…)
          <br>• Extraire des photos et du texte libre d’autres variables explicatives</p>""", unsafe_allow_html=True)          