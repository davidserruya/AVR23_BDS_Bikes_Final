import streamlit as st
import pandas as pd
import pickle

st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(['Polynomial features','Traitement de Marque','Traitement des valeurs extrêmes','Récapitulatif'])

with tab1:
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Modèle de base</h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le modèle linéaire le plus simple duquel nous sommes partis est composé de la variable cible Dépréciation relative et des variables explicatives suivantes: </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">• Ancienneté (variable numérique)
      <br>• <i>Prix d’origine</i> (variable numérique)
      <br>• <i>État du vélo</i>  (variable catégorielle non ordonnée à 4 modalités)
      <br>• <i>Flag électrique</i>  (variable catégorielle non ordonnée à 2 modalités)
    <br>• <i>Catégories</i>  (variable catégorielle non ordonnée à 16 modalités)
    <br>• <i>Marques</i>  (variable catégorielle non ordonnée à 104 modalités)</p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le modèle peut s’écrire sous la forme: </p>""", unsafe_allow_html=True)
     st.write("$Dépréciation(i) = C + \\alpha \\times Ancienneté(i)"
         " + \\beta \\times PrixOrigine(i)"
         " + \\delta \\times Electrique(i)"
         " + \\epsilon_1 \\times Etat_1(i) + \\epsilon_2 \\times Etat_2(i) + \\ldots"
         " + \\gamma_1 \\times Catégorie_1(i) + \\gamma_2 \\times Catégorie_2(i) + \\ldots"
         " + \\kappa_1 \\times Marque_1(i) + \\kappa_2 \\times Marque_2(i) + \\ldots"
         " + Reste(i)$")
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Après exclusion des classes pour lesquelles moins de 20 représentants sont présents dans le jeu de données Train, 54 paramètres restent à estimer dans ce modèle linéaire de base. </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Sous cette forme le paramètre C représente la dépréciation moyenne en sortie de magasin et Alpha la dépréciation annuelle moyenne des vélos. L’appartenance d’un vélo à une certaine classe (ex: Marque Bianchi, Catégorie VTT, Électrique…) affecte sa dépréciation en sortie de magasin (dépréciation initiale) mais ne modifie pas son rythme moyen de dépréciation dans le temps. Hors ceci est contraire à l’intuition et à l’approche heuristique des acteurs professionnels du vélo : ceux-ci considèrent qu’un vélo se déprécie dans le temps à un rythme spécifique selon qu’il est électrique ou non, qu’il appartient à telle marque ou à telle catégorie… </p>""", unsafe_allow_html=True)
     
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Modèle avec variables croisées</h2>', unsafe_allow_html=True)    
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Pour tester cette approche nous avons construit un second modèle linéaire en introduisant les variables catégorielles croisées avec (multipliées par) la variable Ancienneté. Le modèle est équivalent à une transformation par la classe PolynomialFeatures de Scikit-Learn ou l’ordre est fixé à 2 et seuls les termes croisés entre Ancienneté et les variables catégorielles sont retenus. </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ce second modèle linéaire peut s’écrire sous la forme : </p>""", unsafe_allow_html=True)
     st.write("$Dépréciation(i) = C + \\alpha \\times Ancienneté(i)"
         " + \\beta \\times PrixOrigine(i)"
         " + \\delta \\times Electrique(i)"
         " + \\epsilon_1 \\times Etat_1(i) + \\epsilon_2 \\times Etat_2(i) + \\ldots"
         " + \\gamma_1 \\times Catégorie_1(i) + \\gamma_2 \\times Catégorie_2(i) + \\ldots"
         " + \\kappa_1 \\times Marque_1(i) + \\kappa_2 \\times Marque_2(i) + \\ldots"
         " + \\delta_{Croisé} \\times Ancienneté(i) \\times Electrique(i)"
         " + \\epsilon_{Croisé_k} \\times Ancienneté(i) \\times Etat_k(i) + \\ldots"
         " + \\gamma_{Croisé_k} \\times Ancienneté(i) \\times Catégorie_k(i) + \\ldots"
         " + \\kappa_{Croisé_k} \\times Ancienneté(i) \\times Marque_k(i) + \\ldots"
         " + Reste(i)$")
     
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Après exclusion des classes pour lesquelles moins de 20 représentants sont présents dans le jeu de données Train, 106 paramètres restent à estimer dans ce modèle linéaire avec variables croisées. </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ces deux modèles théoriques sont estimés par l’intermédiaire d’un algorithme de Machine à Vecteur Support appliqué à la régression linéaire (LinearSVR), avec le choix d’un seuil des écarts tolérés de epsilon fixé à 5% et une recherche de l’hyperparamètre de pénalisation C optimal pour chaque modèle. Cet algorithme est entraîné sur le jeu de données dans lequel les classes ayant moins de 20 représentants dans le jeu Train sont exclues (cf. section “Traitement des Marques avec peu de représentants”). </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong></p>""", unsafe_allow_html=True)
    
     col1, col2, col3 = st.columns([1, 0.2, 1])  # Répartition des colonnes
     with col1:
        st.markdown('<p style="text-align: center; "font-family:Arial,sans-serif;font-size: 25px"><strong>Modèle linéaire de base</strong></p>', unsafe_allow_html=True)
        st.image("assets/images/modele_base.png", use_column_width=True)
     col2.write("")
     with col3:
        st.markdown('<p style="text-align: center; "font-family:Arial,sans-serif;font-size: 25px"><strong>Modèle avec variables croisées</strong></p>', unsafe_allow_html=True)
        st.image("assets/images/modele_tx.png", use_column_width=True)
     
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les résultats obtenus sur le jeu de données Test avec les variables croisées sont légèrement supérieurs à ceux obtenus avec le modèle de base. Cette différence n’est cependant pas significative et ne permet pas de dire que l’utilisation des variables croisées augmente le caractère explicatif du modèle linéaire ni que les prédictions sont plus précises. </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’emploi des variables croisées est cependant intéressant en ce qu’il permet de valider et de quantifier les intuitions des acteurs professionnels du vélo qui anticipent des dépréciations annuelles des vélos différenciées selon leur marque, catégorie, … <br>Dans la section “Interprétation”, nous étudierons en détail les coefficients linéaires des variables croisées et nous constaterons qu’ils sont interprétables, intuitifs et qu’ils confortent la compréhension expérimentale des professionnels. </p>""", unsafe_allow_html=True)

with tab2:
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Lorsque nous utilisons un modèle linéaire avc l'ensemble des variables, nous obtenons des scores de métriques aberrants et extremes. 
     Ces résultats sont dûs au fait que beaucoup de variables binairisés originaires de la variable Marque et Catégorie ne posèdent peu ou aucune occurence dasn le jeu d'entraînement. Pour pallier à cet obstacle, nous allons tenter différentes techniques autour de ces variables.</p>""", unsafe_allow_html=True)

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">1. Suppresion des variables très peu représentées</h3>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous allons utiliser un modèle simple de régression linéaire dans lequel nous allons enlever les variables très peu représentées.
       Il s'agit des variables pour lesquelles il y a moins de 3 occurences dans le jeu d'entraînement.<strong> 24 variables ont été suppprimées</strong>. Nous avons ensuite entraîné ce nouveau jeu de données sur un modèle de régression linéaire simple</p>""", unsafe_allow_html=True)     
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Résultats</strong></p>""", unsafe_allow_html=True)
     st.image('assets/images/score_marque_sup.png')

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">2. Regroupement des variables marques par clustering</h3>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Cette approche basée sur le clustering vise à regrouper les marques similaires ensemble et ainsi réduire le nombre de variables liées à la binarisation de Marque. Nous avons utilisé plusieurs techniques pour obtenir le nombre de clusters optimal mais aussi différentes méthodes de clustering pour regrouper les variables.</p>""", unsafe_allow_html=True) 
     option = st.selectbox('',('K-means','Agglomerative Clustering'),index=0) 
     if option=='K-means':
         radio = st.radio("**Nombre de clusters optimal**",('Méthode du coude', 'Méthode de silhouette'))
         if radio=='Méthode du coude':
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/methode_coude.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La courbe obtenue décroit fortement et change de trajectoire après k = 4. 
             Ainsi, <strong>le nombre de clusters optimal est 4</strong>.
             <br>La répartition des clusters avec la méthode des coudes est présentée ci-dessous :</p>""", unsafe_allow_html=True) 
             st.image("assets/images/cluster_k_1.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Chaque marque a été ensuite remplacé par le cluster le plus représenté :</p>""", unsafe_allow_html=True) 
             st.image("assets/images/cluster_k_2.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons ensuite entrainé et testé ce nouveau jeu de données sur un modèle de régression linéaire.<br><br><strong>Résultats</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/score_k_1.png") 
         else: 
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/methode_sil.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Le nombre de clusters le plus satisfaisant est 2</strong> car pour n_cluters = 2 le coefficient de silhouette de partitionnement est le plus élevé</strong>.
             <br><br>La répartition des clusters avec la méthode des coudes est présentée ci-dessous :</p>""", unsafe_allow_html=True) 
             st.image("assets/images/cluster_k_3.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Chaque marque a été ensuite remplacé par le cluster le plus représenté :</p>""", unsafe_allow_html=True) 
             st.image("assets/images/cluster_k_4.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons ensuite entrainé et testé ce nouveau jeu de données sur un modèle de régression linéaire.<br><br><strong>Résultats</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/score_k_3.png") 
     else:
         radio2 = st.radio("**Nombre de clusters optimal**",['Dendogramme'])
         if radio2=='Dendogramme':
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/dendogramme.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le dendrogramme suggère <strong>un découpage en 4 groupes</strong> car il s'agit du plus grand saut entre deux clusters.
             <br><br>La répartition des clusters avec la méthode du dendogramme est présentée ci-dessous :</p>""", unsafe_allow_html=True)
             st.image("assets/images/cluster_k_5.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Chaque marque a été ensuite remplacé par le cluster le plus représenté :</p>""", unsafe_allow_html=True) 
             st.image("assets/images/cluster_k_6.png")
             st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons ensuite entrainé et testé ce nouveau jeu de données sur un modèle de régression linéaire.<br><br><strong>Résultats</strong></p>""", unsafe_allow_html=True) 
             st.image("assets/images/score_k_4.png") 

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">3. Frequency Encoding variable Marque</h3>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le Frequency Encoding est une technique de transformation de variables catégorielles en utilisant l'information sur la fréquence d'apparition de chaque catégorie dans un ensemble de données. 
     <strong>Le Frequency Encoding attribue à chaque catégorie une valeur correspondant à sa fréquence relative dans le jeu de données</strong>.</p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Formule</strong></p>""", unsafe_allow_html=True)
     st.latex(r'''frequency=df.groupby('Marque').size()/len(df)''')
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><br>Nous avons ensuite entrainé et testé ce nouveau jeu de données sur un modèle de régression linéaire.<br><br><strong>Résultats</strong></p>""", unsafe_allow_html=True) 
     st.image("assets/images/score_k_5.png") 

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">4. Mean Encoding variable Marque</h3>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L'idée principale du mean encoding est de <strong>remplacer chaque catégorie d'une variable catégorielle par la valeur moyenne de la variable cible pour cette catégorie</strong>.</p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Formule</strong></p>""", unsafe_allow_html=True)
     st.latex(r'''mean=df_.groupby('Marque')['Depreciation\%'].mean()''')
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><br>Nous avons ensuite entrainé et testé ce nouveau jeu de données sur un modèle de régression linéaire.<br><br><strong>Résultats</strong></p>""", unsafe_allow_html=True) 
     st.image("assets/images/score_k_6.png") 

     st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Conclusion</h3>', unsafe_allow_html=True)
         
     
     scores_cluster = pd.read_pickle("assets/tables/scores_cluster.pkl")

     st.dataframe(scores_cluster)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ces approches sont efficaces pour résoudre le problème de résultats incohérents observés dans certains modèles linéaires lorsqu'ils sont appliqués à l'ensemble des variables. 
     Elles permettent de réduire les effets indésirables liés à des variables moins pertinentes ou potentiellement redondantes. 
     Cependant, nos travaux de recherche ont démontré que ces approches n'ont pas produit de scores supérieurs à ceux obtenus avec d'autres modèles que nous avons testés.</p>""", unsafe_allow_html=True) 

with tab3:
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La filtration des valeurs extrêmes est souvent utilisée dans le contexte de l'analyse des résidus dans les modèles de régression. Les résidus sont la différence entre les valeurs prédites par le modèle et les valeurs réelles. Cette technique peut être utilisée dans le but de détecter les valeurs aberrantes ou les observations atypiques dans les résidus.  </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ce procédé permet également de stabiliser le modèle. En effet, les résidus anormaux ou influents peuvent indiquer des erreurs de modélisation ou des observations inhabituelles qui peuvent affecter la stabilité et la performance du modèle. En filtrant ces résidus, on peut améliorer la stabilité du modèle et réduire l'impact des valeurs aberrantes sur les résultats. Enfin, en éliminant les résidus atypiques, on peut améliorer la précision et la fiabilité des prédictions du modèle. Les valeurs aberrantes peuvent introduire du bruit ou biaiser les prédictions, et en les filtrant, on peut obtenir des résultats plus cohérents et précis.  </p>""", unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:optima;color:#2596be;">Filtration des résidus extrêmes en valeur absolue</h2>', unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’objectif est d'éliminer les valeurs extrêmes en utilisant les résidus d’un modèle de régression linéaire. Plus précisément, l’identification des outliers se fait en vérifiant si la valeur absolue des résidus est inférieure à un certain seuil. Les valeurs inférieures à ce seuil sont donc considérées comme non-outliers. </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Le seuil est défini comme</strong>:</p>""", unsafe_allow_html=True)
    st.latex(r'''Coefficient * \text{Écart-type des résidus}''')
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">On utilise ensuite le modèle LinearSVR avec différents seuils de filtrage des outliers sur le jeu de données d'entraînement. Cette fonction nous permet de retourner les erreurs absolues moyennes et les pourcentages de points se situant dans une marge epsilon définie pour chaque seuil de filtrage. </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’objectif de cette recherche a été de trouver le seuil optimal pour minimiser la MAE.<br><br><strong>Graphique</strong> </p>""", unsafe_allow_html=True)
    st.image("assets/images/MAE_seuil_filtration.png", use_column_width=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Le choix du seuil d’exclusion des outliers sur le jeu d'entraînement à 1.25 semble optimal </strong> du point de vue de la MAE comme du point de vue de la proportion des observations dont les résidus sont sous le seuil de 10%. </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le résultat de la filtration des outliers (seuil à 1.25) donne les observations suivantes:  </p>""", unsafe_allow_html=True)
    st.image("assets/images/outliers_abs_graph.png", use_column_width=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">On ne remarque cependant aucune amélioration significative du score sur le jeu de données test :</p>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 0.2, 1])  # Répartition des colonnes
    with col1:
        st.markdown('<p style="text-align: center; "font-family:Arial,sans-serif;font-size: 25px"><strong>Résultats du modèle sans filtration</strong></p>', unsafe_allow_html=True)
        st.image("assets/images/res_abs_unfiltered.png", use_column_width=True)
    col2.write("")
    with col3:
        st.markdown('<p style="text-align: center; "font-family:Arial,sans-serif;font-size: 25px"><strong>Résultats du modèle avec filtration</strong></p>', unsafe_allow_html=True)
        st.image("assets/images/res_abs_filtered.png", use_column_width=True)

    st.markdown('<h2 style="font-family:optima;color:#2596be;">Filtration des résidus par leur IQR</h2>', unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">La technique de l'écart interquartile (IQR) consiste à mesurer la dispersion des données en utilisant la différence entre le quartile supérieur et le quartile inférieur. L'écart interquartile est une mesure de dispersion qui représente la différence entre le troisième quartile (75e percentile) et le premier quartile (25e percentile) d'une distribution de données. En multipliant cet écart interquartile par un coefficient K, on obtient alors un seuil acceptable. Ce seuil acceptable correspond à une marge de tolérance permettant d'identifier les valeurs résiduelles qui s'éloignent considérablement de la tendance centrale de la distribution.  </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Formule</strong></p>""", unsafe_allow_html=True)
    st.latex(r'''IQR = Q3 - Q1''')
    st.latex(r'''\text{Seuil inférieur} : Q1 - K * IQR''')
    st.latex(r'''\text{Seuil supérieur} : Q3 + K * IQR''')
    st.latex(r'''outliers = (residuals < \text{Seuil inférieur} ) | (residuals > \text{Seuil supérieur}) ''')
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><br>Pour notre part, nous avons utilisé cette méthode sur le jeu de données d'entraînement avec un modèle SVR.<strong> Le coefficient le plus performant pour définir le seul acceptable était de 1.5</strong>. Il s’agissait du seuil avec le meilleur score MAE sur le jeu d'entraînement. </p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"><strong>Graphique</strong> </p>""", unsafe_allow_html=True)
    st.image("assets/images/MAE_IQR_seuil.png", use_column_width=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Une   fois   le   seuil  défini à 1.5, nous avons supprimé les valeurs en dehors de cette plage sur le jeu Train. Le résultat de la filtration donne les observations suivantes:  </p>""", unsafe_allow_html=True)
    st.image("assets/images/outliers_iqr_graph.png", use_column_width=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons par la suite entraîné et testé ce nouveau modèle sur différentes métriques.<br><br><strong>Résultats</strong> </p>""", unsafe_allow_html=True)
    st.image("assets/images/res_iqr.png", use_column_width=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Cette technique également ne conduit pas à des résultats probants. Les scores sur le jeu d'entrainement sont améliorés, ce qui est attendu puisque celui-ci ne contient plus les résidus extrêmes. En revanche on ne constate aucune amélioration significative des résultats sur le jeu de données test. </p>""", unsafe_allow_html=True)

with tab4:
    st.markdown('<h2 style="font-family:optima;color:#2596be;">Récapitulatif</h2>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Objectifs</h3>', unsafe_allow_html=True)
    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">
                • Stabiliser les modèles<br>
                • Améliorer les résultats<br>
                • Valider et quantifier les intuitions des acteurs professionnels<br></p>""", unsafe_allow_html=True) 
    
    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Plan d\'action</h3>', unsafe_allow_html=True)
    st.image('assets/images/recap1.png')

    st.markdown('<h3 style="font-family:optima;color:#00c4cc;">Conclusions</h3>', unsafe_allow_html=True)

    st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">
                • <strong>Stabilisation des modèles</strong>: Nous obtenons des résultats cohérents et stables sur l'ensemble des modèles.<br>
                • <strong>Amélioration des résultats</strong>: Certaines approches améliorent nos scores mais pas de façon significative.<br>
                • <strong>Validation des intuitions des acteurs professionnels</strong>: L'approche Polynomial Features tend vers cette objectif.<br></p>""", unsafe_allow_html=True)    
    

    
