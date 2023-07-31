import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import date
import joblib
import pandas as pd
import numpy as np

st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1><br>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Modèle linéaire','Skater','Shap'])

# Fonction d'impression des graphes de courbes de dépréciation
def graph_depreciation_curve(df_h,curves):

    today = date.today()
    fract_time = float((today - date(today.year-1,9,15)).days / 365)
    millesime = [2023-i for i in range(0,10)]
    anciennete = np.array([0]+[y+fract_time for y in range(0,int(today.year)+1-2015)])
    dep_generic = df_h.iloc[0]["Intercept"]+df_h.iloc[0]["Etat_Très Bon État"]+df_h.iloc[0]["PrixOrigine_per_1k"]*1500/1000+(df_h.iloc[0]["Anciennete"]+df_h.iloc[0]["t_Etat_Très Bon État"])*anciennete
    
    fig,ax = plt.subplots(figsize=(13,9))
    for curve in curves:
        dep_curve = df_h.iloc[0]["Intercept"]+anciennete*df_h.iloc[0]["Anciennete"] + df_h.iloc[0][curves[curve]["Etat"]] +df_h.iloc[0]["PrixOrigine_per_1k"]*curves[curve]["PrixOrigine"]/1000+ anciennete*(df_h.iloc[0]["t_"+curves[curve]["Etat"]])
        etiquette = "Vélo - "+curves[curve]["Etat"][5:]
        if curves[curve]["Elec"]=="Elec_True":
            dep_curve+=df_h.iloc[0]["Elec_True"] + anciennete*(df_h.iloc[0]["t_Elec_True"])
            etiquette+=" - Electrique"
        if curves[curve]["Marque"]!="None":
            dep_curve+=df_h.iloc[0][curves[curve]["Marque"]] + anciennete*(df_h.iloc[0]["t_"+curves[curve]["Marque"]])
            etiquette+=" - Marque: "+curves[curve]["Marque"][7:]
        if curves[curve]["Categorie"]!="None":
            dep_curve+=df_h.iloc[0][curves[curve]["Categorie"]] + anciennete*(df_h.iloc[0]["t_"+curves[curve]["Categorie"]])
            etiquette+=" - Catégorie: "+curves[curve]["Categorie"][10:]
        etiquette += f" - Prix Orig: {curves[curve]['PrixOrigine']} EUR"           
        plt.plot(anciennete,dep_curve,label=etiquette)
    plt.legend()
    plt.xlabel("Millésime")
    plt.ylabel("Dépréciation relative")
    plt.title("Dépréciation relative en fonction du Millésime")
    ax.set_xticks(anciennete)
    ax.set_xticklabels(millesime)
    formatter = mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(base=0.05))
    ax.yaxis.grid(True, linestyle='--', which='both', color='lightgray', linewidth=0.7, alpha=0.7, zorder=0)
    return fig

# Chargement du dataset pour accéder à la liste des Marques, Catégories, Etat Vélo
df = pd.read_csv('assets/dataset/data_ML.csv')
liste_marques = sorted(list(df['Marque'].unique()))

with tab1:
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons choisi d’étudier les résultats du <strong>modèle LinearSVR</strong> de façon approfondie car il se prête facilement à un exercice d’interprétation et de comparaison avec la pratique des acteurs professionnels, que ses scores sont conformes à ceux des autres modèles (y compris les modèles plus complexes de type Ensemble) et que sa fonction de perte est la plus conforme à l’objectif du projet : maximiser le nombre de vélos pour lesquels la prédiction de prix est dans une marge d’erreur jugée acceptable aux vues de la nature des données. </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le modèle inclut <strong>les variables croisées</strong> (cf Méthodologie - Polynomial features) et est estimé par l’intermédiaire d’un algorithme de Machine à Vecteur Support appliqué à la régression linéaire (LinearSVR), avec le choix d’un seuil des écarts tolérés de Epsilon fixé à 5% et une recherche de l’hyperparamètre de pénalisation C optimal (C=3 au final). Cet algorithme est entraîné sur le jeu de données dans lequel les classes ayant moins de 20 représentants dans le jeu Train sont exclues (cf paragraphe “Traitement des Marques avec peu de représentants”). </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous rappelons <strong>les résultats</strong> de ce modèle : </p>""", unsafe_allow_html=True)    
     st.image("assets/images/modele_tx.png", use_column_width=True)
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Interprétation des coefficients</h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Afin de pouvoir interpréter les coefficients estimés en comparant directement leurs valeurs nous leur appliquons successivement les deux transformations suivantes :  </p>""", unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">• Inversion de la normalisation des variables continues (Ancienneté et toutes ses variables croisées, ainsi que Prix d’Origine)
      <br>• Multiplication du coefficient de Prix d’Origine par 1000 afin de le mettre à une échelle comparable aux autres variables (coefficient pour 1000 Euros de Prix d’Origine) </p>""", unsafe_allow_html=True)
     
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le graphe ci-dessous présente les coefficients des variables Ancienneté et Prix d’Origine ainsi que les 20 coefficients autres de plus grande valeur absolue :  </p>""", unsafe_allow_html=True)
     st.image("assets/images/coefs_svr_tx.png", use_column_width=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">En analysant les coefficients sous l’angle des constantes (décote initiale en sortie de magasin) et des coefficients croisés avec l’ancienneté (dépréciation annuelle) nous pouvons interpréter les différents facteurs explicatifs des prix d’occasion.  </p>""", unsafe_allow_html=True)
     option_effet = st.selectbox('**Facteur explicatif :**',('Effet sortie du magasin','Dépréciation annuelle','Effet Électrique',"Prix d'origine","État du vélo","Catégories","Marques"),index=0)
     if option_effet=='Effet sortie du magasin':
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Un vélo musculaire perd en moyenne 17,6% de sa valeur dès sa sortie de magasin (si déclaré dans l’Etat “Comme neuf”).</p>""", unsafe_allow_html=True)
     elif option_effet=='Dépréciation annuelle':
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Un vélo musculaire déclaré en “Très bon état” se déprécie ensuite de 3% par an en moyenne.</p>""", unsafe_allow_html=True)
     elif option_effet=='Effet Électrique':
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les vélos Électriques bénéficient d'une surcote initiale (en sortie de magasin) de 5.4% par rapport aux vélos musculaires mais se déprécient ensuite plus vite de 0.8% par an en moyenne.</p>""", unsafe_allow_html=True)
     elif option_effet=="Prix d'origine":
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Plus un vélo est cher, plus sa décote initiale est importante, de l'ordre de 1,7% par 1000 Euros de Prix d'origine supplémentaire.</p>""", unsafe_allow_html=True)
     elif option_effet=="État du vélo":
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les décotes initiales par rapport à un Etat “Comme Neuf” sont de : “Très Bon Etat” -6.9%, “Bon Etat”: -10.8%, “Usé”: -18.8%. Les rythmes de dépréciation annuelle sont ensuite moins prononcés pour ces Etats par rapport à “Comme Neuf “: “Très Bon État” se déprécie 2% moins vite par an, “Bon Etat”: 2.3% moins vite par an et “Usé”: 3.2% moins vite par an.</p>""", unsafe_allow_html=True)
     elif option_effet=="Catégories":
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Les catégories qui bénéficient de la plus grande surcote initiale sont généralement les catégories spécialisées performance (Dirt/jump +5,5%, Aero/CLM +3,6%, Trail/Enduro +4,6%, Gravel/CX +5,4% , Route +4,9%...), tandis que les catégories plus générales et de loisirs ou les catégories de niche sont décotées (Loisir -3,6%, Hybrides -2,1%, VTC-1,6%, Piste -3,6%, Pliant...). La dépréciation annuelle est généralement inversement ordonnée par rapport à la décote/surcote initiale et peut compenser en 2 à 3 ans l'effet de celle-ci. Il semble donc y avoir deux sortes de catégories : un groupe de catégories les plus demandées mais qui se déprécient plus vite dans le temps et un groupe de catégories moins recherchées mais qui sont perçues comme s'usant moins vite.</p>""", unsafe_allow_html=True)
     elif option_effet=="Marques":
          st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Certaines marques spécialisées et/ou bénéficiant d'une très bonne image bénéficient d'une très grande surcote initiale (Moustache +5,6%, Cervélo +7,9%, Canyon +9,2%...) par rapport aux marques non spécialisées dans le cyclisme (Gitane -1,0%, Matra...), aux marques d'entrée de gamme (Nakamura -6,7%, BTwin -1,0%) ou aux marques généralistes sans image d'excellence (Giant +0,0% par exemple). Le rythme de dépréciation annuelle est généralement inverse de la décote initiale (plus une marque est initialement décotée moins elle se déprécie vite en comparaison aux autres), mais cet effet est de moindre amplitude et est loin de pouvoir compenser la décote initiale même après 5 ans.</p>""", unsafe_allow_html=True)
     
     st.markdown('<h2 style="font-family:optima;color:#2596be;">Courbes de dépréciation  dans le temps</h2>', unsafe_allow_html=True)
     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;"> </p>""", unsafe_allow_html=True)

     # Création du graphe des courbes de dépréciation
     with st.form("inter_form"):
          st.markdown("<h3 style='text-align: center;padding-top:0rem'>Choisissez un type de vélo</h1>", unsafe_allow_html=True)
          st.markdown("<h4 style='text-align: left;padding-top:0rem'>Type 1</h1>", unsafe_allow_html=True)
          col1, col2 = st.columns([1,1])
          with col1:
               marque1=st.selectbox('Marque',liste_marques,key='marque1',index=34)
               catégorie1 = st.selectbox('Catégorie:',('Route','Aero / CLM','Gravel / CX','Piste','Trail / Enduro','Cross-country','Descente',
                                               'Dirt / Jump','Urbain','Hybrides','VTC','VTT','Loisir','Fatbike','Speedbike','Pliant'),key='categorie1',index=10)
               electrique1 = st.radio("Electrique ?", ('Oui', 'Non'),horizontal=True,key='electric1',index=1)
              
          with col2:
               prixorigine1=st.number_input('Prix achat vélo:',min_value=0, max_value=20000, step=1, format="%d", value=1500,key='prix1')  
               etatvelo1 = st.selectbox('État vélo:',('Comme Neuf','Très Bon État','Bon État','Usé'),key='etat1',index=1)
     
          st.markdown("<h4 style='text-align: left;padding-top:0rem'>Type 2</h1>", unsafe_allow_html=True)
          col1, col2 = st.columns([1,1])    
          with col1:
               marque2=st.selectbox('Marque',liste_marques,index=34)
               catégorie2 = st.selectbox('Catégorie:',('Route','Aero / CLM','Gravel / CX','Piste','Trail / Enduro','Cross-country','Descente',
                                               'Dirt / Jump','Urbain','Hybrides','VTC','VTT','Loisir','Fatbike','Speedbike','Pliant'),index=10)
               electrique2 = st.radio("Electrique ?", ('Oui', 'Non'),horizontal=True,index=1)     
          with col2:
               prixorigine2=st.number_input('Prix achat vélo:',min_value=0, max_value=20000, step=1, format="%d", value=1500)  
               etatvelo2 = st.selectbox('État vélo:',('Comme Neuf','Très Bon État','Bon État','Usé'),index=1)
    
          submitted = st.form_submit_button("Tracer")

     
     coefficients_h = joblib.load('assets/modeles/coefs_svr_tx_h.pkl')
     curves={}
     if submitted:
          if electrique1=="Oui":
               tag_elec1="Elec_True"
          else:
               tag_elec1="Elec_False"
          curves['c1'] = {"Elec":tag_elec1,"Etat":"Etat_"+etatvelo1,"PrixOrigine":prixorigine1,"Marque":"Marque_"+marque1,"Categorie":"Categorie_"+catégorie1}
          if electrique2=="Oui":
               tag_elec2="Elec_True"
          else:
               tag_elec2="Elec_False"
          curves['c2'] = {"Elec":tag_elec2,"Etat":"Etat_"+etatvelo2,"PrixOrigine":prixorigine2,"Marque":"Marque_"+marque2,"Categorie":"Categorie_"+catégorie2}        
          fig = graph_depreciation_curve(coefficients_h,curves)
          st.pyplot(fig, use_container_width=True)

with tab2:
     st.markdown('<h3 style="font-family:optima;color:#2596be;">Interprétation avec Skater</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons choisi d’étudier les résultats du modèle <i>GradientBoostingRegressor</i>.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Voici le résultat des 15 coefficients les + forts identifiés avec Skater :</p>"""
     , unsafe_allow_html=True)

     st.image("assets/images/skater_features.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ce résultat est à comparer avec celui de <i>LinearSVR</i>. On retrouve effectivement Ancienneté et PrixOrigine, 
     mais les autres coefficients diffèrent entre les deux modèles : 75% des 15 coefficients les + forts sont différents.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Cette approche par <i>GradientBoostingRegressor</i> n’a eu aucun effet sur la MAE. On a obtenu des résultats 
     très similaires avec le modèle <i>LinearSVR</i> mais l'interprétation a démontré une différence au niveau des coefficients.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Voici l’analyse locale d’un vélo sélectionné au hasard :</p>"""
     , unsafe_allow_html=True)

     st.image("assets/images/skater_local.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Et voici les caractéristiques du vélo :</p>"""
     , unsafe_allow_html=True)
     
     st.image("assets/images/skater_velo4118.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">On a choisi le vélo référencé #4418 dans le dataset à titre d’exemple. Il s’agit d’un vélo de la marque Felt, 
     acheté récemment à 2699 euros. Mais l'interprétation locale de ce vélo a identifié d’autres marques comme Neomouv/Bergamont/Adris (v=0) ou encore deux variables croisées Gitane/Adris avec 
     Ancienneté (v=0.05) dans le top 15 des features. Ces features présentes dans l'interprétation impactant la dépréciation du vélo n’ont aucun sens... Comment Gitane ou Adris seraient des 
     caratéristiques justifiant le calcul de la dépréciation d’un vélo d’une autre marque ? Néanmoins, on retrouve Ancienneté comme étant le coefficient avec le + fort impact sur la dépréciation. 
     On retrouve aussi l'état du vélo (Etat_Comme Neuf) dans la liste de ces coefficients. Finalement, le modèle a prédit une dépréciation de 13% vs. 11%.</p>"""
     , unsafe_allow_html=True)


with tab3:
     st.markdown('<h3 style="font-family:optima;color:#2596be;">Interprétation avec Shap</h3>', unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Nous avons choisi d’étudier les résultats du modèle <i>StackingRegressor</i>.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Voici le résultat des 15 coefficients les + forts identifiés avec Shap :</p>"""
     , unsafe_allow_html=True)

     st.image("assets/images/shap_features.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Ce résultat est à comparer avec celui de <i>LinearSVR</i>. On retrouve effectivement Ancienneté et PrixOrigine, 
     mais les autres coefficients diffèrent entre les deux modèles : 75% des 15 coefficients les + forts sont différents. Néanmoins, on retrouve des similitudes avec l'autre modèle ensembliste. 
     Seulement 40% de ces coefficients les + forts sont différents entre les modèles ensemblistes.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Cette approche par <i>StackingRegressor</i> n’a eu aucun effet sur la MAE. On a obtenu des résultats 
     très similaires avec le modèle <i>LinearSVR</i> mais l'interprétation a démontré une différence au niveau des coefficients.</p>"""
     , unsafe_allow_html=True)

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Voici l’analyse locale d’un vélo sélectionné au hasard :</p>"""
     , unsafe_allow_html=True)

     st.image("assets/images/shap_local.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Et voici les caractéristiques du vélo :</p>"""
     , unsafe_allow_html=True)
     
     st.image("assets/images/shap_velo4118.png")

     st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">On a d’un côté en bleu les coefficients ayant un impact positif sur la dépréciation, et en rouge les coefficients 
     ayant un impact négatif sur la dépréciation. On retrouve Ancienneté comme étant le coefficient avec le + fort impact sur la dépréciation. On retrouve aussi l'état du vélo (Etat_Comme Neuf) 
     dans la liste de ces coefficients. Par contre, ce modèle ensembliste semble avoir les mêmes défauts. On a des incohérences sur le choix des coefficients les + forts déterminant la dépréciation. 
     Logiquement, la dépréciation de ce vélo de la marque Felt ne devrait pas être impactée par une variable associée à une autre marque comme Argon 18. De même, la dépréciation de ce vélo est dépendante 
     des différentes variables liées à différents états de vélo. On devrait s’attendre à seulement avoir une seule variable liée à l'état de vétusté du vélo.</p>"""
     , unsafe_allow_html=True)