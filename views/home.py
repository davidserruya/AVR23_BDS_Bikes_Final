import streamlit as st
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu


#page home
st.image("assets/images/banner.png")
st.markdown('<h1 style="text-align: center;font-family:optima;">BDS_Bikes</h1>', unsafe_allow_html=True)

st.markdown('<h3 style="font-family:optima;color:#2596be;">Contexte du projet</h3>', unsafe_allow_html=True)
st.markdown('<h4 style="font-family:optima;color:#00c4cc;">Le marché du vélo d\'occasion en France 🥐</h4>', unsafe_allow_html=True)
st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le marché des vélos d\'occasion en France connaît une forte croissance, représentant plus de 120 millions d\'euros par an, dont plus de 90% sont des transactions entre particuliers.
                Pourtant, ce marché manque de transparence en raison de l\'absence d\'un registre officiel des transactions et d\'une côte officielle pour les vélos d\'occasion, similaires à la cote <i>Argus</i>.
                Les données exactes sur le volume et les prix ne sont pas facilement accessibles ni partagées, entravant ainsi l\'établissement d\'un marché efficace et équitable.
                <br><br>Aujourd\'hui, les estimations de la valeur des vélos d\'occasion reposent principalement sur l\'expérience et l\'intuition des acteurs impliqués, tels que les revendeurs et les particuliers. Cependant, ces évaluations sont souvent empiriques et ne sont pas vérifiées ou confirmées par des méthodes scientifiques.</p>""", unsafe_allow_html=True)


st.markdown('<h4 style="font-family:optima;color:#00c4cc;">Le marché du vélo d\'occasion à l\'étranger 🌎</h4>', unsafe_allow_html=True)
url = "https://www.bicyclebluebook.com/"
st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">À l'étranger, il existe quelques exemples, bien établis, développés par des entreprises privées dans différents pays comme aux Etats-Unis avec <a href={url}>www.bicyclebluebook.com</a>. Néanmoins, ces cas manquent aux critères de transparence sur l’algorithme que l’on attendrait d’une côte qui aurait pour ambition de servir de référence commune à un marché national.</p><br>""", unsafe_allow_html=True)


image='assets/images/stripe.png'
st.image(image, use_column_width=True)

st.markdown('<h3 style="font-family:optima;color:#2596be;">Problématique</h3>', unsafe_allow_html=True)
st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Le marché du vélo d'occasion souffre d'un manque de transparence et de références officielles, ce qui limite l'établissement d'un marché efficace et équitable. 
L'absence d'un registre officiel des transactions et d'une côte officielle pour les vélos d'occasion rend difficile la détermination de la valeur réelle des vélos sur ce marché.
 Les estimations actuelles reposent principalement sur des évaluations subjectives basées sur l'expérience et l'intuition des acteurs impliqués.</p><br>""", unsafe_allow_html=True)
st.image(image, use_column_width=True)

st.markdown('<h3 style="font-family:optima;color:#2596be;">Objectif du projet</h3>', unsafe_allow_html=True)
st.markdown('<h4 style="font-family:optima;color:#00c4cc;">Création d\'une côte de vélo 🛞</h4>', unsafe_allow_html=True)
st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">L’objectif du projet est donc d’analyser la formation des prix sur le marché des vélos d’occasion en France, de déterminer si il existe des variables qui déterminent les prix de revente et si il est possible de construire une côte, actualisée dynamiquement, qui puisse servir de point de référence aux acheteurs et aux vendeurs. 
                <br><br>Cette côte devrait pouvoir s’appliquer à tout type de vélos, quels que soient leur modèle, millésime, usure, catégorie, marque, état de vétusté, avec ou sans assistance électrique, et refléter objectivement le prix moyen de revente de ces vélos à un moment donné.</p><br>""", unsafe_allow_html=True)
    
st.markdown('<h4 style="font-family:optima;color:#00c4cc;">Vers une côte transparente et durable 🌿</h4>', unsafe_allow_html=True)
     
st.markdown("""<p style="font-family:Arial,sans-serif;text-align: justify;">Notre projet repose sur l'utilisation de méthodes de Data Science avec pour objectif d'apporter une transparence et une clarté à nos travaux et résultats.
            En renforçant la transparence et en fournissant des informations fiables sur ce marché, nous visons à établir un environnement de confiance pour tous les acteurs impliqués. Cela favorisera le développement des échanges et contribuera à promouvoir une industrie du vélo plus socialement responsable et durable.</p><br>""", unsafe_allow_html=True)
    
st.image(image, use_column_width=True)

st.markdown('<h3 style="font-family:optima;color:#2596be;">L\'équipe du projet</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("assets/images/team.png")

with col3:
    st.write(' ')
    
    
# Pied de page
st.markdown("---")
st.markdown("© 2023. Tous droits réservés.")




    