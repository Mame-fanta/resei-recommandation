# ============================================================
# RESEI — Application finale de recommandation IA
# Prototype MVP pour mémoire OZAMBA / RESEI
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="RESEI — Orientation IA",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "RESEI_dataset_5000_final.csv"
MODEL_PATH = "modele_resei_recommandation.joblib"

# Charte RESEI
PRIMARY = "#1A3A6A"
DARK = "#161329"
DARKER = "#0F0E1F"
ACCENT = "#57C1DA"
WHITE = "#FFFFFF"
BG = "#F5F7FA"
TEXT = "#161329"
MUTED = "#657185"
BORDER = "#DDE7F0"
SOFT = "#EEF7FA"
SUCCESS = "#0F8A5F"
WARNING = "#B7791F"
DANGER = "#C24141"

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Outfit', 'Segoe UI', sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {DARKER} 0%, {DARK} 58%, {PRIMARY} 100%); }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div {{ color: #F5F7FA; }}
.brand-card {{ padding: 18px; border-radius: 22px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); margin-bottom: 24px; }}
.brand-row {{ display:flex; align-items:center; gap:12px; }}
.brand-mark {{ width:48px; height:48px; border-radius:14px; background:#D9F6FB; color:{PRIMARY}; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:22px; }}
.brand-title {{ font-size:27px; font-weight:900; color:white; letter-spacing:.02em; }}
.brand-subtitle {{ font-size:12px; color:#D7DAE5; margin-top:-4px; }}
.sidebar-note {{ margin-top:35px; padding-top:24px; border-top:1px solid rgba(255,255,255,.20); color:#C9D0DE; font-size:13px; line-height:1.6; }}
.page-title {{ font-size:34px; font-weight:900; color:{TEXT}; letter-spacing:-.03em; margin-bottom:8px; }}
.page-subtitle {{ font-size:16px; line-height:1.6; color:{MUTED}; max-width:1040px; margin-bottom:24px; }}
.hero {{ background:{PRIMARY}; border-radius:30px; padding:42px; color:white; min-height:250px; box-shadow:0 18px 40px rgba(26,58,106,.20); position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-110px; top:-120px; width:390px; height:390px; border-radius:50%; background:radial-gradient(circle, rgba(87,193,218,.60), rgba(87,193,218,.05) 66%, transparent 72%); }}
.hero-kicker {{ font-size:13px; font-weight:800; color:#D9F6FB; text-transform:uppercase; letter-spacing:.08em; margin-bottom:14px; position:relative; z-index:1; }}
.hero-title {{ font-size:44px; line-height:1.08; font-weight:900; letter-spacing:-.04em; margin-bottom:18px; position:relative; z-index:1; color:white; }}
.hero-text {{ font-size:16px; line-height:1.7; color:#EAF6FA; max-width:780px; position:relative; z-index:1; }}
.card, .soft-card, .metric-card {{ background:white; border:1px solid {BORDER}; border-radius:22px; padding:24px; box-shadow:0 10px 24px rgba(26,58,106,.06); color:{TEXT}; }}
.soft-card {{ height:100%; }}
.card h1,.card h2,.card h3,.card h4,.card p,.card span,.card div,.soft-card h3,.soft-card p {{ color:{TEXT}; }}
.step-label {{ display:inline-block; padding:6px 12px; border-radius:999px; background:{SOFT}; color:{PRIMARY}; font-size:12px; font-weight:800; margin-top:14px; }}
.metric-card {{ min-height:118px; }}
.metric-label {{ color:{MUTED}; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.07em; }}
.metric-value {{ color:{TEXT}; font-size:28px; line-height:1.15; font-weight:900; margin-top:10px; }}
.metric-delta {{ color:{SUCCESS}; font-size:12px; font-weight:800; margin-top:8px; }}
.result-card {{ background:{DARK}; border-radius:26px; padding:30px; color:white; box-shadow:0 18px 42px rgba(22,19,41,.25); margin-top:22px; }}
.result-card * {{ color:white; }}
.result-kicker {{ color:#BDECF5; font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
.result-title {{ color:white; font-size:34px; font-weight:900; line-height:1.15; margin-top:10px; margin-bottom:20px; }}
.score {{ color:{ACCENT}; font-size:56px; font-weight:900; line-height:1; }}
.explain-box {{ background:{SOFT}; border:1px solid #CDECF4; border-radius:20px; padding:20px; color:{TEXT}; }}
.explain-box * {{ color:{TEXT}; }}
.module-row {{ padding:14px 0; border-bottom:1px solid #EEF2F6; display:flex; justify-content:space-between; gap:12px; align-items:center; }}
.badge {{ padding:6px 10px; border-radius:999px; background:{SOFT}; color:{PRIMARY}; font-size:11px; font-weight:800; white-space:nowrap; }}
.success-box {{ background:#EAF8F1; border:1px solid #B8E6CF; color:#064E3B; border-radius:18px; padding:16px; margin-bottom:12px; }}
.alert-box {{ background:#FFF7ED; border:1px solid #FED7AA; color:#7C2D12; border-radius:18px; padding:16px; margin-bottom:12px; }}
div[data-testid="stButton"] > button {{ background:{PRIMARY}; color:white; border:0; border-radius:14px; padding:.70rem 1.20rem; font-weight:800; }}
div[data-testid="stButton"] > button:hover {{ background:{DARK}; color:white; border:0; }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
.stTabs [data-baseweb="tab"] {{ background:#EAF0F6; border-radius:999px; padding:9px 18px; color:{TEXT}; font-weight:800; }}
.stTabs [aria-selected="true"] {{ background:{PRIMARY}; color:white; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Chargement modèle et données
# ------------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    st.error("Modèle introuvable : modele_resei_recommandation.joblib")
    st.stop()

df = load_dataset()
model = load_model()

# ------------------------------------------------------------
# Session
# ------------------------------------------------------------
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = [
        {"parcours":"Sport Agent Certificate", "pertinence":"Positif", "action":"Inscription au parcours", "commentaire":"La recommandation correspond bien à mon objectif."},
        {"parcours":"Gaming Business Certificate", "pertinence":"Neutre", "action":"Consultation d’un autre parcours", "commentaire":"J’hésite encore avec Entertainment."},
        {"parcours":"Esport Management Certificate", "pertinence":"Négatif", "action":"Refus de la recommandation", "commentaire":"Je cherche plutôt un parcours gaming."},
    ]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def details_for(prediction):
    data = {
        "Sport Agent Certificate": {
            "tagline":"Gestion de carrière, négociation, scouting et représentation sportive.",
            "modules":["Introduction au métier d’agent sportif", "Négociation et contrats", "Scouting et détection de talents", "Gestion de carrière et image du joueur", "Cas pratique final"],
            "opportunities":["Atelier négociation", "Rencontre scout", "Simulation contrat"]},
        "Sport Marketing Certificate": {
            "tagline":"Marketing sportif, sponsoring, activation de marque et stratégie digitale.",
            "modules":["Fondamentaux du marketing sportif", "Sponsoring et partenariats", "Communication de marque", "Campagnes digitales sport", "Étude de cas marque sportive"],
            "opportunities":["Mission sponsoring", "Challenge marque", "Atelier campagne digitale"]},
        "Esport Management Certificate": {
            "tagline":"Management d’équipe esport, organisation, tournois et partenariats.",
            "modules":["Écosystème esport", "Gestion d’équipe", "Organisation de tournois", "Partenariats esport", "Projet final management esport"],
            "opportunities":["Simulation tournoi", "Live manager esport", "Projet équipe"]},
        "Gaming Business Certificate": {
            "tagline":"Création, monétisation et développement d’un projet gaming.",
            "modules":["Marché du gaming", "Business model gaming", "Création de contenu gaming", "Communauté et stratégie digitale", "Pitch projet gaming"],
            "opportunities":["Atelier pitch", "Challenge créateur", "Analyse chaîne gaming"]},
        "Entertainment Management Certificate": {
            "tagline":"Management d’artiste, événementiel, contenu et développement de carrière.",
            "modules":["Industries créatives et entertainment", "Management d’artiste", "Événementiel et production", "Stratégie de contenu", "Cas pratique lancement artiste"],
            "opportunities":["Projet événement", "Masterclass manager", "Analyse carrière artiste"]},
        "Women Sports Leadership Certificate": {
            "tagline":"Leadership, impact, projet et développement du sport féminin.",
            "modules":["Enjeux du sport féminin", "Leadership et prise de parole", "Développement de projet", "Partenariats et impact", "Plan d’action final"],
            "opportunities":["Mentorat leadership", "Projet impact", "Réseau sport féminin"]},
    }
    return data.get(prediction, {"tagline":"Parcours personnalisé.", "modules":["Module 1", "Module 2"], "opportunities":["Opportunité"]})

def predict_profile(input_data):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    if hasattr(model.named_steps["model"], "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = model.named_steps["model"].classes_
        proba_df = pd.DataFrame({"Parcours": classes, "Score": probs}).sort_values("Score", ascending=False)
        score = float(proba_df.iloc[0]["Score"]) * 100
    else:
        proba_df = pd.DataFrame({"Parcours":[pred], "Score":[1.0]})
        score = 100.0
    return pred, score, proba_df

def explain(data, pred):
    return (
        f"Le parcours **{pred}** est recommandé car le profil indique un intérêt pour **{data['domaine_interet']}**, "
        f"un objectif professionnel orienté vers **{data['objectif_professionnel']}**, et une volonté de développer "
        f"la compétence **{data['competences_recherchees']}**. Le niveau déclaré est **{data['niveau_connaissance']}**, "
        f"avec une expérience secteur **{data['experience_secteur']}**."
    )

def feedback_metrics():
    f = pd.DataFrame(st.session_state.feedbacks)
    total = len(f)
    if total == 0:
        return 0, "0%", "0%", "0%"
    return total, f"{round((f['pertinence'].eq('Positif').sum()/total)*100)}%", f"{round((f['pertinence'].eq('Neutre').sum()/total)*100)}%", f"{round((f['pertinence'].eq('Négatif').sum()/total)*100)}%"

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.markdown("""
<div class="brand-card">
  <div class="brand-row">
    <div class="brand-mark">R</div>
    <div>
      <div class="brand-title">RESEI</div>
      <div class="brand-subtitle">Définissez votre héritage</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "Accueil", "Diagnostic IA", "Dashboard apprenant", "Feedbacks", "Espace créateur", "Back-office admin", "Modèle & données"
])

st.sidebar.markdown("""
<div class="sidebar-note">Prototype MVP : diagnostic, recommandation IA, score, feedbacks et pilotage des KPIs.</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Accueil
# ------------------------------------------------------------
if page == "Accueil":
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Define your legacy</div>
      <div class="hero-title">Recommander le bon parcours,<br>au bon apprenant.</div>
      <div class="hero-text">RESEI aide chaque apprenant à choisir un parcours adapté dans le sport business, l’esport, le gaming et l’entertainment grâce à un diagnostic personnalisé, un modèle prédictif, un score d’adéquation et un suivi dans le dashboard.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="soft-card"><h3>Diagnostic personnalisé</h3><p>L’apprenant renseigne son profil, son niveau, son objectif professionnel et ses préférences d’apprentissage.</p><span class="step-label">Étape 1</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="soft-card"><h3>Recommandation IA expliquée</h3><p>Le modèle propose un parcours adapté, affiche un score d’adéquation et explique les raisons de la recommandation.</p><span class="step-label">Étape 2</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="soft-card"><h3>Suivi du parcours</h3><p>L’apprenant suit sa progression, accède aux contenus, aux lives et peut donner un feedback sur la recommandation.</p><span class="step-label">Étape 3</span></div>', unsafe_allow_html=True)
    st.write("")
    a,b,c,d = st.columns(4)
    with a: metric_card("Parcours disponibles", "6")
    with b: metric_card("Dataset POC", "5000 profils")
    with c: metric_card("Modèle retenu", "Random Forest")
    with d: metric_card("F1-score", "0,7148")

# ------------------------------------------------------------
# Diagnostic
# ------------------------------------------------------------
elif page == "Diagnostic IA":
    page_header("Diagnostic IA", "L’apprenant complète un diagnostic guidé. Le modèle analyse ses réponses et propose un parcours avec un score d’adéquation.")
    with st.form("diagnostic"):
        tab1, tab2, tab3 = st.tabs(["Profil", "Objectif", "Engagement"])
        with tab1:
            c1,c2,c3 = st.columns(3)
            with c1:
                age = st.slider("Âge", 18, 45, 23)
                profil = st.selectbox("Profil", ["Étudiant", "Jeune diplômé", "En reconversion", "Entrepreneur", "Créateur de contenu", "Professionnel du sport"])
            with c2:
                niveau_etude = st.selectbox("Niveau d'étude", ["bac", "licence", "master", "autodidacte", "professionnel"])
                niveau_connaissance = st.selectbox("Niveau de connaissance", ["Débutant", "Intermédiaire", "Avancé"])
            with c3:
                experience_secteur = st.selectbox("Expérience secteur", ["Aucune", "< 1 an", "1-3 ans", "3-5 ans", "5+ ans"])
                budget = st.selectbox("Budget", ["Faible", "Moyen", "Élevé"])
        with tab2:
            c1,c2,c3 = st.columns(3)
            with c1:
                domaine_interet = st.selectbox("Domaine d'intérêt", ["Sport Business", "Marketing Sportif", "Esport", "Gaming", "Entertainment", "Women Leadership", "Football", "Événementiel Sportif", "Innovation & Technologie"])
                objectif_professionnel = st.selectbox("Objectif professionnel", ["Devenir agent sportif", "Représenter des athlètes", "Responsable marketing sportif", "Développer des partenariats sportifs", "Manager une équipe esport", "Organiser des compétitions esport", "Travailler dans l’esport", "Créer un projet gaming", "Développer un business gaming", "Développer une carrière dans le gaming", "Gérer une communauté gaming", "Gérer des artistes ou talents", "Talent manager", "Produire des événements entertainment", "Leader dans le sport féminin", "Développer des projets d’inclusion dans le sport", "Créer une entreprise Sport Tech", "Diriger une organisation sportive", "Directeur de club sportif", "Responsable communication sportive", "Développer une marque sportive"])
            with c2:
                competences_recherchees = st.selectbox("Compétence recherchée", ["Négociation", "Marketing", "Communication", "Gestion de projet", "Business development", "Scouting", "Esport", "Gaming", "Leadership", "Digital", "Branding", "Analyse de données", "Community management", "Inclusion", "Production", "Stratégie", "Événementiel"])
                objectif_court_terme = st.selectbox("Objectif court terme", ["découvrir un métier", "acquérir une compétence", "obtenir un certificat", "trouver une opportunité professionnelle", "développer un projet personnel"])
            with c3:
                langue_preferee = st.selectbox("Langue préférée", ["Français", "Anglais", "Arabe"])
                format_prefere = st.selectbox("Format préféré", ["Vidéo", "Live", "PDF + exercices", "Mixte"])
        with tab3:
            c1,c2,c3 = st.columns(3)
            with c1:
                temps_disponible_semaine = st.slider("Temps disponible par semaine", 2, 20, 8)
                motivation_score = st.slider("Motivation", 25, 100, 80)
            with c2:
                autonomie_score = st.slider("Autonomie", 1, 10, 7)
                progression_moyenne = st.slider("Progression moyenne actuelle (%)", 0, 100, 50)
            with c3:
                engagement_score = st.slider("Engagement", 1, 10, 7)
                participation_live = st.selectbox("Participation aux lives", ["jamais", "rarement", "parfois", "souvent"])
                ressources_sauvegardees = st.slider("Ressources sauvegardées", 0, 30, 8)
                notes_personnelles = st.slider("Notes personnelles", 0, 50, 12)
                feedback_recommandation = st.selectbox("Feedback précédent", ["positif", "neutre", "négatif", "aucun"])
        submitted = st.form_submit_button("Analyser mon profil et recommander un parcours")

    input_data = {"age":age,"profil":profil,"niveau_etude":niveau_etude,"niveau_connaissance":niveau_connaissance,"domaine_interet":domaine_interet,"objectif_professionnel":objectif_professionnel,"competences_recherchees":competences_recherchees,"experience_secteur":experience_secteur,"temps_disponible_semaine":temps_disponible_semaine,"budget":budget,"langue_preferee":langue_preferee,"format_prefere":format_prefere,"motivation_score":motivation_score,"autonomie_score":autonomie_score,"objectif_court_terme":objectif_court_terme,"progression_moyenne":progression_moyenne,"engagement_score":engagement_score,"participation_live":participation_live,"ressources_sauvegardees":ressources_sauvegardees,"notes_personnelles":notes_personnelles,"feedback_recommandation":feedback_recommandation}

    if submitted:
        pred, score, proba_df = predict_profile(input_data)
        st.session_state.prediction = pred
        st.session_state.score = score
        st.session_state.proba_df = proba_df
        st.session_state.input_data = input_data
        st.session_state.explanation = explain(input_data, pred)

    if "prediction" in st.session_state:
        pred = st.session_state.prediction
        score = st.session_state.score
        details = details_for(pred)
        st.markdown(f'<div class="result-card"><div class="result-kicker">Parcours recommandé</div><div class="result-title">{pred}</div><div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;"><div><div class="result-kicker">Score d’adéquation</div><div class="score">{score:.1f}%</div></div><div style="max-width:680px;font-size:16px;line-height:1.6;">{details["tagline"]}</div></div></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1.15,.85])
        with col1:
            st.markdown('<div class="explain-box">', unsafe_allow_html=True)
            st.markdown("### Pourquoi ce parcours ?")
            st.markdown(st.session_state.explanation)
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("### Modules proposés")
            for i,m in enumerate(details["modules"],1):
                st.markdown(f'<div class="module-row"><div><b>Module {i}</b><br><span style="color:{MUTED};">{m}</span></div><span class="badge">À planifier</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown("### Classement des parcours")
            chart = st.session_state.proba_df.copy()
            chart["Score"] = chart["Score"] * 100
            st.bar_chart(chart.set_index("Parcours"))
            st.markdown("### Feedback rapide")
            pertinence = st.radio("Cette recommandation est-elle pertinente ?", ["Positif", "Neutre", "Négatif"], horizontal=True)
            action = st.selectbox("Action après recommandation", ["Inscription au parcours", "Consultation d’un autre parcours", "Refus de la recommandation", "À décider plus tard"])
            commentaire = st.text_area("Commentaire", placeholder="Expliquer le retour de l’apprenant...")
            if st.button("Enregistrer le feedback"):
                st.session_state.feedbacks.append({"parcours":pred,"pertinence":pertinence,"action":action,"commentaire":commentaire if commentaire else "Aucun commentaire"})
                st.success("Feedback enregistré.")

# ------------------------------------------------------------
# Dashboard apprenant
# ------------------------------------------------------------
elif page == "Dashboard apprenant":
    page_header("Dashboard apprenant", "La recommandation devient un parcours d’apprentissage : progression, modules, contenus, lives et espace personnel.")
    if "prediction" not in st.session_state:
        st.info("Faites d’abord un diagnostic IA pour personnaliser ce dashboard.")
    else:
        pred, score, data = st.session_state.prediction, st.session_state.score, st.session_state.input_data
        details = details_for(pred)
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Parcours", pred)
        with c2: metric_card("Score IA", f"{score:.1f}%", "+ recommandation forte")
        with c3: metric_card("Progression", f"{data['progression_moyenne']}%")
        with c4: metric_card("Engagement", f"{data['engagement_score']}/10")
        st.write("")
        tabs = st.tabs(["Overview", "Profile", "My Path", "My Learning Space", "Recommendations", "Library", "Live Sessions"])
        with tabs[0]:
            a,b = st.columns([1.2,.8])
            with a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### {pred}")
                st.write(details["tagline"])
                st.progress(min(int(data["progression_moyenne"]),100))
                st.write("**Prochaine action recommandée :** commencer le premier module du parcours.")
                st.markdown('</div>', unsafe_allow_html=True)
            with b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Objectif de la semaine")
                st.write("Terminer un module, sauvegarder deux ressources et préparer une question pour la prochaine Live Session.")
                st.write(f"**Motivation :** {data['motivation_score']}/10")
                st.write(f"**Autonomie :** {data['autonomie_score']}/10")
                st.write(f"**Ressources sauvegardées :** {data['ressources_sauvegardees']}")
                st.markdown('</div>', unsafe_allow_html=True)
        with tabs[1]: st.dataframe(pd.DataFrame([data]).T.rename(columns={0:"Valeur"}), use_container_width=True)
        with tabs[2]: st.dataframe(pd.DataFrame({"Étape":[f"Module {i}" for i in range(1,len(details['modules'])+1)],"Contenu":details['modules'],"Statut":["À commencer","À commencer","À commencer","Programmé","À venir"][:len(details['modules'])]}), use_container_width=True)
        with tabs[3]:
            a,b = st.columns(2)
            with a:
                st.text_area("Notes personnelles", placeholder="Ex : points importants du module 1...")
                st.text_area("Questions pour les lives", placeholder="Préparer mes questions...")
            with b:
                st.text_input("Objectif de la semaine", value="Terminer le premier module")
                st.multiselect("Ressources favorites", ["Guide métier","Replay live","Checklist projet","Modèle de pitch"])
                st.checkbox("Ajouter ce parcours aux favoris")
        with tabs[4]: st.dataframe(pd.DataFrame({"Type":["Parcours","Module","Live Session","Ressource","Opportunité"],"Recommandation":[pred,details['modules'][0],"Live avec un expert","Guide PDF métier",details['opportunities'][0]],"Raison":["Meilleur score IA","Début du parcours","Lien avec l’objectif","Ressource utile","Adaptée au profil"]}), use_container_width=True)
        with tabs[5]: st.dataframe(pd.DataFrame({"Contenu":["Masterclass introduction","Replay expert","Fiche métier","Cas pratique","Quiz final"],"Format":["Vidéo","Replay","PDF","Exercice","Quiz"],"Durée":["25 min","45 min","8 pages","30 min","15 min"],"Recommandé":["Oui","Oui","Oui","Oui","Plus tard"]}), use_container_width=True)
        with tabs[6]: st.dataframe(pd.DataFrame({"Live Session":["Comprendre le métier","Construire son réseau","Cas pratique avec expert"],"Date":["Semaine 1","Semaine 2","Semaine 4"],"Statut":["Disponible","Recommandé","À venir"]}), use_container_width=True)

# ------------------------------------------------------------
# Feedbacks
# ------------------------------------------------------------
elif page == "Feedbacks":
    page_header("Feedbacks", "Les feedbacks permettent de mesurer la pertinence des recommandations et d’améliorer progressivement le modèle IA.")
    total, pos, neutre, neg = feedback_metrics()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Feedbacks reçus", str(total))
    with c2: metric_card("Positifs", pos)
    with c3: metric_card("Neutres", neutre)
    with c4: metric_card("Négatifs", neg)
    st.write("")
    a,b = st.columns([.9,1.1])
    with a:
        st.markdown("### Ajouter un feedback")
        parcours = st.selectbox("Parcours concerné", ["Sport Agent Certificate", "Sport Marketing Certificate", "Esport Management Certificate", "Gaming Business Certificate", "Entertainment Management Certificate", "Women Sports Leadership Certificate"])
        pertinence = st.radio("Pertinence", ["Positif", "Neutre", "Négatif"], horizontal=True)
        action = st.selectbox("Action après recommandation", ["Inscription au parcours", "Consultation d’un autre parcours", "Refus de la recommandation", "À décider plus tard"])
        commentaire = st.text_area("Commentaire", placeholder="Ex : recommandation claire, pas assez précise...")
        if st.button("Enregistrer ce feedback"):
            st.session_state.feedbacks.append({"parcours":parcours,"pertinence":pertinence,"action":action,"commentaire":commentaire if commentaire else "Aucun commentaire"})
            st.success("Feedback ajouté.")
    with b:
        st.markdown("### Derniers retours")
        feedback_df = pd.DataFrame(st.session_state.feedbacks)
        st.dataframe(feedback_df, use_container_width=True)
        if not feedback_df.empty:
            counts = feedback_df["pertinence"].value_counts().reset_index(); counts.columns=["Pertinence","Nombre"]
            st.bar_chart(counts.set_index("Pertinence"))

# ------------------------------------------------------------
# Espace créateur
# ------------------------------------------------------------
elif page == "Espace créateur":
    page_header("Espace créateur / expert", "Les experts peuvent gérer leurs contenus, programmer des lives et suivre la performance de leurs formations.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Contenus publiés", "18", "+3 ce mois")
    with c2: metric_card("Apprenants inscrits", "426", "+11%")
    with c3: metric_card("Taux de complétion", "58%", "+4 pts")
    with c4: metric_card("Note moyenne", "4.6/5", "Très positif")
    st.write("")
    tabs = st.tabs(["Overview", "My Content", "Live Sessions", "Analytics"])
    with tabs[0]: st.write("- 42 nouveaux apprenants cette semaine\n- 3 feedbacks demandent plus d’exemples pratiques\n- 1 live session programmée\n- 2 contenus en attente de validation")
    with tabs[1]: st.dataframe(pd.DataFrame({"Contenu":["Introduction sport business","Négociation","Sponsoring","Cas pratique","Replay expert"],"Statut":["Publié","Publié","Brouillon","À valider","Publié"],"Apprenants":[120,88,0,0,64],"Complétion":["61%","54%","-","-","48%"]}), use_container_width=True)
    with tabs[2]: st.dataframe(pd.DataFrame({"Live":["Q&A métier","Atelier cas pratique","Session mentorat"],"Date":["Vendredi","Semaine prochaine","À planifier"],"Participants":[72,0,0],"Statut":["Programmé","Brouillon","Idée"]}), use_container_width=True)
    with tabs[3]: st.line_chart(pd.DataFrame({"Vues":[80,110,130,125,160,190],"Inscriptions":[15,18,25,22,31,35],"Complétions":[8,10,14,13,18,22]}))

# ------------------------------------------------------------
# Admin
# ------------------------------------------------------------
elif page == "Back-office admin":
    page_header("Back-office administrateur", "L’équipe RESEI pilote la plateforme avec les KPIs, les revenus en euros, les recommandations et les feedbacks.")
    total,pos,neutre,neg = feedback_metrics()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Utilisateurs actifs", "1 248", "+12%")
    with c2: metric_card("Recommandations", "842", "+18%")
    with c3: metric_card("Acceptation", "68%", "+5 pts")
    with c4: metric_card("Revenus", "3 850 €", "+9%")
    st.write("")
    tabs = st.tabs(["Overview", "Users", "Content Management", "Recommendations", "Analytics", "Alerts"])
    with tabs[0]:
        indicator = st.selectbox("Indicateur à afficher", ["Active users","Visits","Sessions","Enrollments","Started paths","Certificates","Revenue (€)","Completion rate"])
        vals = {"Active users":[620,690,720,760,820,900],"Visits":[1200,1350,1500,1480,1720,1900],"Sessions":[820,880,940,1000,1080,1200],"Enrollments":[90,105,120,118,135,148],"Started paths":[70,82,88,94,103,121],"Certificates":[12,18,22,26,30,38],"Revenue (€)":[1200,1500,1800,2000,2800,3850],"Completion rate":[42,45,47,50,52,54]}
        st.line_chart(pd.DataFrame({"Semaine":["S1","S2","S3","S4","S5","S6"], indicator: vals[indicator]}).set_index("Semaine"))
    with tabs[1]: st.dataframe(pd.DataFrame({"Rôle":["Apprenant","Créateur","Administrateur"],"Nombre":[1248,36,5],"Actifs 30 jours":[842,28,5]}), use_container_width=True)
    with tabs[2]: st.dataframe(pd.DataFrame({"Contenu":["Masterclass sponsoring","Guide agent sportif","Replay esport","Module leadership"],"Créateur":["Expert A","Expert B","Expert C","Expert D"],"Statut":["À valider","Publié","À corriger","Publié"],"Priorité":["Haute","Normale","Haute","Normale"]}), use_container_width=True)
    with tabs[3]:
        if df is not None:
            dist = df["parcours_recommande"].value_counts().reset_index(); dist.columns=["Parcours","Nombre"]
            st.bar_chart(dist.set_index("Parcours"))
        st.dataframe(pd.DataFrame({"Parcours":["Sport Agent Certificate","Sport Marketing Certificate","Esport Management Certificate","Gaming Business Certificate","Entertainment Management Certificate","Women Sports Leadership Certificate"],"Taux acceptation":["66%","71%","62%","59%","65%","74%"],"Taux inscription":["51%","55%","46%","44%","49%","60%"],"Taux abandon":["22%","18%","26%","29%","24%","16%"]}), use_container_width=True)
    with tabs[4]: st.dataframe(pd.DataFrame({"KPI":["Taux de clic","Taux d’acceptation","Taux d’inscription","Taux de complétion","Satisfaction","Taux d’abandon","Feedbacks positifs"],"Valeur":["74%","68%","52%","54%","4.4/5","21%",pos],"Interprétation":["Les recommandations sont consultées","Les parcours proposés sont globalement pertinents","Une partie importante passe à l’action","La complétion doit encore être améliorée","Les retours sont positifs","Point de vigilance sur certains parcours","Les feedbacks alimentent l’amélioration continue"]}), use_container_width=True)
    with tabs[5]:
        st.markdown('<div class="alert-box"><b>Feedbacks :</b> analyser les retours négatifs pour améliorer le diagnostic.</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box"><b>Contenus :</b> 2 modules attendent une validation administrateur.</div>', unsafe_allow_html=True)
        st.markdown('<div class="success-box"><b>Signal positif :</b> Women Sports Leadership Certificate obtient un bon taux d’acceptation.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Modèle & données
# ------------------------------------------------------------
elif page == "Modèle & données":
    page_header("Modèle IA et données", "Cette page explique la partie technique : dataset POC de 5000 profils, extraction SQL, validation réelle exploratoire, modèles testés, résultats et limites.")
    c1,c2 = st.columns([1,1])
    with c1:
        if df is not None:
            metric_card("Lignes du dataset", str(df.shape[0]))
            st.write("")
            st.dataframe(df.head(8), use_container_width=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Pipeline technique")
        st.write("1. Simulation de profils apprenants")
        st.write("2. Séparation X / y")
        st.write("3. One-Hot Encoding des variables catégorielles")
        st.write("4. Standardisation des variables numériques")
        st.write("5. Entraînement des modèles")
        st.write("6. Choix du Random Forest")
        st.write("7. Parcours recommandé + score")
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    results = pd.DataFrame({"Modèle":["Random Forest","Logistic Regression","Decision Tree","KNN"],"Accuracy":[0.752,0.743,0.708,0.659],"F1-score":[0.7527,0.7429,0.7094,0.6595],"Temps prédiction moyen":["0.634 ms","0.016 ms","0.015 ms","0.148 ms"]})
    st.dataframe(results, use_container_width=True)
    st.bar_chart(results.set_index("Modèle")[["F1-score"]])
    st.write("")
    st.markdown('<div class="card"><p><b>Données simulées :</b> elles démontrent la faisabilité, mais le modèle devra être validé avec des données réelles.</p><p><b>Feedbacks :</b> les clics, inscriptions, refus et abandons permettront d’améliorer la recommandation.</p><p><b>Explicabilité :</b> le score d’adéquation reste une aide à la décision, pas une vérité absolue.</p></div>', unsafe_allow_html=True)
