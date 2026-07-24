"""
Olist BI Copilot — Version complète (Etapes 1 à 18)
Lancer : streamlit run olist_copilot.py
Installer : pip install streamlit plotly pandas numpy pyarrow google-generativeai scikit-learn prophet fpdf2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import datetime, io, base64, warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Olist BI Copilot", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0D1117}
[data-testid="stSidebar"]{background:#161B22;border-right:1px solid rgba(255,255,255,0.08)}
[data-testid="stSidebar"] *{color:#E6EDF3!important}
h1,h2,h3{color:#E6EDF3!important}
.block-container{padding:1.5rem 2rem}
.kpi-card{background:#1C2333;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:18px 20px;margin-bottom:4px}
.kpi-label{font-size:11px;color:#8B949E;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
.kpi-value{font-size:28px;font-weight:700;font-family:monospace;line-height:1}
.kpi-sub{font-size:11px;color:#8B949E;margin-top:5px}
.alert-red{background:#2D1515;border:1px solid #D85A30;border-radius:10px;padding:12px 16px;margin:6px 0}
.alert-orange{background:#2D1F10;border:1px solid #BA7517;border-radius:10px;padding:12px 16px;margin:6px 0}
.alert-green{background:#0F2018;border:1px solid #1D9E75;border-radius:10px;padding:12px 16px;margin:6px 0}
.summary-box{background:#161B22;border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:20px 24px;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

TEAL,PURPLE,CORAL,AMBER,BLUE = "#1D9E75","#534AB7","#D85A30","#BA7517","#185FA5"

# ════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path):
    master    = pd.read_parquet(f"{path}olist_master.parquet")
    rfm       = pd.read_parquet(f"{path}olist_rfm.parquet")
    delivered = master[master["order_status"]=="delivered"].copy()
    bins   = [-999,-7,-3,0,3,7,14,999]
    labels = ["<-7j","-7a-3j","-3a0j","0a+3j","+3a+7j","+7a+14j",">+14j"]
    delivered["delay_bucket"] = pd.cut(delivered["delay_days"],bins=bins,labels=labels)
    return master, delivered, rfm

@st.cache_data
def compute_kpis(master_hash, delivered_hash, rfm_hash, path):
    master, delivered, rfm = load_data(path)
    monthly = (master.groupby("purchase_yearmonth")["total_price"].sum()
               .reset_index().sort_values("purchase_yearmonth").iloc[1:-1])
    growth = 0.0
    if len(monthly) >= 2:
        prev = monthly["total_price"].iloc[-2]
        last = monthly["total_price"].iloc[-1]
        growth = (last/prev - 1)*100 if prev > 0 else 0

    top_cats = (delivered.groupby("category_en")["total_price"].sum()
                .nlargest(5).reset_index())
    seg_summary = (rfm.groupby("segment")
        .agg(nb=("customer_unique_id","count"),rev=("monetary","sum"))
        .reset_index().sort_values("rev",ascending=False))

    return {
        "total_revenue"  : delivered["total_price"].sum(),
        "total_orders"   : delivered["order_id"].nunique(),
        "total_customers": master["customer_unique_id"].nunique(),
        "total_sellers"  : master["seller_id"].nunique(),
        "avg_ticket"     : delivered["total_price"].mean(),
        "avg_score"      : delivered["review_score"].mean(),
        "late_rate"      : delivered["is_late"].mean()*100,
        "repeat_rate"    : (rfm["frequency"]>=2).mean()*100,
        "growth_last"    : growth,
        "note_temps"     : delivered.loc[delivered["is_late"]==0,"review_score"].mean(),
        "note_retard"    : delivered.loc[delivered["is_late"]==1,"review_score"].mean(),
        "top_cats"       : top_cats,
        "seg_summary"    : seg_summary,
        "monthly"        : monthly,
    }

def build_context(kpis):
    top_str = "\n".join(f"  - {r['category_en']}: R$ {r['total_price']:,.0f}"
                        for _,r in kpis["top_cats"].iterrows())
    seg_str = "\n".join(f"  - {r['segment']}: {r['nb']:,} clients, R$ {r['rev']:,.0f}"
                        for _,r in kpis["seg_summary"].iterrows())
    return f"""
=== OLIST BI COPILOT — DONNEES TEMPS REEL ===
CA total        : R$ {kpis['total_revenue']:,.0f}
Commandes       : {kpis['total_orders']:,}
Clients uniques : {kpis['total_customers']:,}
Vendeurs actifs : {kpis['total_sellers']:,}
Panier moyen    : R$ {kpis['avg_ticket']:.0f}
Note moyenne    : {kpis['avg_score']:.2f}/5
Taux retard     : {kpis['late_rate']:.1f}%
Taux reachat    : {kpis['repeat_rate']:.1f}%
Croissance      : {kpis['growth_last']:+.1f}%
Note si a temps : {kpis['note_temps']:.2f} vs retard : {kpis['note_retard']:.2f}
TOP 5 CATEGORIES:\n{top_str}
SEGMENTS RFM:\n{seg_str}
"""

# ════════════════════════════════════════════════════════════
# GEMINI
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def call_gemini_cached(api_key, prompt):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Erreur Gemini : {e}"

def call_gemini(api_key, system_prompt, messages):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash",
                                      system_instruction=system_prompt)
        history = [{"role":"user" if m["role"]=="user" else "model",
                    "parts":[m["content"]]} for m in messages[:-1]]
        chat = model.start_chat(history=history)
        return chat.send_message(messages[-1]["content"]).text
    except Exception as e:
        return f"Erreur Gemini : {e}"

# ════════════════════════════════════════════════════════════
# DETECTION ANOMALIES
# ════════════════════════════════════════════════════════════
def detect_anomalies(kpis, master, delivered, rfm):
    anomalies, recommandations = [], []

    # Taux de retard
    lr = kpis["late_rate"]
    if lr > 15:
        anomalies.append({"niveau":"CRITIQUE","icone":"🔴","titre":"Taux de retard très élevé",
            "detail":f"{lr:.1f}% de commandes en retard","impact":"Fort impact satisfaction"})
        recommandations.append({"prio":1,"action":"Auditer les vendeurs avec >20% retard",
            "detail":"Identifier les transporteurs défaillants et renégocier"})
    elif lr > 10:
        anomalies.append({"niveau":"ALERTE","icone":"🟠","titre":"Taux de retard élevé",
            "detail":f"{lr:.1f}% de retard","impact":"Impact modéré"})
        recommandations.append({"prio":2,"action":"Surveiller les vendeurs à risque",
            "detail":"Alertes automatiques pour vendeurs >10% retard"})

    # Note moyenne
    sc = kpis["avg_score"]
    if sc < 3.5:
        anomalies.append({"niveau":"CRITIQUE","icone":"🔴","titre":"Satisfaction critique",
            "detail":f"Note {sc:.2f}/5","impact":"Risque perte massive clients"})
        recommandations.append({"prio":1,"action":"Enquête satisfaction immédiate",
            "detail":"Contacter clients note ≤2 avec compensation"})
    elif sc < 4.0:
        anomalies.append({"niveau":"ALERTE","icone":"🟠","titre":"Satisfaction à améliorer",
            "detail":f"Note {sc:.2f}/5","impact":"Risque churn progressif"})
        recommandations.append({"prio":2,"action":"Analyser avis négatifs par catégorie",
            "detail":"Contacter vendeurs avec mauvaises notes"})

    # Réachat
    rr = kpis["repeat_rate"]
    if rr < 5:
        anomalies.append({"niveau":"CRITIQUE","icone":"🔴","titre":"Fidélisation très faible",
            "detail":f"{rr:.1f}% de réachat","impact":"CAC élevé sans rétention"})
        recommandations.append({"prio":1,"action":"Programme de fidélité urgent",
            "detail":"Coupons pour clients inactifs >90 jours"})
    elif rr < 10:
        anomalies.append({"niveau":"ALERTE","icone":"🟠","titre":"Faible fidélisation",
            "detail":f"{rr:.1f}% de réachat","impact":"Rentabilité réduite"})
        recommandations.append({"prio":2,"action":"Campagne email réactivation",
            "detail":"Cibler les hibernants avec offre personnalisée"})

    # Croissance
    gr = kpis["growth_last"]
    if gr < -10:
        anomalies.append({"niveau":"CRITIQUE","icone":"🔴","titre":"Chute du CA",
            "detail":f"CA -{abs(gr):.1f}% vs mois précédent","impact":"Tendance préoccupante"})
        recommandations.append({"prio":1,"action":"Analyse causes de la baisse",
            "detail":"Vérifier catégorie/région/saisonnalité"})
    elif gr > 20:
        anomalies.append({"niveau":"BON","icone":"🟢","titre":"Forte croissance",
            "detail":f"CA +{gr:.1f}%","impact":"Tendance très positive"})
        recommandations.append({"prio":3,"action":"Capitaliser sur la croissance",
            "detail":"Augmenter stocks des catégories qui progressent"})

    # Churn
    churn_rate = (rfm["recency"]>180).mean()*100
    if churn_rate > 50:
        anomalies.append({"niveau":"ALERTE","icone":"🟠","titre":"Fort risque churn",
            "detail":f"{churn_rate:.1f}% clients inactifs >180j","impact":"Perte revenus potentielle"})
        recommandations.append({"prio":2,"action":"Campagne réactivation",
            "detail":f"Contacter {rfm[rfm['recency']>180].shape[0]:,} clients inactifs"})

    return anomalies, sorted(recommandations, key=lambda x: x["prio"])

# ════════════════════════════════════════════════════════════
# GENERATION PDF
# ════════════════════════════════════════════════════════════
def generate_pdf(kpis, anomalies, recommandations, ia_summary=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # En-tête
    pdf.set_fill_color(13, 17, 23)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(29, 158, 117)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 10)
    pdf.cell(0, 10, "Olist BI Copilot — Rapport Executif", ln=True)
    pdf.set_text_color(139, 148, 158)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Genere le {datetime.datetime.now().strftime('%d/%m/%Y a %H:%M')}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    # KPIs
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(29, 158, 117)
    pdf.cell(0, 8, "INDICATEURS CLES", ln=True)
    pdf.ln(3)

    kpi_data = [
        ("Chiffre d'affaires total", f"R$ {kpis['total_revenue']:,.0f}"),
        ("Commandes livrees",         f"{kpis['total_orders']:,}"),
        ("Clients uniques",           f"{kpis['total_customers']:,}"),
        ("Panier moyen",              f"R$ {kpis['avg_ticket']:.0f}"),
        ("Note moyenne clients",      f"{kpis['avg_score']:.2f} / 5"),
        ("Taux de retard",            f"{kpis['late_rate']:.1f}%"),
        ("Taux de reachat",           f"{kpis['repeat_rate']:.1f}%"),
        ("Croissance dernier mois",   f"{kpis['growth_last']:+.1f}%"),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    for label, value in kpi_data:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(100, 7, label, border="B")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, value, border="B", ln=True)
    pdf.ln(8)

    # Anomalies
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(216, 90, 48)
    pdf.cell(0, 8, "ANOMALIES DETECTEES", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for a in anomalies:
        niveau_colors = {"CRITIQUE":(216,90,48),"ALERTE":(186,117,23),"BON":(29,158,117)}
        c = niveau_colors.get(a["niveau"],(100,100,100))
        pdf.set_text_color(*c)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"[{a['niveau']}] {a['titre']}", ln=True)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"  {a['detail']} — Impact: {a['impact']}", ln=True)
        pdf.ln(2)
    pdf.ln(5)

    # Recommandations
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(29, 158, 117)
    pdf.cell(0, 8, "RECOMMANDATIONS PRIORITAIRES", ln=True)
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)
    for i, r in enumerate(recommandations, 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"#{i} — {r['action']}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, f"  {r['detail']}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
    pdf.ln(5)

    # Analyse IA
    if ia_summary:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(83, 74, 183)
        pdf.cell(0, 8, "ANALYSE IA — OLIST BI COPILOT", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        # Nettoyer le texte pour le PDF
        clean = ia_summary.replace("**","").replace("##","").replace("#","").replace("*","")
        pdf.multi_cell(0, 5, clean[:3000])

    # Top catégories
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(29, 158, 117)
    pdf.cell(0, 8, "TOP 5 CATEGORIES — CA", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for _, row in kpis["top_cats"].iterrows():
        pdf.cell(120, 7, str(row["category_en"]), border="B")
        pdf.cell(0, 7, f"R$ {row['total_price']:,.0f}", border="B", ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(0, 5, "Rapport genere automatiquement par Olist BI Copilot — Powered by Gemini AI", ln=True)

    return pdf.output()

# ════════════════════════════════════════════════════════════
# PREVISIONS ML (Prophet)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def run_forecast(path, n_months=6):
    try:
        from prophet import Prophet
        master, _, _ = load_data(path)
        ts = (master.groupby("purchase_yearmonth")["total_price"].sum()
              .reset_index().sort_values("purchase_yearmonth").iloc[1:-1])
        ts["ds"] = pd.to_datetime(ts["purchase_yearmonth"] + "-01")
        ts["y"]  = ts["total_price"]

        m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.3)
        m.fit(ts[["ds","y"]])
        future   = m.make_future_dataframe(periods=n_months, freq="MS")
        forecast = m.predict(future)
        return ts, forecast
    except Exception as e:
        return None, str(e)

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 Olist BI Copilot")
    st.markdown("---")
    DATA_PATH = st.text_input("📁 Chemin données",
        value="C:/Users/HP/Downloads/olist_dashbord1/")
    st.markdown("---")
    GEMINI_KEY = st.text_input("🔑 Clé API Gemini (gratuite)",
        type="password", placeholder="AIza...",
        help="Clé gratuite sur aistudio.google.com")
    if not GEMINI_KEY:
        st.markdown("<small style='color:#BA7517'>👉 [Clé gratuite](https://aistudio.google.com/app/apikey)</small>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#1D9E75'>✅ Clé configurée</small>",
                    unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Accueil & Résumé",
        "📊 Vue générale",
        "🗺️ Géographie",
        "🚚 Logistique",
        "👥 Clients RFM",
        "🏪 Vendeurs",
        "🔍 Anomalies & Recommandations",
        "📈 Comparaison périodes",
        "🔮 Prévisions ML",
        "📄 Rapport PDF",
        "🤖 Assistant IA",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='color:#8B949E'>Olist BI Copilot v3.0<br>Toutes les étapes intégrées</small>",
                unsafe_allow_html=True)

# ── Chargement ───────────────────────────────────────────────
try:
    master, delivered, rfm = load_data(DATA_PATH)
    kpis = compute_kpis(hash(DATA_PATH), hash(DATA_PATH), hash(DATA_PATH), DATA_PATH)
except Exception as e:
    st.error(f"❌ Impossible de charger les données depuis `{DATA_PATH}`\n\n{e}")
    st.info("👉 Vérifie que `olist_master.parquet` et `olist_rfm.parquet` sont dans ce dossier.")
    st.stop()

# ════════════════════════════════════════════════════════════
# PAGE 0 — ACCUEIL & RESUME
# ════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Résumé":
    st.markdown("## 📢 Résumé du jour — Olist BI Copilot")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔔 Alertes automatiques")
        checks = [
            (kpis["late_rate"] <= 10,   f"🟢 Livraisons OK : {kpis['late_rate']:.1f}% retard",
                                         f"🔴 Retards élevés : {kpis['late_rate']:.1f}%"),
            (kpis["avg_score"] >= 4.0,  f"🟢 Satisfaction bonne : {kpis['avg_score']:.2f}/5",
                                         f"🟠 Satisfaction : {kpis['avg_score']:.2f}/5"),
            (kpis["growth_last"] >= 0,  f"🟢 CA en hausse : {kpis['growth_last']:+.1f}%",
                                         f"🔴 CA en baisse : {kpis['growth_last']:+.1f}%"),
            (kpis["repeat_rate"] >= 10, f"🟢 Fidélisation : {kpis['repeat_rate']:.1f}%",
                                         f"🟠 Faible fidélisation : {kpis['repeat_rate']:.1f}%"),
        ]
        for ok, msg_ok, msg_ko in checks:
            css = "alert-green" if ok else "alert-orange"
            msg = msg_ok if ok else msg_ko
            st.markdown(f'<div class="{css}"><b>{msg}</b></div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### 📊 KPIs instantanés")
        for label, val in [
            ("💰 CA Total",    f"R$ {kpis['total_revenue']/1e6:.2f}M"),
            ("📦 Commandes",   f"{kpis['total_orders']:,}"),
            ("👤 Clients",     f"{kpis['total_customers']:,}"),
            ("🛒 Panier moy.", f"R$ {kpis['avg_ticket']:.0f}"),
            ("⭐ Note moy.",   f"{kpis['avg_score']:.2f}/5"),
            ("📈 Croissance",  f"{kpis['growth_last']:+.1f}%"),
        ]:
            st.markdown(f"**{label}** &nbsp;&nbsp; `{val}`")

    st.markdown("---")
    st.markdown("#### 🤖 Analyse IA automatique")
    if not GEMINI_KEY:
        st.info("Entre ta clé API Gemini dans la sidebar.")
    else:
        if "auto_summary" not in st.session_state:
            with st.spinner("Analyse en cours..."):
                ctx = build_context(kpis)
                prompt = f"""Tu es Olist BI Copilot. Données : {ctx}
Génère un résumé exécutif :
1. 📊 SYNTHESE (3 phrases)
2. ✅ POINTS FORTS (2-3 avec chiffres)
3. ⚠️ POINTS A AMELIORER (2-3 avec chiffres)
4. 💡 RECOMMANDATION PRIORITAIRE (1 action concrète)
Réponds en français."""
                st.session_state["auto_summary"] = call_gemini_cached(GEMINI_KEY, prompt)
        st.markdown(st.session_state["auto_summary"])
        if st.button("🔄 Rafraîchir"):
            del st.session_state["auto_summary"]
            st.rerun()

# ════════════════════════════════════════════════════════════
# PAGE 1 — VUE GENERALE
# ════════════════════════════════════════════════════════════
elif page == "📊 Vue générale":
    st.markdown("## 📊 Vue générale")
    f1,f2,f3 = st.columns(3)
    with f1:
        yrs = sorted(master["purchase_year"].dropna().unique().astype(int))
        sel_y = st.multiselect("Année", yrs, default=yrs)
    with f2:
        sts = sorted(master["customer_state"].dropna().unique())
        sel_s = st.multiselect("État", sts, default=sts)
    with f3:
        cats = sorted(master["category_en"].dropna().unique())
        sel_c = st.multiselect("Catégorie", cats, default=cats)

    df = master[master["purchase_year"].isin(sel_y) &
                master["customer_state"].isin(sel_s) &
                master["category_en"].isin(sel_c)]
    dfd = delivered[delivered["purchase_year"].isin(sel_y) &
                    delivered["customer_state"].isin(sel_s) &
                    delivered["category_en"].isin(sel_c)]
    st.markdown("---")

    k1,k2,k3,k4 = st.columns(4)
    for col,lbl,val,sub,clr in [
        (k1,"CA","f\"R$ {dfd['total_price'].sum()/1e6:.2f}M\"",
             "f\"Panier: R$ {dfd['total_price'].mean():.0f}\"",TEAL),
        (k2,"Commandes","f\"{dfd['order_id'].nunique():,}\"",
             "f\"{df['customer_unique_id'].nunique():,} clients\"",PURPLE),
        (k3,"Note moy.","f\"{dfd['review_score'].mean():.2f}/5\"",
             "f\"Retards: {dfd['is_late'].mean()*100:.1f}%\"",AMBER),
        (k4,"Vendeurs","f\"{df['seller_id'].nunique():,}\"",
             "f\"{df['category_en'].nunique()} catégories\"",CORAL),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value" style="color:{clr}">{eval(val)}</div>
            <div class="kpi-sub">{eval(sub)}</div>
        </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns([2,1])
    with c1:
        mf = (df.groupby("purchase_yearmonth").agg(revenue=("total_price","sum"))
              .reset_index().sort_values("purchase_yearmonth").iloc[1:-1])
        mf["ds"] = pd.to_datetime(mf["purchase_yearmonth"]+"-01")
        fig = go.Figure(go.Scatter(x=mf["ds"],y=mf["revenue"],fill="tozeroy",
            fillcolor="rgba(29,158,117,0.12)",line=dict(color=TEAL,width=2.5),
            mode="lines+markers",marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>R$ %{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Evolution CA mensuel",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=300,
            xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        pay = df["payment_type"].value_counts().reset_index()
        pay.columns = ["type","count"]
        pay = pay[pay["type"]!="not_defined"]
        pay["label"] = pay["type"].map({"credit_card":"Carte","boleto":"Boleto",
                                         "voucher":"Voucher","debit_card":"Débit"}).fillna(pay["type"])
        fig2 = go.Figure(go.Pie(labels=pay["label"],values=pay["count"],hole=0.5,
            marker_colors=[TEAL,PURPLE,CORAL,AMBER],textfont=dict(size=10)))
        fig2.update_layout(title="Paiements",paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"),height=300,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2,use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        cat = (dfd.groupby("category_en").agg(revenue=("total_price","sum"))
               .reset_index().dropna().nlargest(10,"revenue").sort_values("revenue"))
        fig3 = go.Figure(go.Bar(y=cat["category_en"],x=cat["revenue"],orientation="h",
            marker=dict(color=cat["revenue"],colorscale="Teal",showscale=False),
            hovertemplate="%{y}<br>R$ %{x:,.0f}<extra></extra>"))
        fig3.update_layout(title="Top 10 catégories",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=350,
            xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False),
            margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        hr = df.groupby("purchase_hour")["order_id"].nunique().reset_index()
        hr.columns = ["hour","nb"]
        fig4 = go.Figure(go.Bar(x=hr["hour"],y=hr["nb"],
            marker=dict(color=hr["nb"],colorscale="Purples",showscale=False),
            hovertemplate="%{x}h → %{y:,}<extra></extra>"))
        fig4.update_layout(title="Commandes par heure",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=350,
            xaxis=dict(showgrid=False,dtick=2),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig4,use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIE
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Géographie":
    st.markdown("## 🗺️ Géographie des ventes")
    geo = (delivered.groupby("customer_state")
           .agg(nb_orders=("order_id","nunique"),revenue=("total_price","sum"),
                avg_score=("review_score","mean"),late_rate=("is_late","mean"))
           .reset_index())
    geo["late_pct"] = geo["late_rate"]*100
    metric = st.selectbox("Métrique",["nb_orders","revenue","avg_score","late_pct"],
        format_func=lambda x:{"nb_orders":"Commandes","revenue":"CA","avg_score":"Note","late_pct":"Retard%"}[x])
    c1,c2 = st.columns([3,2])
    with c1:
        top10 = geo.sort_values(metric,ascending=False).head(10).sort_values(metric)
        fig = go.Figure(go.Bar(y=top10["customer_state"],x=top10[metric],orientation="h",
            marker_color={"nb_orders":BLUE,"revenue":TEAL,"avg_score":AMBER,"late_pct":CORAL}[metric],opacity=0.85))
        fig.update_layout(title="Top 10 états",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=400,
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=False),margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.dataframe(geo.sort_values(metric,ascending=False)
            [["customer_state","nb_orders","revenue","avg_score","late_pct"]]
            .rename(columns={"customer_state":"État","nb_orders":"Cmds","revenue":"CA",
                             "avg_score":"Note","late_pct":"Retard%"})
            .round(2).reset_index(drop=True),height=400,use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — LOGISTIQUE
# ════════════════════════════════════════════════════════════
elif page == "🚚 Logistique":
    st.markdown("## 🚚 Logistique & Satisfaction")
    k1,k2,k3,k4 = st.columns(4)
    for col,lbl,val,sub,clr in [
        (k1,"Délai médian réel",f"{delivered['days_to_deliver_actual'].median():.0f}j","",TEAL),
        (k2,"Délai médian estimé",f"{delivered['days_to_deliver_estimated'].median():.0f}j","",PURPLE),
        (k3,"Taux retard",f"{delivered['is_late'].mean()*100:.1f}%",
         f"Moy: {delivered.loc[delivered['is_late']==1,'delay_days'].mean():.1f}j",CORAL),
        (k4,"Note globale",f"{delivered['review_score'].mean():.2f}/5","",AMBER),
    ]:
        col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{lbl}</div>
            <div class="kpi-value" style="color:{clr}">{val}</div>
            <div class="kpi-sub">{sub}</div></div>""",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        ds = (delivered.groupby("delay_bucket",observed=True)
              .agg(avg_score=("review_score","mean"),nb=("order_id","count")).reset_index())
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=ds["delay_bucket"].astype(str),y=ds["nb"],name="Commandes",
            marker_color=[TEAL,TEAL,TEAL,AMBER,CORAL,CORAL,"#A83010"],opacity=0.65),secondary_y=False)
        fig.add_trace(go.Scatter(x=ds["delay_bucket"].astype(str),y=ds["avg_score"],name="Note",
            mode="lines+markers",line=dict(color=PURPLE,width=2.5),marker=dict(size=8)),secondary_y=True)
        fig.update_layout(title="Note vs retard",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=350,
            margin=dict(l=0,r=30,t=40,b=0))
        fig.update_yaxes(range=[1,5.5],secondary_y=True,showgrid=False,color=PURPLE)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=delivered["days_to_deliver_actual"].clip(upper=60),
            name="Réel",nbinsx=40,opacity=0.7,marker_color=TEAL))
        fig2.add_trace(go.Histogram(x=delivered["days_to_deliver_estimated"].clip(upper=60),
            name="Estimé",nbinsx=40,opacity=0.5,marker_color=PURPLE))
        fig2.update_layout(barmode="overlay",title="Délai réel vs estimé",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"),height=350,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2,use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — CLIENTS RFM
# ════════════════════════════════════════════════════════════
elif page == "👥 Clients RFM":
    st.markdown("## 👥 Segmentation clients RFM")
    seg = (rfm.groupby("segment")
           .agg(nb=("customer_unique_id","count"),rec=("recency","mean"),
                freq=("frequency","mean"),mon=("monetary","mean"),rev=("monetary","sum"))
           .reset_index().sort_values("rev",ascending=False))
    seg["pct_c"] = seg["nb"]/seg["nb"].sum()*100
    seg["pct_r"] = seg["rev"]/seg["rev"].sum()*100
    sc = [TEAL,"#5BAF8E",BLUE,AMBER,CORAL,"#A83010",PURPLE]
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=seg["segment"],values=seg["nb"],hole=0.55,
            marker_colors=sc,textinfo="label+percent",textfont=dict(size=10)))
        fig.update_layout(title="Clients par segment",paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"),height=350,showlegend=False,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(x=seg["segment"],y=seg["rev"]/1000,marker_color=sc,opacity=0.85,
            text=seg["pct_r"].apply(lambda x:f"{x:.0f}%"),textposition="outside",
            textfont=dict(color="#CBD5E0")))
        fig2.update_layout(title="CA par segment",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#CBD5E0"),height=350,
            xaxis=dict(showgrid=False),yaxis=dict(title="K BRL"),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2,use_container_width=True)
    st.dataframe(seg.rename(columns={"segment":"Segment","nb":"Clients","pct_c":"% Clients",
        "rec":"Récence(j)","freq":"Fréq.","mon":"Montant(BRL)","pct_r":"% CA"})
        .round(1),use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 5 — VENDEURS
# ════════════════════════════════════════════════════════════
elif page == "🏪 Vendeurs":
    st.markdown("## 🏪 Performance des vendeurs")
    mo = st.slider("Commandes minimum",10,200,30,step=10)
    sp = (delivered.groupby("seller_id")
          .agg(nb=("order_id","count"),rev=("total_price","sum"),
               score=("review_score","mean"),late=("is_late","mean"))
          .reset_index().query(f"nb>={mo}"))
    sp["late_pct"] = sp["late"]*100
    fig = px.scatter(sp,x="late_pct",y="score",size="nb",color="rev",
        color_continuous_scale="Teal",hover_data={"seller_id":True,"nb":True,"rev":":.0f"},
        labels={"late_pct":"Retard(%)","score":"Note","nb":"Cmds","rev":"CA"},
        title="Vendeurs : Retard vs Note")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"),height=450,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig,use_container_width=True)
    sp["composite"] = (sp["score"]/5)*0.5+(1-sp["late"])*0.3+(sp["rev"]/sp["rev"].max())*0.2
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 🏆 Top 10")
        t = sp.nlargest(10,"composite")[["seller_id","nb","rev","score","late_pct"]].copy()
        t["seller_id"] = t["seller_id"].str[:10]+"..."
        t.columns=["Vendeur","Cmds","CA","Note","Retard%"]
        st.dataframe(t.style.format({"CA":"{:,.0f}","Note":"{:.2f}","Retard%":"{:.1f}"}),
                     use_container_width=True,hide_index=True)
    with c2:
        st.markdown("### 🔴 Flop 10")
        f = sp.nsmallest(10,"composite")[["seller_id","nb","rev","score","late_pct"]].copy()
        f["seller_id"] = f["seller_id"].str[:10]+"..."
        f.columns=["Vendeur","Cmds","CA","Note","Retard%"]
        st.dataframe(f.style.format({"CA":"{:,.0f}","Note":"{:.2f}","Retard%":"{:.1f}"}),
                     use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALIES & RECOMMANDATIONS
# ════════════════════════════════════════════════════════════
elif page == "🔍 Anomalies & Recommandations":
    st.markdown("## 🔍 Détection d'anomalies & Recommandations")
    anomalies, recommandations = detect_anomalies(kpis, master, delivered, rfm)

    critiques = len([a for a in anomalies if a["niveau"]=="CRITIQUE"])
    alertes   = len([a for a in anomalies if a["niveau"]=="ALERTE"])
    bons      = len([a for a in anomalies if a["niveau"]=="BON"])

    k1,k2,k3 = st.columns(3)
    k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Critiques</div>
        <div class="kpi-value" style="color:{CORAL}">{critiques}</div></div>""",unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Alertes</div>
        <div class="kpi-value" style="color:{AMBER}">{alertes}</div></div>""",unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Positifs</div>
        <div class="kpi-value" style="color:{TEAL}">{bons}</div></div>""",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 🚨 Anomalies")
        for a in anomalies:
            css = {"CRITIQUE":"alert-red","ALERTE":"alert-orange","BON":"alert-green"}.get(a["niveau"],"alert-orange")
            st.markdown(f'<div class="{css}"><b>{a["icone"]} {a["titre"]}</b><br>'
                        f'<small>{a["detail"]}</small></div>',unsafe_allow_html=True)
    with c2:
        st.markdown("### 💡 Recommandations")
        for i,r in enumerate(recommandations,1):
            clr = {1:CORAL,2:AMBER,3:TEAL}.get(r["prio"],TEAL)
            st.markdown(f'<div class="summary-box" style="border-left:3px solid {clr};padding-left:16px">'
                        f'<b style="color:{clr}">#{i} {r["action"]}</b><br>'
                        f'<small style="color:#CBD5E0">{r["detail"]}</small></div>',unsafe_allow_html=True)

    if GEMINI_KEY and st.button("🤖 Analyse IA approfondie",type="primary"):
        with st.spinner("Analyse..."):
            txt = "\n".join(f"- {a['icone']} {a['titre']}: {a['detail']}" for a in anomalies)
            prompt = f"""Anomalies Olist détectées:\n{txt}\n
Pour chaque anomalie: 1) Pourquoi c'est un problème 2) Cause probable
3) Plan d'action 3 étapes 4) Impact si rien n'est fait
Termine par la PRIORITE ABSOLUE. Réponds en français."""
            st.markdown(call_gemini_cached(GEMINI_KEY, prompt))

# ════════════════════════════════════════════════════════════
# PAGE 7 — COMPARAISON PERIODES
# ════════════════════════════════════════════════════════════
elif page == "📈 Comparaison périodes":
    st.markdown("## 📈 Comparaison des périodes")

    months = sorted(master["purchase_yearmonth"].dropna().unique())
    c1,c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Période 1", months, index=max(0,len(months)-3))
    with c2:
        p2 = st.selectbox("Période 2", months, index=max(0,len(months)-2))

    def get_period_stats(period):
        df_p = delivered[delivered["purchase_yearmonth"]==period]
        return {
            "CA"       : df_p["total_price"].sum(),
            "Commandes": df_p["order_id"].nunique(),
            "Note"     : df_p["review_score"].mean(),
            "Retard%"  : df_p["is_late"].mean()*100,
            "Panier"   : df_p["total_price"].mean(),
        }

    s1, s2 = get_period_stats(p1), get_period_stats(p2)

    st.markdown(f"### {p1} vs {p2}")
    cols = st.columns(5)
    metrics = ["CA","Commandes","Note","Retard%","Panier"]
    for col, m in zip(cols, metrics):
        v1, v2 = s1[m], s2[m]
        delta = ((v2/v1)-1)*100 if v1 > 0 else 0
        arrow = "🟢 +" if delta > 0 else "🔴 "
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{m}</div>
            <div class="kpi-value" style="color:{TEAL}">{v2:,.1f}</div>
            <div class="kpi-sub">{arrow}{delta:.1f}% vs {p1}</div>
        </div>""", unsafe_allow_html=True)

    # Graphique comparatif
    fig = go.Figure()
    fig.add_trace(go.Bar(name=p1, x=metrics,
        y=[s1[m] for m in metrics], marker_color=TEAL, opacity=0.7))
    fig.add_trace(go.Bar(name=p2, x=metrics,
        y=[s2[m] for m in metrics], marker_color=PURPLE, opacity=0.7))
    fig.update_layout(barmode="group",title=f"Comparaison {p1} vs {p2}",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"),height=400,
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(font=dict(color="#CBD5E0")),margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig,use_container_width=True)

    # Evolution mensuelle complète
    st.markdown("### 📊 Evolution mensuelle complète")
    monthly_full = (delivered.groupby("purchase_yearmonth")
        .agg(CA=("total_price","sum"),Commandes=("order_id","nunique"),
             Note=("review_score","mean"),Retard=("is_late","mean"))
        .reset_index().sort_values("purchase_yearmonth"))
    monthly_full["ds"] = pd.to_datetime(monthly_full["purchase_yearmonth"]+"-01")
    monthly_full["Retard%"] = monthly_full["Retard"]*100

    metric_comp = st.selectbox("Métrique à afficher",["CA","Commandes","Note","Retard%"])
    fig2 = go.Figure(go.Scatter(x=monthly_full["ds"],y=monthly_full[metric_comp],
        fill="tozeroy",fillcolor=f"rgba(83,74,183,0.12)",
        line=dict(color=PURPLE,width=2.5),mode="lines+markers",marker=dict(size=5)))
    fig2.update_layout(title=f"Evolution {metric_comp} par mois",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"),height=350,
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2,use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 8 — PREVISIONS ML
# ════════════════════════════════════════════════════════════
elif page == "🔮 Prévisions ML":
    st.markdown("## 🔮 Prévisions des ventes (Prophet)")

    n_months = st.slider("Mois à prévoir",3,12,6)

    with st.spinner("Calcul des prévisions..."):
        ts, forecast = run_forecast(DATA_PATH, n_months)

    if ts is None:
        st.error(f"Erreur Prophet : {forecast}")
        st.info("Installe Prophet : `pip install prophet`")
    else:
        # KPIs prévisions
        future_preds = forecast[forecast["ds"] > ts["ds"].max()]
        avg_pred = future_preds["yhat"].mean()
        last_real = ts["y"].iloc[-1]
        growth_pred = (avg_pred/last_real-1)*100 if last_real > 0 else 0

        k1,k2,k3 = st.columns(3)
        k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">CA moyen prévu/mois</div>
            <div class="kpi-value" style="color:{TEAL}">R$ {avg_pred/1000:.0f}K</div></div>""",
            unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Dernier mois réel</div>
            <div class="kpi-value" style="color:{PURPLE}">R$ {last_real/1000:.0f}K</div></div>""",
            unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Evolution prévue</div>
            <div class="kpi-value" style="color:{TEAL if growth_pred>0 else CORAL}">{growth_pred:+.1f}%</div></div>""",
            unsafe_allow_html=True)

        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["ds"],y=ts["y"],mode="lines+markers",
            name="CA réel",line=dict(color=TEAL,width=2.5),marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=forecast["ds"],y=forecast["yhat"],
            mode="lines",name="Prévision",line=dict(color=PURPLE,width=2,dash="dot")))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"],forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"],forecast["yhat_lower"][::-1]]),
            fill="toself",fillcolor="rgba(83,74,183,0.12)",
            line=dict(color="rgba(255,255,255,0)"),name="Intervalle 80%"))
        fig.add_vline(x=ts["ds"].max().timestamp()*1000,
            line_width=2,line_dash="dash",line_color=CORAL)
        fig.update_layout(title=f"Prévision CA — {n_months} prochains mois",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"),height=450,
            xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(255,255,255,0.06)",tickformat=",.0f"),
            legend=dict(font=dict(color="#CBD5E0")),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)

        # Tableau prévisions
        st.markdown("### 📋 Détail des prévisions")
        pred_table = future_preds[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        pred_table.columns = ["Mois","CA prévu (BRL)","Borne basse","Borne haute"]
        pred_table["Mois"] = pred_table["Mois"].dt.strftime("%B %Y")
        st.dataframe(pred_table.style.format(
            {"CA prévu (BRL)":"{:,.0f}","Borne basse":"{:,.0f}","Borne haute":"{:,.0f}"}),
            use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 9 — RAPPORT PDF
# ════════════════════════════════════════════════════════════
elif page == "📄 Rapport PDF":
    st.markdown("## 📄 Génération de rapport PDF")
    st.markdown("Génère un rapport complet avec KPIs, anomalies, recommandations et analyse IA.")

    include_ia = st.checkbox("Inclure l'analyse IA (nécessite la clé Gemini)", value=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Le rapport contiendra :**")
        st.markdown("- ✅ Résumé exécutif\n- ✅ KPIs clés\n- ✅ Anomalies détectées\n- ✅ Recommandations prioritaires\n- ✅ Top 5 catégories")
        if include_ia:
            st.markdown("- ✅ Analyse IA Gemini")

    with col2:
        anomalies, recommandations = detect_anomalies(kpis, master, delivered, rfm)

        if st.button("📥 Générer et télécharger le PDF", type="primary"):
            with st.spinner("Génération du rapport..."):
                ia_text = ""
                if include_ia and GEMINI_KEY:
                    ctx = build_context(kpis)
                    prompt_pdf = f"""Données Olist: {ctx}
Génère un rapport exécutif complet avec:
1. Synthèse business (5 phrases)
2. Points forts (3 points avec chiffres)
3. Points à améliorer (3 points)
4. Stratégie recommandée (5 actions)
5. Conclusion
Réponds en français, sans markdown, texte simple."""
                    ia_text = call_gemini_cached(GEMINI_KEY, prompt_pdf)

                try:
                    pdf_bytes = generate_pdf(kpis, anomalies, recommandations, ia_text)
                    filename = f"olist_rapport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="📥 Télécharger le rapport PDF",
                        data=bytes(pdf_bytes),
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success(f"✅ Rapport généré : {filename}")
                except Exception as e:
                    st.error(f"Erreur PDF : {e}")
                    st.info("Installe fpdf2 : `pip install fpdf2`")

# ════════════════════════════════════════════════════════════
# PAGE 10 — ASSISTANT IA
# ════════════════════════════════════════════════════════════
elif page == "🤖 Assistant IA":
    st.markdown("## 🤖 Olist BI Copilot — Assistant IA")
    st.markdown("Pose n'importe quelle question en **langage naturel**. L'assistant connaît toutes tes données.")

    if not GEMINI_KEY:
        st.warning("Entre ta clé API Gemini dans la sidebar.")
        st.markdown("**Clé gratuite** → [aistudio.google.com](https://aistudio.google.com/app/apikey)")
    else:
        ctx = build_context(kpis)
        system_prompt = f"""Tu es Olist BI Copilot, expert en data analytics et stratégie e-commerce.
Données temps réel : {ctx}
Règles :
- Réponds toujours en français
- Utilise les vrais chiffres des données
- Structure avec titres et emojis
- Recommandations concrètes et actionnables
- Garde la mémoire de toute la conversation
- Comprends le langage naturel (ex: "et les clients?" = suite du contexte précédent)
- Tu peux faire des analyses SWOT, PESTEL, comparaisons, stratégies marketing
"""
        # Suggestions
        st.markdown("**💬 Questions rapides :**")
        c1,c2,c3,c4 = st.columns(4)
        suggestions = {
            c1:"Analyse globale et points critiques",
            c2:"Fais une analyse SWOT de mon business",
            c3:"Quels clients risquent de partir ?",
            c4:"Stratégie marketing pour augmenter le CA",
        }
        for col,text in suggestions.items():
            if col.button(text,use_container_width=True):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role":"user","content":text})
                st.rerun()

        st.markdown("---")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ex: Pourquoi mes ventes baissent ? Quels vendeurs accompagner ? Fais une analyse PESTEL...")
        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Olist BI Copilot analyse..."):
                    answer = call_gemini(GEMINI_KEY, system_prompt, st.session_state.messages)
                    st.markdown(answer)
                    st.session_state.messages.append({"role":"assistant","content":answer})

        if st.session_state.get("messages"):
            if st.button("🗑️ Effacer la conversation"):
                st.session_state.messages = []
                st.rerun()
