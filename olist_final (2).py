"""
Olist BI Copilot — Dashboard Streamlit + Assistant IA (Gemini gratuit)
Lancer : streamlit run olist_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# ── Config page ─────────────────────────────────────────────
st.set_page_config(
    page_title="Olist BI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0D1117; }
[data-testid="stSidebar"]          { background: #161B22; border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }
h1, h2, h3 { color: #E6EDF3 !important; }
.block-container { padding: 1.5rem 2rem; }
.kpi-card {
    background: #1C2333;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 4px;
}
.kpi-label { font-size: 11px; color: #8B949E; font-weight: 600;
             letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; font-family: monospace; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #8B949E; margin-top: 5px; }
.alert-red  { background:#2D1515; border:1px solid #D85A30; border-radius:10px; padding:12px 16px; margin:6px 0; }
.alert-orange { background:#2D1F10; border:1px solid #BA7517; border-radius:10px; padding:12px 16px; margin:6px 0; }
.alert-green  { background:#0F2018; border:1px solid #1D9E75; border-radius:10px; padding:12px 16px; margin:6px 0; }
.summary-box  { background:#161B22; border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:20px 24px; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des données ───────────────────────────────────
@st.cache_data
def load_data(path):
    master    = pd.read_parquet(f"{path}olist_master.parquet")
    rfm       = pd.read_parquet(f"{path}olist_rfm.parquet")
    delivered = master[master["order_status"] == "delivered"].copy()
    bins   = [-999, -7, -3, 0, 3, 7, 14, 999]
    labels = ["< -7j","-7 a -3j","-3 a 0j","0 a +3j","+3 a +7j","+7 a +14j","> +14j"]
    delivered["delay_bucket"] = pd.cut(delivered["delay_days"], bins=bins, labels=labels)
    return master, delivered, rfm

# ── Calcul KPIs globaux ──────────────────────────────────────
def compute_kpis(master, delivered, rfm):
    monthly = (master
        .groupby("purchase_yearmonth")["total_price"].sum()
        .reset_index().sort_values("purchase_yearmonth").iloc[1:-1]
    )
    growth = 0.0
    if len(monthly) >= 2:
        growth = (monthly["total_price"].iloc[-1] / monthly["total_price"].iloc[-2] - 1) * 100

    top_cats = (delivered.groupby("category_en")["total_price"].sum()
                .nlargest(5).reset_index())
    top_cats_str = "\n".join(
        f"  - {r['category_en']} : R$ {r['total_price']:,.0f}"
        for _, r in top_cats.iterrows())

    seg_summary = (rfm.groupby("segment")
        .agg(nb=("customer_unique_id","count"), rev=("monetary","sum"))
        .reset_index().sort_values("rev", ascending=False))
    seg_str = "\n".join(
        f"  - {r['segment']} : {r['nb']:,} clients, R$ {r['rev']:,.0f}"
        for _, r in seg_summary.iterrows())

    note_temps  = delivered.loc[delivered["is_late"]==0,"review_score"].mean()
    note_retard = delivered.loc[delivered["is_late"]==1,"review_score"].mean()

    return {
        "total_revenue"  : delivered["total_price"].sum(),
        "total_orders"   : delivered["order_id"].nunique(),
        "total_customers": master["customer_unique_id"].nunique(),
        "total_sellers"  : master["seller_id"].nunique(),
        "avg_ticket"     : delivered["total_price"].mean(),
        "avg_score"      : delivered["review_score"].mean(),
        "late_rate"      : delivered["is_late"].mean() * 100,
        "repeat_rate"    : (rfm["frequency"] >= 2).mean() * 100,
        "growth_last"    : growth,
        "note_temps"     : note_temps,
        "note_retard"    : note_retard,
        "top_cats_str"   : top_cats_str,
        "seg_str"        : seg_str,
        "monthly"        : monthly,
    }

def build_context(kpis):
    return f"""
=== OLIST BI COPILOT — DONNEES EN TEMPS REEL ===

INDICATEURS CLES :
- CA total              : R$ {kpis['total_revenue']:,.0f}
- Commandes livrees     : {kpis['total_orders']:,}
- Clients uniques       : {kpis['total_customers']:,}
- Vendeurs actifs       : {kpis['total_sellers']:,}
- Panier moyen          : R$ {kpis['avg_ticket']:.0f}
- Note moyenne          : {kpis['avg_score']:.2f} / 5
- Taux de retard        : {kpis['late_rate']:.1f}%
- Taux de reachat       : {kpis['repeat_rate']:.1f}%
- Croissance dernier mois: {kpis['growth_last']:+.1f}%
- Note si livre a temps : {kpis['note_temps']:.2f} vs en retard : {kpis['note_retard']:.2f}

TOP 5 CATEGORIES (CA) :
{kpis['top_cats_str']}

SEGMENTS RFM :
{kpis['seg_str']}
"""

# ── Appel Gemini avec cache (1h) ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def call_gemini_cached(api_key, prompt):
    """Appel caché — pour résumé auto et anomalies (évite de répéter l'appel API)."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur Gemini : {e}"

def call_gemini(api_key, system_prompt, messages):
    """Appel conversationnel avec historique — pour l'assistant IA."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        chat = model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        return f"Erreur Gemini : {e}"

# ── Résumé automatique à l'ouverture ────────────────────────
def render_auto_summary(kpis, api_key):
    st.markdown("## 📢 Résumé du jour — Olist BI Copilot")

    # Alertes automatiques sans IA
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔔 Alertes automatiques")
        if kpis["late_rate"] > 10:
            st.markdown(f'<div class="alert-red">🔴 <b>Taux de retard élevé</b> : {kpis["late_rate"]:.1f}% des commandes en retard</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">🟢 <b>Livraisons OK</b> : seulement {kpis["late_rate"]:.1f}% de retard</div>', unsafe_allow_html=True)

        if kpis["avg_score"] < 4.0:
            st.markdown(f'<div class="alert-orange">🟠 <b>Satisfaction à surveiller</b> : note moyenne {kpis["avg_score"]:.2f}/5</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">🟢 <b>Satisfaction bonne</b> : {kpis["avg_score"]:.2f}/5</div>', unsafe_allow_html=True)

        if kpis["growth_last"] < 0:
            st.markdown(f'<div class="alert-red">🔴 <b>CA en baisse</b> : {kpis["growth_last"]:+.1f}% vs mois précédent</div>', unsafe_allow_html=True)
        elif kpis["growth_last"] > 0:
            st.markdown(f'<div class="alert-green">🟢 <b>CA en hausse</b> : {kpis["growth_last"]:+.1f}% vs mois précédent</div>', unsafe_allow_html=True)

        if kpis["repeat_rate"] < 10:
            st.markdown(f'<div class="alert-orange">🟠 <b>Faible fidélisation</b> : seulement {kpis["repeat_rate"]:.1f}% de réachat</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">🟢 <b>Fidélisation</b> : {kpis["repeat_rate"]:.1f}% de réachat</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 KPIs instantanés")
        metrics = [
            ("💰 CA Total",       f"R$ {kpis['total_revenue']/1e6:.2f}M"),
            ("📦 Commandes",       f"{kpis['total_orders']:,}"),
            ("👤 Clients",         f"{kpis['total_customers']:,}"),
            ("🛒 Panier moyen",    f"R$ {kpis['avg_ticket']:.0f}"),
            ("⭐ Note moyenne",    f"{kpis['avg_score']:.2f} / 5"),
            ("📈 Croissance",      f"{kpis['growth_last']:+.1f}%"),
        ]
        for label, value in metrics:
            st.markdown(f"**{label}** &nbsp;&nbsp; `{value}`")

    st.markdown("---")

    # Analyse IA automatique
    st.markdown("#### 🤖 Analyse IA automatique")
    if not api_key:
        st.info("Entre ta clé API Gemini dans la sidebar pour activer l'analyse IA automatique.")
        return

    if "auto_summary" not in st.session_state:
        with st.spinner("L'assistant analyse tes données..."):
            context = build_context(kpis)
            prompt_auto = f"""
Tu es Olist BI Copilot, un expert en e-commerce et data analytics.

Voici les données actuelles :
{context}

Génère un résumé exécutif structuré contenant :

1. 📊 SYNTHESE GLOBALE (3 phrases max sur la santé du business)
2. ✅ POINTS FORTS (2-3 points positifs avec chiffres)
3. ⚠️ POINTS A AMELIORER (2-3 problèmes identifiés avec chiffres)
4. 💡 RECOMMANDATION PRIORITAIRE (1 action concrète à faire maintenant)

Sois direct, précis, utilise les vrais chiffres. Réponds en français.
"""
            result = call_gemini_cached(api_key, prompt_auto)
            st.session_state["auto_summary"] = result

    st.markdown(st.session_state["auto_summary"])

    if st.button("🔄 Rafraîchir l'analyse"):
        del st.session_state["auto_summary"]
        st.rerun()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Olist BI Copilot")
    st.markdown("---")

    DATA_PATH = st.text_input(
        "📁 Chemin des données",
        value="C:/Users/HP/olist_dashboard/",
        help="Dossier contenant olist_master.parquet et olist_rfm.parquet"
    )

    st.markdown("---")

    GEMINI_KEY = st.text_input(
        "🔑 Clé API Gemini (gratuite)",
        type="password",
        placeholder="AIza...",
        help="Obtiens ta clé GRATUITE sur aistudio.google.com"
    )
    if not GEMINI_KEY:
        st.markdown("<small style='color:#BA7517'>👉 [Obtenir une clé gratuite](https://aistudio.google.com/app/apikey)</small>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#1D9E75'>✅ Clé API configurée</small>", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil & Résumé", "📊 Vue générale", "🗺️ Géographie",
         "🚚 Logistique", "👥 Clients RFM", "🏪 Vendeurs",
         "🔍 Anomalies & Recommandations", "🤖 Assistant IA"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<small style='color:#8B949E'>Olist BI Copilot v2.0<br>Powered by Gemini AI</small>",
                unsafe_allow_html=True)

# ── Chargement ───────────────────────────────────────────────
try:
    master, delivered, rfm = load_data(DATA_PATH)
    kpis = compute_kpis(master, delivered, rfm)
except Exception as e:
    st.error(f"❌ Impossible de charger les données depuis `{DATA_PATH}`\n\n{e}")
    st.info("👉 Vérifie que `olist_master.parquet` et `olist_rfm.parquet` sont dans ce dossier.")
    st.stop()

TEAL, PURPLE, CORAL, AMBER, BLUE = "#1D9E75","#534AB7","#D85A30","#BA7517","#185FA5"


# ════════════════════════════════════════════════════════════
# PAGE 0 — Accueil & Résumé automatique
# ════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Résumé":
    render_auto_summary(kpis, GEMINI_KEY)


# ════════════════════════════════════════════════════════════
# PAGE 1 — Vue générale
# ════════════════════════════════════════════════════════════
elif page == "📊 Vue générale":
    st.markdown("## 📊 Vue générale")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        all_years = sorted(master["purchase_year"].dropna().unique().astype(int))
        selected_years = st.multiselect("Année", all_years, default=all_years)
    with col_f2:
        all_states = sorted(master["customer_state"].dropna().unique())
        selected_states = st.multiselect("État", all_states, default=all_states)
    with col_f3:
        all_cats = sorted(master["category_en"].dropna().unique())
        selected_cats = st.multiselect("Catégorie", all_cats, default=all_cats)

    df = master[
        master["purchase_year"].isin(selected_years) &
        master["customer_state"].isin(selected_states) &
        master["category_en"].isin(selected_cats)
    ]
    df_del = delivered[
        delivered["purchase_year"].isin(selected_years) &
        delivered["customer_state"].isin(selected_states) &
        delivered["category_en"].isin(selected_cats)
    ]

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    kpi_list = [
        (k1, "Chiffre d affaires", f"R$ {df_del['total_price'].sum()/1e6:.2f}M",
         f"Panier moyen : R$ {df_del['total_price'].mean():.0f}", TEAL),
        (k2, "Commandes", f"{df_del['order_id'].nunique():,}",
         f"{df['customer_unique_id'].nunique():,} clients", PURPLE),
        (k3, "Note moyenne", f"{df_del['review_score'].mean():.2f} / 5",
         f"Retards : {df_del['is_late'].mean()*100:.1f}%", AMBER),
        (k4, "Vendeurs", f"{df['seller_id'].nunique():,}",
         f"{df['category_en'].nunique()} categories", CORAL),
    ]
    for col, label, value, sub, color in kpi_list:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("###")
    c1, c2 = st.columns([2, 1])

    with c1:
        monthly_f = (df
            .groupby("purchase_yearmonth").agg(revenue=("total_price","sum"))
            .reset_index().sort_values("purchase_yearmonth").iloc[1:-1]
        )
        monthly_f["ds"] = pd.to_datetime(monthly_f["purchase_yearmonth"] + "-01")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_f["ds"], y=monthly_f["revenue"],
            fill="tozeroy", fillcolor="rgba(29,158,117,0.12)",
            line=dict(color=TEAL, width=2.5), mode="lines+markers", marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>R$ %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            title="Evolution du CA mensuel",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickformat=",.0f"),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pay = df["payment_type"].value_counts().reset_index()
        pay.columns = ["type","count"]
        pay = pay[pay["type"] != "not_defined"]
        pay_labels = {"credit_card":"Carte credit","boleto":"Boleto",
                      "voucher":"Voucher","debit_card":"Carte debit"}
        pay["label"] = pay["type"].map(pay_labels).fillna(pay["type"])
        fig2 = go.Figure(go.Pie(
            labels=pay["label"], values=pay["count"],
            hole=0.5, marker_colors=[TEAL,PURPLE,CORAL,AMBER], textfont=dict(size=10)
        ))
        fig2.update_layout(
            title="Modes de paiement", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=300,
            margin=dict(l=0,r=0,t=40,b=0), legend=dict(font=dict(color="#CBD5E0"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        cat = (df_del.groupby("category_en").agg(revenue=("total_price","sum"))
            .reset_index().dropna().nlargest(10,"revenue").sort_values("revenue"))
        fig3 = go.Figure(go.Bar(
            y=cat["category_en"], x=cat["revenue"], orientation="h",
            marker=dict(color=cat["revenue"], colorscale="Teal", showscale=False),
            hovertemplate="%{y}<br>R$ %{x:,.0f}<extra></extra>"
        ))
        fig3.update_layout(
            title="Top 10 categories - CA",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=350,
            xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False),
            margin=dict(l=0,r=20,t=40,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        hour = df.groupby("purchase_hour")["order_id"].nunique().reset_index()
        hour.columns = ["hour","nb_orders"]
        fig4 = go.Figure(go.Bar(
            x=hour["hour"], y=hour["nb_orders"],
            marker=dict(color=hour["nb_orders"], colorscale="Purples", showscale=False),
            hovertemplate="%{x}h -> %{y:,} commandes<extra></extra>"
        ))
        fig4.update_layout(
            title="Commandes par heure",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=350,
            xaxis=dict(showgrid=False, dtick=2),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — Géographie
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Géographie":
    st.markdown("## 🗺️ Géographie des ventes")
    geo = (delivered.groupby("customer_state")
        .agg(nb_orders=("order_id","nunique"), revenue=("total_price","sum"),
             avg_score=("review_score","mean"), late_rate=("is_late","mean"))
        .reset_index())
    geo["late_pct"] = geo["late_rate"] * 100

    metric = st.selectbox("Métrique", ["nb_orders","revenue","avg_score","late_pct"],
        format_func=lambda x: {"nb_orders":"Commandes","revenue":"CA (BRL)",
                                "avg_score":"Note moyenne","late_pct":"Taux retard (%)"}[x])

    c1, c2 = st.columns([3,2])
    with c1:
        top10 = geo.sort_values(metric, ascending=False).head(10).sort_values(metric)
        colors_map = {"nb_orders":BLUE,"revenue":TEAL,"avg_score":AMBER,"late_pct":CORAL}
        fig = go.Figure(go.Bar(
            y=top10["customer_state"], x=top10[metric], orientation="h",
            marker_color=colors_map[metric], opacity=0.85))
        fig.update_layout(
            title="Top 10 etats",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(
            geo.sort_values(metric, ascending=False)
               .assign(late_pct=lambda d: d["late_pct"].round(1),
                       avg_score=lambda d: d["avg_score"].round(2),
                       revenue=lambda d: d["revenue"].round(0))
               [["customer_state","nb_orders","revenue","avg_score","late_pct"]]
               .rename(columns={"customer_state":"Etat","nb_orders":"Commandes",
                                "revenue":"CA","avg_score":"Note","late_pct":"Retard %"})
               .reset_index(drop=True),
            height=400, use_container_width=True)

    fig2 = px.scatter(geo, x="nb_orders", y="avg_score",
        size="revenue", color="late_pct", hover_name="customer_state",
        color_continuous_scale="RdYlGn_r",
        labels={"nb_orders":"Commandes","avg_score":"Note","late_pct":"Retard (%)"},
        title="Commandes vs Note par etat")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"), height=380,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — Logistique
# ════════════════════════════════════════════════════════════
elif page == "🚚 Logistique":
    st.markdown("## 🚚 Logistique & Satisfaction")

    k1,k2,k3,k4 = st.columns(4)
    for col, label, value, sub, color in [
        (k1,"Delai median reel",   f"{delivered['days_to_deliver_actual'].median():.0f} j","",TEAL),
        (k2,"Delai median estime", f"{delivered['days_to_deliver_estimated'].median():.0f} j","",PURPLE),
        (k3,"Taux de retard",      f"{delivered['is_late'].mean()*100:.1f}%",
         f"Retard moyen : {delivered.loc[delivered['is_late']==1,'delay_days'].mean():.1f}j",CORAL),
        (k4,"Note globale",        f"{delivered['review_score'].mean():.2f} / 5","",AMBER),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("###")
    c1,c2 = st.columns(2)

    with c1:
        ds = (delivered.groupby("delay_bucket", observed=True)
              .agg(avg_score=("review_score","mean"), nb_orders=("order_id","count"))
              .reset_index())
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(
            x=ds["delay_bucket"].astype(str), y=ds["nb_orders"], name="Commandes",
            marker_color=[TEAL,TEAL,TEAL,AMBER,CORAL,CORAL,"#A83010"], opacity=0.65),
            secondary_y=False)
        fig.add_trace(go.Scatter(
            x=ds["delay_bucket"].astype(str), y=ds["avg_score"], name="Note moy.",
            mode="lines+markers", line=dict(color=PURPLE, width=2.5), marker=dict(size=8)),
            secondary_y=True)
        fig.update_layout(
            title="Note client selon le retard",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=350,
            legend=dict(font=dict(color="#CBD5E0")), margin=dict(l=0,r=30,t=40,b=0))
        fig.update_yaxes(range=[1,5.5], secondary_y=True, showgrid=False, color=PURPLE)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=delivered["days_to_deliver_actual"].clip(upper=60),
            name="Reel", nbinsx=40, opacity=0.7, marker_color=TEAL))
        fig2.add_trace(go.Histogram(x=delivered["days_to_deliver_estimated"].clip(upper=60),
            name="Estime", nbinsx=40, opacity=0.5, marker_color=PURPLE))
        fig2.update_layout(
            barmode="overlay", title="Delai reel vs estime",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=350,
            xaxis=dict(title="Jours", showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(font=dict(color="#CBD5E0")), margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    dstate = (delivered.groupby("customer_state")
        .agg(avg_delay=("days_to_deliver_actual","mean"), late_pct=("is_late","mean"))
        .reset_index().sort_values("avg_delay", ascending=False))
    dstate["late_pct"] *= 100
    fig3 = px.bar(dstate, x="customer_state", y="avg_delay", color="late_pct",
        color_continuous_scale="RdYlGn_r",
        labels={"customer_state":"Etat","avg_delay":"Delai moyen (j)","late_pct":"Retard %"},
        title="Delai et retard par etat")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"), height=350,
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — Clients RFM
# ════════════════════════════════════════════════════════════
elif page == "👥 Clients RFM":
    st.markdown("## 👥 Segmentation clients RFM")
    seg = (rfm.groupby("segment")
        .agg(nb_clients=("customer_unique_id","count"),
             avg_recency=("recency","mean"), avg_frequency=("frequency","mean"),
             avg_monetary=("monetary","mean"), total_revenue=("monetary","sum"))
        .reset_index().sort_values("total_revenue", ascending=False))
    seg["pct_clients"] = seg["nb_clients"]/seg["nb_clients"].sum()*100
    seg["pct_revenue"] = seg["total_revenue"]/seg["total_revenue"].sum()*100

    seg_colors = [TEAL,"#5BAF8E",BLUE,AMBER,CORAL,"#A83010",PURPLE]
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=seg["segment"], values=seg["nb_clients"],
            hole=0.55, marker_colors=seg_colors, textinfo="label+percent",
            textfont=dict(size=10)))
        fig.update_layout(title="Repartition des clients", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E0"), height=350, showlegend=False,
            margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Bar(x=seg["segment"], y=seg["total_revenue"]/1000,
            marker_color=seg_colors, opacity=0.85,
            text=seg["pct_revenue"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside", textfont=dict(color="#CBD5E0")))
        fig2.update_layout(title="CA par segment", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E0"), height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="K BRL"),
            margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Detail des segments")
    st.dataframe(
        seg.assign(
            avg_recency=lambda d: d["avg_recency"].round(0).astype(int),
            avg_frequency=lambda d: d["avg_frequency"].round(2),
            avg_monetary=lambda d: d["avg_monetary"].round(0),
            pct_clients=lambda d: d["pct_clients"].round(1),
            pct_revenue=lambda d: d["pct_revenue"].round(1),
        )[["segment","nb_clients","pct_clients","avg_recency","avg_frequency","avg_monetary","pct_revenue"]]
        .rename(columns={"segment":"Segment","nb_clients":"Clients","pct_clients":"% clients",
            "avg_recency":"Recence (j)","avg_frequency":"Freq.","avg_monetary":"Montant (BRL)","pct_revenue":"% CA"}),
        use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — Vendeurs
# ════════════════════════════════════════════════════════════
elif page == "🏪 Vendeurs":
    st.markdown("## 🏪 Performance des vendeurs")
    min_orders = st.slider("Commandes minimum par vendeur", 10, 200, 30, step=10)

    sp = (delivered.groupby("seller_id")
        .agg(nb_orders=("order_id","count"), revenue=("total_price","sum"),
             avg_score=("review_score","mean"), late_rate=("is_late","mean"))
        .reset_index().query(f"nb_orders >= {min_orders}"))
    sp["late_pct"] = sp["late_rate"] * 100

    k1,k2,k3 = st.columns(3)
    k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Vendeurs analyses</div>
        <div class="kpi-value" style="color:{TEAL}">{len(sp):,}</div>
        <div class="kpi-sub">avec >= {min_orders} commandes</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Note mediane</div>
        <div class="kpi-value" style="color:{AMBER}">{sp['avg_score'].median():.2f} / 5</div>
        <div class="kpi-sub">&nbsp;</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Retard median</div>
        <div class="kpi-value" style="color:{CORAL}">{sp['late_pct'].median():.1f}%</div>
        <div class="kpi-sub">&nbsp;</div></div>""", unsafe_allow_html=True)

    fig = px.scatter(sp, x="late_pct", y="avg_score", size="nb_orders", color="revenue",
        color_continuous_scale="Teal",
        hover_data={"seller_id":True,"nb_orders":True,"revenue":":.0f","late_pct":":.1f","avg_score":":.2f"},
        labels={"late_pct":"Retard (%)","avg_score":"Note","nb_orders":"Cmds","revenue":"CA"},
        title="Retard vs Note vendeurs")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"), height=450,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    sp["score"] = ((sp["avg_score"]/5)*0.5 + (1-sp["late_rate"])*0.3 +
                   (sp["revenue"]/sp["revenue"].max())*0.2)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 🏆 Top 10")
        t = sp.nlargest(10,"score")[["seller_id","nb_orders","revenue","avg_score","late_pct"]].copy()
        t["seller_id"] = t["seller_id"].str[:10]+"..."
        t.columns = ["Vendeur","Cmds","CA","Note","Retard %"]
        st.dataframe(t.style.format({"CA":"{:,.0f}","Note":"{:.2f}","Retard %":"{:.1f}"}),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### 🔴 Flop 10")
        f = sp.nsmallest(10,"score")[["seller_id","nb_orders","revenue","avg_score","late_pct"]].copy()
        f["seller_id"] = f["seller_id"].str[:10]+"..."
        f.columns = ["Vendeur","Cmds","CA","Note","Retard %"]
        st.dataframe(f.style.format({"CA":"{:,.0f}","Note":"{:.2f}","Retard %":"{:.1f}"}),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 6 — Anomalies & Recommandations
# ════════════════════════════════════════════════════════════
elif page == "🔍 Anomalies & Recommandations":
    render_anomalies(master, delivered, rfm, GEMINI_KEY)


# ════════════════════════════════════════════════════════════
# PAGE 7 — Assistant IA
# ════════════════════════════════════════════════════════════
elif page == "🤖 Assistant IA":
    st.markdown("## 🤖 Olist BI Copilot — Assistant IA")
    st.markdown("Pose n'importe quelle question sur tes données. L'assistant connaît tous tes KPIs en temps réel.")

    if not GEMINI_KEY:
        st.warning("Entre ta clé API Gemini dans la sidebar pour activer l'assistant.")
        st.markdown("**Clé gratuite** → [aistudio.google.com](https://aistudio.google.com/app/apikey)")
    else:
        context = build_context(kpis)
        system_prompt = f"""Tu es Olist BI Copilot, un expert en data analytics et stratégie e-commerce.
Tu analyses les données d'une marketplace brésilienne.

Données actuelles en temps réel :
{context}

Règles :
- Réponds toujours en français
- Utilise les vrais chiffres des données ci-dessus
- Structure tes réponses avec des titres et des emojis
- Donne des recommandations concrètes et actionnables
- Garde la mémoire de toute la conversation
- Si on te pose une question vague comme "et les clients ?", rappelle-toi du contexte précédent
"""

        # Suggestions rapides
        st.markdown("**💬 Questions rapides :**")
        col1, col2, col3, col4 = st.columns(4)
        suggestions = {
            col1: "Analyse globale et points critiques",
            col2: "Stratégies pour réduire les retards",
            col3: "Comment améliorer la rétention clients ?",
            col4: "Analyse SWOT de mon business",
        }
        for col, text in suggestions.items():
            if col.button(text, use_container_width=True):
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role":"user","content":text})
                st.rerun()

        st.markdown("---")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Afficher historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input utilisateur
        user_input = st.chat_input("Ex: Pourquoi mes ventes baissent ? Quels vendeurs accompagner ?")

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


# ════════════════════════════════════════════════════════════
# MODULE DETECTION ANOMALIES
# ════════════════════════════════════════════════════════════

def detect_anomalies(master, delivered, rfm):
    """Détecte automatiquement les anomalies dans les données."""
    anomalies = []
    recommandations = []

    # ── 1. Taux de retard ────────────────────────────────────
    late_rate = delivered["is_late"].mean() * 100
    if late_rate > 15:
        anomalies.append({
            "niveau": "CRITIQUE",
            "icone": "🔴",
            "titre": "Taux de retard très élevé",
            "detail": f"{late_rate:.1f}% des commandes sont en retard (seuil critique : 15%)",
            "impact": "Fort impact sur la satisfaction client"
        })
        recommandations.append({
            "priorite": 1,
            "action": "Auditer immédiatement les vendeurs avec >20% de retard",
            "detail": "Identifier les transporteurs défaillants et renégocier les contrats logistiques"
        })
    elif late_rate > 10:
        anomalies.append({
            "niveau": "ALERTE",
            "icone": "🟠",
            "titre": "Taux de retard élevé",
            "detail": f"{late_rate:.1f}% des commandes sont en retard (seuil alerte : 10%)",
            "impact": "Impact modéré sur la satisfaction"
        })
        recommandations.append({
            "priorite": 2,
            "action": "Surveiller les vendeurs à risque",
            "detail": "Mettre en place des alertes automatiques pour les vendeurs dépassant 10% de retard"
        })

    # ── 2. Note moyenne ──────────────────────────────────────
    avg_score = delivered["review_score"].mean()
    if avg_score < 3.5:
        anomalies.append({
            "niveau": "CRITIQUE",
            "icone": "🔴",
            "titre": "Satisfaction client critique",
            "detail": f"Note moyenne {avg_score:.2f}/5 — en dessous du seuil acceptable (3.5)",
            "impact": "Risque de perte massive de clients"
        })
        recommandations.append({
            "priorite": 1,
            "action": "Lancer une enquête satisfaction immédiate",
            "detail": "Contacter les clients ayant donné une note ≤2 et offrir une compensation"
        })
    elif avg_score < 4.0:
        anomalies.append({
            "niveau": "ALERTE",
            "icone": "🟠",
            "titre": "Satisfaction à améliorer",
            "detail": f"Note moyenne {avg_score:.2f}/5 — sous l'objectif de 4.0",
            "impact": "Risque de churn progressif"
        })
        recommandations.append({
            "priorite": 2,
            "action": "Analyser les avis négatifs par catégorie",
            "detail": "Identifier les catégories avec les moins bonnes notes et contacter les vendeurs concernés"
        })

    # ── 3. Taux de réachat ───────────────────────────────────
    repeat_rate = (rfm["frequency"] >= 2).mean() * 100
    if repeat_rate < 5:
        anomalies.append({
            "niveau": "CRITIQUE",
            "icone": "🔴",
            "titre": "Fidélisation très faible",
            "detail": f"Seulement {repeat_rate:.1f}% des clients rachètent (objectif : >10%)",
            "impact": "Coût d'acquisition très élevé sans rétention"
        })
        recommandations.append({
            "priorite": 1,
            "action": "Lancer un programme de fidélité urgent",
            "detail": "Envoyer des coupons de réduction aux clients inactifs depuis 90 jours"
        })
    elif repeat_rate < 10:
        anomalies.append({
            "niveau": "ALERTE",
            "icone": "🟠",
            "titre": "Faible fidélisation",
            "detail": f"{repeat_rate:.1f}% de réachat — objectif 10%",
            "impact": "Rentabilité réduite"
        })
        recommandations.append({
            "priorite": 2,
            "action": "Campagne email de réactivation",
            "detail": "Cibler les clients du segment 'Hibernants' avec une offre personnalisée"
        })

    # ── 4. Croissance CA ─────────────────────────────────────
    monthly = (master.groupby("purchase_yearmonth")["total_price"].sum()
               .reset_index().sort_values("purchase_yearmonth").iloc[1:-1])
    if len(monthly) >= 2:
        growth = (monthly["total_price"].iloc[-1] / monthly["total_price"].iloc[-2] - 1) * 100
        if growth < -10:
            anomalies.append({
                "niveau": "CRITIQUE",
                "icone": "🔴",
                "titre": "Chute du CA",
                "detail": f"CA en baisse de {abs(growth):.1f}% vs mois précédent",
                "impact": "Tendance préoccupante à surveiller"
            })
            recommandations.append({
                "priorite": 1,
                "action": "Analyse des causes de la baisse",
                "detail": "Vérifier si la baisse est liée à une catégorie, une région ou une période saisonnière"
            })
        elif growth > 20:
            anomalies.append({
                "niveau": "BON",
                "icone": "🟢",
                "titre": "Forte croissance du CA",
                "detail": f"CA en hausse de {growth:.1f}% vs mois précédent",
                "impact": "Tendance très positive"
            })
            recommandations.append({
                "priorite": 3,
                "action": "Capitaliser sur la croissance",
                "detail": "Identifier les catégories qui progressent le plus et augmenter les stocks"
            })

    # ── 5. Concentration vendeurs (Pareto) ───────────────────
    seller_rev = (delivered.groupby("seller_id")["total_price"].sum()
                  .sort_values(ascending=False).reset_index())
    top20_pct = seller_rev.head(int(len(seller_rev)*0.2))["total_price"].sum()
    total_rev = seller_rev["total_price"].sum()
    pareto = top20_pct / total_rev * 100
    if pareto > 80:
        anomalies.append({
            "niveau": "ALERTE",
            "icone": "🟠",
            "titre": "Forte concentration vendeurs",
            "detail": f"Les 20% meilleurs vendeurs génèrent {pareto:.0f}% du CA",
            "impact": "Risque de dépendance — perte d'un vendeur = impact majeur"
        })
        recommandations.append({
            "priorite": 2,
            "action": "Diversifier le portefeuille vendeurs",
            "detail": "Recruter et accompagner de nouveaux vendeurs pour réduire la dépendance"
        })

    # ── 6. Clients à risque de churn ─────────────────────────
    if "churn_risk" in rfm.columns:
        churn_rate = rfm["churn_risk"].mean() * 100
    else:
        churn_rate = (rfm["recency"] > 180).mean() * 100

    if churn_rate > 50:
        anomalies.append({
            "niveau": "ALERTE",
            "icone": "🟠",
            "titre": "Fort risque de churn",
            "detail": f"{churn_rate:.1f}% des clients inactifs depuis >180 jours",
            "impact": "Perte potentielle de revenus importants"
        })
        recommandations.append({
            "priorite": 2,
            "action": "Campagne de réactivation",
            "detail": f"Contacter les {rfm[rfm['recency']>180].shape[0]:,} clients inactifs avec une offre spéciale"
        })

    return anomalies, sorted(recommandations, key=lambda x: x["priorite"])


def render_anomalies(master, delivered, rfm, api_key):
    """Affiche la page de détection d'anomalies."""
    st.markdown("## 🔍 Détection d'anomalies & Recommandations")

    anomalies, recommandations = detect_anomalies(master, delivered, rfm)

    # ── Résumé rapide ────────────────────────────────────────
    critiques = len([a for a in anomalies if a["niveau"] == "CRITIQUE"])
    alertes   = len([a for a in anomalies if a["niveau"] == "ALERTE"])
    bons      = len([a for a in anomalies if a["niveau"] == "BON"])

    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Points critiques</div>
        <div class="kpi-value" style="color:#D85A30">{critiques}</div>
        <div class="kpi-sub">Action immédiate requise</div>
    </div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Alertes</div>
        <div class="kpi-value" style="color:#BA7517">{alertes}</div>
        <div class="kpi-sub">A surveiller</div>
    </div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Points positifs</div>
        <div class="kpi-value" style="color:#1D9E75">{bons}</div>
        <div class="kpi-sub">Tendances favorables</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("###")
    c1, c2 = st.columns(2)

    # ── Anomalies ────────────────────────────────────────────
    with c1:
        st.markdown("### 🚨 Anomalies détectées")
        if not anomalies:
            st.success("✅ Aucune anomalie détectée — tout semble normal !")
        for a in anomalies:
            couleur = {"CRITIQUE":"alert-red","ALERTE":"alert-orange","BON":"alert-green"}.get(a["niveau"],"alert-orange")
            st.markdown(f"""
            <div class="{couleur}">
                <b>{a['icone']} {a['titre']}</b><br>
                <small>{a['detail']}</small><br>
                <small style="color:#8B949E">Impact : {a['impact']}</small>
            </div>""", unsafe_allow_html=True)

    # ── Recommandations ──────────────────────────────────────
    with c2:
        st.markdown("### 💡 Recommandations prioritaires")
        if not recommandations:
            st.success("✅ Aucune action urgente requise !")
        for i, r in enumerate(recommandations, 1):
            couleur_prio = {1:"#D85A30", 2:"#BA7517", 3:"#1D9E75"}.get(r["priorite"],"#8B949E")
            st.markdown(f"""
            <div class="summary-box" style="border-left: 3px solid {couleur_prio}; padding-left:16px;">
                <b style="color:{couleur_prio}">#{i} — {r['action']}</b><br>
                <small style="color:#CBD5E0">{r['detail']}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Analyse IA approfondie ───────────────────────────────
    st.markdown("### 🤖 Analyse IA approfondie des anomalies")

    if not api_key:
        st.info("Entre ta clé API Gemini dans la sidebar pour obtenir une analyse IA détaillée.")
        return

    anomalies_text = "\n".join([
        f"- {a['icone']} {a['titre']} : {a['detail']}"
        for a in anomalies
    ])

    if st.button("🔍 Analyser les anomalies avec l'IA", type="primary"):
        with st.spinner("Analyse IA en cours..."):
            prompt = f"""
Tu es Olist BI Copilot. Voici les anomalies détectées automatiquement :

{anomalies_text}

Pour chaque anomalie :
1. Explique clairement POURQUOI c'est un problème
2. Donne la CAUSE PROBABLE
3. Propose UN PLAN D'ACTION concret en 3 étapes
4. Estime l'IMPACT si rien n'est fait

Termine par une PRIORITÉ ABSOLUE — l'action la plus urgente à faire maintenant.
Réponds en français avec des emojis et une structure claire.
"""
            response = call_gemini_cached(api_key, prompt)
            st.markdown(response)

    # ── Graphique anomalies visuelles ────────────────────────
    st.markdown("### 📊 Tableau de bord des indicateurs de santé")

    indicators = {
        "Taux retard (%)": (delivered["is_late"].mean()*100, 10, 15),
        "Note /5": (delivered["review_score"].mean(), 4.0, 3.5),
        "Reachat (%)": ((rfm["frequency"]>=2).mean()*100, 10, 5),
    }

    fig = go.Figure()
    for name, (value, seuil_alerte, seuil_critique) in indicators.items():
        if name == "Note /5":
            color = TEAL if value >= seuil_alerte else (AMBER if value >= seuil_critique else CORAL)
        else:
            color = TEAL if value <= seuil_alerte else (AMBER if value <= seuil_critique else CORAL)

        fig.add_trace(go.Bar(
            name=name, x=[name], y=[value],
            marker_color=color, opacity=0.85,
            text=f"{value:.1f}", textposition="outside",
            textfont=dict(color="#CBD5E0")
        ))

    fig.update_layout(
        title="Indicateurs clés vs seuils",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E0"), height=350,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0,r=0,t=40,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
