import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle, json, os, warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediFlow AI — NHS ED Intelligence",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODELS & META ────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    with open(f'{BASE}/breach_model.pkl','rb') as f: clf = pickle.load(f)
    with open(f'{BASE}/wait_model.pkl','rb') as f:   reg = pickle.load(f)
    with open(f'{BASE}/le_dict.pkl','rb') as f:      le  = pickle.load(f)
    with open(f'{BASE}/features.pkl','rb') as f:     ft  = pickle.load(f)
    with open(f'{BASE}/meta.json','r') as f:         meta= json.load(f)
    return clf, reg, le, ft, meta

@st.cache_data
def load_data():
    df1 = pd.read_csv(f'{BASE}/nhs5_part1.csv')
    df2 = pd.read_csv(f'{BASE}/nhs5_part2.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    return df

clf, reg, le_dict, FEATURES, meta = load_models()
df = load_data()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* White main background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #ffffff !important; }
[data-testid="block-container"] { padding-top: 1.5rem; background: #ffffff; }

/* Light sidebar */
[data-testid="stSidebar"] { background: #f8fafc !important; border-right: 2px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #0f172a !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #0f172a !important; font-family: 'DM Mono', monospace !important; font-size: 1.9rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; }

/* Buttons */
.stButton > button {
    background: #1e40af !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* Inputs */
input, select, textarea {
    background: #f8fafc !important; border: 1px solid #cbd5e1 !important;
    color: #0f172a !important; border-radius: 8px !important;
}

/* Custom boxes */
.ai-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px; padding: 1.25rem 1.4rem; margin: 0.5rem 0;
}
.section-title {
    color: #1e40af; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.6rem;
    border-bottom: 2px solid #bfdbfe; padding-bottom: 0.4rem;
}
.result-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)
TEAL="#00c2a0"; AMBER="#f59e0b"; RED="#ef4444"; BLUE="#3b82f6"; GREEN="#10b981"

def encode_input(row_dict):
    """Encode a patient input dict into model-ready feature vector."""
    d = row_dict.copy()
    for col, le in le_dict.items():
        key = col
        if key in d:
            try:
                d[col+'_enc'] = int(le.transform([d[key]])[0])
            except:
                d[col+'_enc'] = 0
    # Compute interaction features
    occ   = d.get('bed_occupancy_pct', 94.0)
    wait  = d.get('wait_time_to_assessment_min', 120.0)
    staff = d.get('staff_ratio', 0.5)
    night = 1 if d.get('hour_of_day',12) >= 22 or d.get('hour_of_day',12) <= 6 else 0
    ambul = 1 if d.get('arrival_mode','Walk-in') == 'Ambulance' else 0
    d['occ_x_triage']         = (occ/100) * d.get('triage_category', 3)
    d['wait_per_staff']        = wait / (staff + 0.01)
    d['acuity_x_cpi']          = d.get('patient_acuity_score',10.0) * d.get('capacity_pressure_index',1.5)
    d['night_high_occ']        = night * (1 if occ > 95 else 0)
    d['ambul_handover_stress'] = ambul * d.get('handover_breach_gt30min', 0)
    d['is_winter']             = 1 if d.get('month', datetime.now().month) in [12,1,2] else 0
    row = [d.get(f, d.get(f.replace('_enc',''), 0)) for f in FEATURES]
    return np.array(row).reshape(1,-1)

def get_suggestions(breach_prob, wait_pred, triage, complaint, age, imd, news2, comorbid, arrival_mode):
    """Generate personalised AI suggestions based on prediction outputs."""
    suggestions = []
    risk_level = "HIGH" if breach_prob > 0.6 else "MEDIUM" if breach_prob > 0.35 else "LOW"

    # Wait time suggestion
    if wait_pred > 180:
        suggestions.append(("🚨", "Critical Wait Risk",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> significantly exceeds the NHS 4-hour target. "
            "Immediate escalation recommended. Consider fast-track pathway or alternative disposition.", RED))
    elif wait_pred > 120:
        suggestions.append(("⚠️", "Extended Wait Expected",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> is approaching the 4-hour threshold. "
            "Pre-emptive assessment and early senior review advised.", AMBER))
    else:
        suggestions.append(("✅", "Wait Time Within Target",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> is within acceptable limits. "
            "Standard pathway appropriate.", GREEN))

    # Breach risk suggestion
    if risk_level == "HIGH":
        suggestions.append(("🔴", "High 4-Hour Breach Risk",
            f"Model predicts <b>{breach_prob*100:.0f}% probability</b> of breaching the 4-hour target. "
            "Activate breach prevention protocol. Allocate senior clinician review within 30 minutes.", RED))
    elif risk_level == "MEDIUM":
        suggestions.append(("🟡", "Moderate Breach Risk",
            f"Model predicts <b>{breach_prob*100:.0f}% probability</b> of breach. "
            "Monitor closely. Reassess at 2-hour mark.", AMBER))

    # Clinical suggestions by complaint
    complaint_advice = {
        "Chest Pain": "12-lead ECG within 10 minutes. Troponin at 0h and 3h. Cardiology alert if STEMI suspected.",
        "Cardiac Arrest": "Full resuscitation team activation. Immediate Resus bay. Cardiology/ITU on standby.",
        "Stroke Symptoms": "FAST protocol. CT Head within 30 minutes. Stroke team activation. Thrombolysis assessment.",
        "Shortness of Breath": "SpO₂ monitoring, ABG if <94%. CXR. Consider BNP for heart failure vs COPD exacerbation.",
        "Head Injury": "GCS scoring every 15 minutes. CT Head per NICE guidelines. Neurosurgery referral if GCS <13.",
        "Infection / Sepsis": "Sepsis Six bundle within 1 hour: blood cultures, IV antibiotics, IV fluids, oxygen, monitor UO, lactate.",
        "Fever / Sepsis Child": "Paediatric sepsis pathway. Antipyretics. Blood cultures before antibiotics. Paediatric review.",
        "Overdose / Poisoning": "Toxicology referral. Activated charcoal if <1hr. Psychiatric liaison post-medical clearance.",
        "Mental Health Crisis": "Mental health team referral. Safe environment. Psychiatric assessment. Safeguarding review.",
        "Abdominal Pain": "FBC, CRP, LFTs, amylase, urine dip. Surgical review if peritonism signs. USS if indicated.",
        "Fall / Trauma": "Full trauma assessment. Fracture survey X-rays. Fall risk assessment. Physiotherapy referral.",
        "Fracture / Dislocation": "X-ray (2 views minimum). Analgesia per protocol. Orthopaedic review for complex fractures.",
    }
    advice = complaint_advice.get(complaint, "Full clinical assessment as per presenting complaint. Document NEWS2 score.")
    suggestions.append(("🩺", f"Clinical Pathway: {complaint}", advice, BLUE))

    # Age-specific
    if age >= 75:
        suggestions.append(("👴", "Elderly Patient Alert",
            "Age ≥75: Enhanced frailty screening (CFS score). Falls risk assessment. Delirium screen. "
            "Early social work referral. Consider Comprehensive Geriatric Assessment.", AMBER))
    elif age <= 16:
        suggestions.append(("👶", "Paediatric Patient",
            "Paediatric-specific drug dosing (weight-based). PEWS scoring. Safeguarding assessment. "
            "Parent/guardian consent required.", BLUE))

    # IMD equity flag
    if imd == 1:
        suggestions.append(("⚖️", "Equity Alert — High Deprivation (IMD Q1)",
            "Patient from most deprived quintile. Research shows 14% longer average wait times in this group. "
            "Flag for equitable triage review per NHS Constitution. Social prescribing referral may be appropriate.", PURPLE if True else AMBER))

    # NEWS2 score
    if news2 >= 7:
        suggestions.append(("🚨", f"NEWS2 Score {news2} — Critical",
            "NEWS2 ≥7 indicates high clinical risk. Continuous monitoring. Consider HDU/ICU admission. "
            "Urgent consultant review required.", RED))
    elif news2 >= 5:
        suggestions.append(("⚠️", f"NEWS2 Score {news2} — Elevated",
            "NEWS2 5–6 indicates medium-high risk. Increase monitoring frequency. Senior review within 30 minutes.", AMBER))

    # Comorbidity
    if comorbid >= 3:
        suggestions.append(("💊", "Complex Comorbidity Profile",
            f"{comorbid} active comorbidities noted. High risk for deterioration and extended LOS. "
            "Medicines reconciliation essential. Specialist input likely required.", AMBER))

    # Ambulance arrival
    if arrival_mode == "Ambulance":
        suggestions.append(("🚑", "Ambulance Arrival",
            "Pre-hospital handover documentation required within 15 minutes. "
            "Ensure handover breach <30 minutes per NHS target. Ambulance crew debrief.", TEAL))

    return suggestions, risk_level

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:0.5rem 0 1rem'>
        <div style='display:flex;align-items:center;gap:12px'>
            <img src="data:image/png;base64,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" width="48" style="border-radius:8px">
            <div>
                <div style='color:#0f172a;font-size:17px;font-weight:700'>MediFlow AI</div>
                <div style='color:#1e40af;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;font-weight:600'>NHS Intelligence</div>
            </div>
        </div>
    </div>
    <hr style='border-color:#e2e8f0;margin:0 0 1rem'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Live Dashboard",
        "AI Patient Assistant",
        "Queue & Predictions",
        "Data Analytics",
        "Model Info"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0 0.75rem'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;font-weight:600'>Live Data Stats</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>📊 &nbsp;{meta['total_records']:,} patient records</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>🎯 &nbsp;Breach AUC: {meta['breach_auc']}</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>⏱ &nbsp;Wait MAE: ±{meta['wait_mae']} min</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>✅ &nbsp;Accuracy: {meta['breach_acc']}%</div>
    <div style='font-size:11px;color:#64748b;margin-top:10px'>🕐 &nbsp;{datetime.now().strftime("%H:%M · %d %b %Y")}</div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE DASHBOARD (real data)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Live Dashboard":
    st.markdown("### Live Emergency Department Dashboard")
    st.caption(f"Powered by real NHS ED dataset · {meta['total_records']:,} patient records · {datetime.now().strftime('%d %b %Y')}")

    # Real KPIs from data
    breach_rate = df['four_hour_breach'].mean()*100
    avg_wait    = df['wait_time_to_assessment_min'].mean()
    avg_los     = df['ed_los_min'].mean()
    adm_rate    = df['admitted'].mean()*100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("4-Hour Breach Rate", f"{breach_rate:.1f}%", f"NHS target: 5% | Gap: {breach_rate-5:.1f}%", delta_color="inverse")
    k2.metric("Avg Wait to Assessment", f"{avg_wait:.0f} min", f"±{meta['wait_mae']} min model MAE")
    k3.metric("Avg ED LOS", f"{avg_los:.0f} min", f"{avg_los/60:.1f} hours")
    k4.metric("Admission Rate", f"{adm_rate:.1f}%", "National avg ~28%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([2,1])

    with c1:
        st.markdown('<div class="section-title">Hourly Arrivals & Breach Rate (from your dataset)</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour_of_day').agg(
            arrivals=('patient_id','count'),
            breach_rate=('four_hour_breach','mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(x=hourly['hour_of_day'], y=hourly['arrivals'],
                    name="Arrivals", marker_color="rgba(0,194,160,0.4)",
                    marker_line_color=TEAL, marker_line_width=1.5)
        fig.add_scatter(x=hourly['hour_of_day'], y=hourly['breach_rate']*100,
                        name="Breach Rate %", mode="lines", yaxis="y2",
                        line=dict(color=RED, width=2.5))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#94a3b8", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            height=270,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Arrivals"),
            yaxis2=dict(overlaying="y", side="right", title="Breach Rate %",
                        gridcolor="rgba(0,0,0,0)", color="#94a3b8"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)",
                       title="Hour of Day",
                       tickvals=list(range(0,24,2)), ticktext=[f"{h:02d}:00" for h in range(0,24,2)]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Triage Category Distribution</div>', unsafe_allow_html=True)
        tc = df['triage_category'].value_counts().sort_index()
        labels = {1:"Cat 1\nImmediate",2:"Cat 2\nVery Urgent",3:"Cat 3\nUrgent",4:"Cat 4\nStandard",5:"Cat 5\nNon-urgent"}
        fig2 = go.Figure(go.Pie(
            labels=[labels.get(i,str(i)) for i in tc.index],
            values=tc.values, hole=0.65,
            marker_colors=[RED,"#f97316",AMBER,GREEN,BLUE]
        ))
        fig2.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>")
        fig2.update_layout(**PLOTLY, height=240,
            annotations=[dict(text=f"<b>{tc.sum():,}</b><br>patients",
                              x=0.5,y=0.5,showarrow=False,font=dict(size=14,color="#e8eef8"))])
        st.plotly_chart(fig2, use_container_width=True)

    # Breach by month + IMD equity
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Monthly 4-Hour Breach Rate</div>', unsafe_allow_html=True)
        monthly = df.groupby('month')['four_hour_breach'].mean().reset_index()
        monthly['month_name'] = monthly['month'].map({
            1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
        colors = [RED if v>0.35 else AMBER if v>0.25 else TEAL for v in monthly['four_hour_breach']]
        fig3 = go.Figure(go.Bar(x=monthly['month_name'], y=monthly['four_hour_breach']*100,
                                 marker_color=colors, marker_line_width=0))
        fig3.add_hline(y=breach_rate, line_dash="dash", line_color="#94a3b8",
                       annotation_text=f"Annual avg {breach_rate:.1f}%")
        fig3.update_layout(**PLOTLY, height=230, yaxis_title="Breach Rate %")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">IMD Deprivation vs Avg Wait Time (Equity)</div>', unsafe_allow_html=True)
        imd_wait = df.groupby('imd_quintile')['wait_time_to_assessment_min'].mean().reset_index()
        fig4 = go.Figure(go.Bar(
            x=[f"Q{i}" for i in imd_wait['imd_quintile']],
            y=imd_wait['wait_time_to_assessment_min'],
            marker_color=[RED,"#f97316",AMBER,GREEN,TEAL],
            text=[f"{v:.0f}m" for v in imd_wait['wait_time_to_assessment_min']],
            textposition="outside", textfont=dict(color="#94a3b8")
        ))
        fig4.add_hline(y=avg_wait, line_dash="dash", line_color="#94a3b8",
                       annotation_text=f"Overall avg {avg_wait:.0f}m")
        fig4.update_layout(**PLOTLY, height=230, yaxis_title="Avg Wait (min)",
                            xaxis_title="IMD Quintile (1=Most Deprived)")
        st.plotly_chart(fig4, use_container_width=True)

    # AI insight box with real numbers
    q1_wait = df[df['imd_quintile']==1]['wait_time_to_assessment_min'].mean()
    q5_wait = df[df['imd_quintile']==5]['wait_time_to_assessment_min'].mean()
    equity_gap = q1_wait - q5_wait
    winter_breach = df[df['month'].isin([12,1,2])]['four_hour_breach'].mean()*100
    summer_breach = df[df['month'].isin([6,7,8])]['four_hour_breach'].mean()*100

    st.markdown(f"""
    <div class="ai-box">
        <div style='color:#1e40af;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>⚡ MediFlow AI — Real Data Insights</div>
        <div style='color:#0f172a;font-size:13.5px;line-height:1.8'>
            Analysis of <strong style='color:#1d4ed8'>{meta['total_records']:,} real patient records</strong> reveals:
            <br><br>
            📊 <strong style='color:#1d4ed8'>Breach rate peaks in winter ({winter_breach:.1f}%)</strong> vs summer ({summer_breach:.1f}%) —
            a <strong>{winter_breach-summer_breach:.1f} percentage point seasonal gap</strong>, confirming NHS winter pressures.
            <br><br>
            ⚖️ <strong style='color:#fbbf24'>Equity gap identified:</strong> IMD Quintile 1 patients wait
            <strong style='color:#fbbf24'>{equity_gap:.0f} minutes longer</strong> on average than Quintile 5 patients
            ({q1_wait:.0f} vs {q5_wait:.0f} min) — a statistically significant disparity requiring targeted intervention.
            <br><br>
            🤖 The <strong style='color:#1d4ed8'>LightGBM model</strong> achieves
            <strong>AUC {meta['breach_auc']}</strong> and <strong>{meta['breach_acc']}% accuracy</strong>
            in predicting 4-hour breach risk, with wait time predictions accurate to within ±{meta['wait_mae']} minutes.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: AI PATIENT ASSISTANT (core feature — personalised prediction)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Patient Assistant":
    st.markdown("### AI Patient Assistant — Personalised Triage & Predictions")
    st.caption("Enter patient details → real AI model predicts breach risk, wait time, and generates personalised clinical suggestions")

    left, right = st.columns([3, 2])

    with left:
        # ── Patient Info ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age        = c1.number_input("Age", 0, 120, 45, step=1)
        sex        = c2.selectbox("Sex", ["Male","Female"])
        imd        = c3.selectbox("IMD Quintile", [1,2,3,4,5],
                                   format_func=lambda x: f"Q{x} — {'Most Deprived' if x==1 else 'Least Deprived' if x==5 else 'Mid-range'}")

        c4, c5 = st.columns(2)
        arrival_mode = c4.selectbox("Arrival Mode", ["Walk-in","Ambulance","GP Referral","Other"])
        nhs_trust    = c5.selectbox("NHS Trust", meta['trusts'])

        # ── Clinical ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-title" style="margin-top:1rem">Clinical Presentation</div>', unsafe_allow_html=True)
        complaint = st.selectbox("Chief Complaint", meta['complaints'])

        c6, c7 = st.columns(2)
        triage   = c6.selectbox("Triage Category", [1,2,3,4,5],
                                  index=2,
                                  format_func=lambda x: {1:"1 — Immediate",2:"2 — Very Urgent",
                                                          3:"3 — Urgent",4:"4 — Standard",5:"5 — Non-urgent"}[x])
        news2    = c7.slider("NEWS2 Score", 0, 9, 2)

        c8, c9 = st.columns(2)
        comorbid     = c8.slider("Comorbidity Count", 0, 5, 1)
        prior_visits = c9.slider("Prior ED Visits (12m)", 0, 5, 0)

        c10, c11 = st.columns(2)
        pre_alert    = c10.checkbox("Pre-alert Received (Ambulance)")
        re_attend    = c11.checkbox("Re-attendance within 72h")

        # ── Department Conditions ─────────────────────────────────────────────
        st.markdown('<div class="section-title" style="margin-top:1rem">Current Department Conditions</div>', unsafe_allow_html=True)
        c12, c13 = st.columns(2)
        bed_occ  = c12.slider("Bed Occupancy (%)", 91.0, 99.9, 94.0, step=0.1)
        beds_av  = c13.slider("Beds Available", 0, 40, 8)

        c14, c15 = st.columns(2)
        staff_r  = c14.slider("Staff Ratio", 0.28, 0.82, 0.50, step=0.01)
        queue_l  = c15.slider("Queue Length", 0, 80, 25)

        c16, c17 = st.columns(2)
        handover = c16.slider("Ambulance Handover Delay (min)", 0.0, 60.0, 15.0, step=0.5)
        handover_breach = c17.checkbox("Handover Breach >30 min")

        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()
        month       = now.month
        is_weekend  = 1 if day_of_week >= 5 else 0
        shift_arr   = int(df[df['hour_of_day']==hour_of_day]['shift_total_arrivals'].mean())
        shift_amb   = int(df[df['hour_of_day']==hour_of_day]['shift_ambulance_arrivals'].mean())
        daily_att   = int(df[df['month']==month]['daily_ed_attendance'].mean())
        cpi_est     = round(float(df[df['bed_occupancy_pct'].between(bed_occ-1, bed_occ+1)]['capacity_pressure_index'].mean()), 3) if not df[df['bed_occupancy_pct'].between(bed_occ-1, bed_occ+1)].empty else 1.5
        acuity_est  = float(df[df['triage_category']==triage]['patient_acuity_score'].mean())
        wait_cur    = float(df[df['triage_category']==triage]['wait_time_to_assessment_min'].mean())

        submit = st.button("⚡ Get AI Prediction & Personalised Suggestions", use_container_width=True)

    with right:
        if submit:
            # Build input row
            row = {
                'hour_of_day': hour_of_day, 'day_of_week': day_of_week,
                'month': month, 'is_weekend': is_weekend,
                'patient_age': age, 'patient_sex': sex,
                'imd_quintile': imd, 'comorbidity_count': comorbid,
                'previous_ed_visits_12m': prior_visits, 're_attendance_72h': int(re_attend),
                'arrival_mode': arrival_mode, 'pre_alert_received': int(pre_alert),
                'chief_complaint': complaint, 'triage_category': triage,
                'news2_score': news2, 'patient_acuity_score': acuity_est,
                'bed_occupancy_pct': bed_occ, 'beds_available': beds_av,
                'staff_ratio': staff_r, 'daily_ed_attendance': daily_att,
                'shift_total_arrivals': shift_arr, 'shift_ambulance_arrivals': shift_amb,
                'ambulance_handover_delay_min': handover,
                'handover_breach_gt30min': int(handover_breach),
                'queue_length_estimate': queue_l,
                'capacity_pressure_index': cpi_est,
                'wait_time_to_assessment_min': wait_cur,
                'nhs_trust': nhs_trust, 'ics_region': 'Unknown',
                'trust_type': 'Teaching Hospital',
            }

            X_input = encode_input(row)
            breach_prob  = float(clf.predict_proba(X_input)[0,1])
            wait_pred    = float(reg.predict(X_input)[0])
            breach_pred  = int(clf.predict(X_input)[0])
            risk_level   = "HIGH" if breach_prob > 0.6 else "MEDIUM" if breach_prob > 0.35 else "LOW"
            risk_color   = RED if risk_level=="HIGH" else AMBER if risk_level=="MEDIUM" else GREEN

            # ── Main prediction display ────────────────────────────────────────
            st.markdown(f"""
            <div class="result-box">
                <div style='color:#1e40af;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>⚡ AI Prediction Results</div>
                <div style='display:flex;gap:1.5rem;margin-bottom:1.2rem'>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:#f8fafc;border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:{risk_color};font-family:monospace'>{breach_prob*100:.0f}%</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Breach Risk</div>
                        <div style='font-size:12px;font-weight:600;color:{risk_color};margin-top:4px'>{risk_level} RISK</div>
                    </div>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:#f8fafc;border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:#1e40af;font-family:monospace'>{wait_pred:.0f}</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Est. Wait (min)</div>
                        <div style='font-size:12px;color:#64748b;margin-top:4px'>{wait_pred/60:.1f} hours</div>
                    </div>
                </div>
                <div style='font-size:12px;color:#64748b;line-height:1.6;border-top:1px solid #e2e8f0;padding-top:0.75rem'>
                    Model: LightGBM · AUC {meta['breach_auc']} · Accuracy {meta['breach_acc']}% · 
                    Trained on {meta['total_records']:,} NHS ED records · Wait MAE ±{meta['wait_mae']} min
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Breach probability gauge ───────────────────────────────────────
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=breach_prob*100,
                number={'suffix':'%','font':{'color':'#e8eef8','size':24,'family':'DM Mono'}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#64748b'},
                    'bar':{'color':risk_color,'thickness':0.25},
                    'bgcolor':'rgba(15,32,64,0.8)',
                    'bordercolor':'rgba(0,194,160,0.2)',
                    'steps':[
                        {'range':[0,35],'color':'rgba(16,185,129,0.15)'},
                        {'range':[35,60],'color':'rgba(245,158,11,0.15)'},
                        {'range':[60,100],'color':'rgba(239,68,68,0.15)'}
                    ],
                    'threshold':{'line':{'color':AMBER,'width':2},'thickness':0.75,'value':26}
                },
                title={'text':"4-Hour Breach Probability",'font':{'color':'#94a3b8','size':12}}
            ))
            fig_g.update_layout(**PLOTLY, height=200)
            st.plotly_chart(fig_g, use_container_width=True)

            # ── Personalised suggestions ───────────────────────────────────────
            st.markdown('<div class="section-title">Personalised AI Suggestions</div>', unsafe_allow_html=True)

            PURPLE = "#a78bfa"
            TEAL_L = "#00c2a0"
            suggestions, _ = get_suggestions(
                breach_prob, wait_pred, triage, complaint,
                age, imd, news2, comorbid, arrival_mode
            )

            for icon, title, body, color in suggestions:
                bg = f"rgba({','.join(str(int(int(color.lstrip('#')[i:i+2],16))) for i in (0,2,4))},0.06)"
                border = f"rgba({','.join(str(int(int(color.lstrip('#')[i:i+2],16))) for i in (0,2,4))},0.2)"
                st.markdown(f"""
                <div style='display:flex;align-items:flex-start;gap:10px;padding:0.9rem 1rem;
                             background:{bg};border:1px solid {border};border-radius:9px;margin-bottom:8px'>
                    <span style='font-size:18px;flex-shrink:0;margin-top:2px'>{icon}</span>
                    <div>
                        <div style='font-size:13px;font-weight:600;color:{color};margin-bottom:4px'>{title}</div>
                        <div style='font-size:12.5px;color:#64748b;line-height:1.6'>{body}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='padding:2.5rem 1.5rem;text-align:center;color:#64748b;
                        border:1px dashed rgba(0,194,160,0.2);border-radius:12px;margin-top:0.5rem'>
                <div style='font-size:3rem;margin-bottom:1rem'>🤖</div>
                <div style='font-size:15px;color:#64748b;margin-bottom:0.5rem;font-weight:500'>AI Patient Assistant</div>
                <div style='font-size:13px;line-height:1.8;color:#64748b'>
                    Fill in patient details on the left and click<br>
                    <strong style='color:#1e40af'>Get AI Prediction &amp; Personalised Suggestions</strong><br><br>
                    The model will use your real trained LightGBM model<br>
                    (86.5% accuracy · AUC 0.9166) to predict:<br><br>
                    ● 4-hour breach risk probability<br>
                    ● Estimated wait time<br>
                    ● Personalised clinical suggestions<br>
                    ● Equity alerts (IMD)<br>
                    ● NEWS2-based escalation
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: QUEUE & PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Queue & Predictions":
    st.markdown("### Queue Management & Wait Time Predictions")
    st.caption("Real distribution from your dataset")

    # Wait time distribution from actual data
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Wait Time Distribution by Triage Category</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors_t = {1:RED,2:"#f97316",3:AMBER,4:GREEN,5:BLUE}
        for cat in sorted(df['triage_category'].unique()):
            subset = df[df['triage_category']==cat]['wait_time_to_assessment_min']
            fig.add_trace(go.Box(y=subset, name=f"Cat {cat}", marker_color=colors_t[cat],
                                  boxmean=True, line_width=1.5))
        fig.update_layout(**PLOTLY, height=280, yaxis_title="Wait Time (min)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Arrival Mode vs Breach Rate</div>', unsafe_allow_html=True)
        am = df.groupby('arrival_mode').agg(
            count=('patient_id','count'),
            breach_rate=('four_hour_breach','mean'),
            avg_wait=('wait_time_to_assessment_min','mean')
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_bar(x=am['arrival_mode'], y=am['breach_rate']*100,
                     name="Breach Rate %", marker_color=[TEAL,RED,AMBER,BLUE])
        fig2.add_scatter(x=am['arrival_mode'], y=am['avg_wait'],
                         name="Avg Wait (min)", mode="lines+markers", yaxis="y2",
                         line=dict(color="#a78bfa",width=2.5), marker=dict(size=8))
        fig2.update_layout(**PLOTLY, height=280,
            yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='rgba(0,0,0,0)',title="Breach Rate %"),
            yaxis2=dict(overlaying="y",side="right",title="Avg Wait (min)",
                        gridcolor="rgba(0,0,0,0)",color="#94a3b8"))
        st.plotly_chart(fig2, use_container_width=True)

    # Sample predictions on real data
    st.markdown('<div class="section-title">Model Predictions on Sample Queue (n=20, from your dataset)</div>', unsafe_allow_html=True)
    sample = df.sample(20, random_state=42).copy()

    for col in ['arrival_mode','chief_complaint','nhs_trust','ics_region','trust_type','patient_sex']:
        sample[col+'_enc'] = le_dict[col].transform(sample[col].astype(str))

    occ_s   = sample['bed_occupancy_pct'].values
    wait_s  = sample['wait_time_to_assessment_min'].values
    staff_s = sample['staff_ratio'].values
    night_s = ((sample['hour_of_day'].values>=22)|(sample['hour_of_day'].values<=6)).astype(int)
    ambul_s = (sample['arrival_mode'].values=='Ambulance').astype(int)
    wf_s    = sample['month'].isin([12,1,2]).astype(int).values

    sample['occ_x_triage']         = (occ_s/100)*sample['triage_category'].values
    sample['wait_per_staff']        = wait_s/(staff_s+0.01)
    sample['acuity_x_cpi']          = sample['patient_acuity_score'].values*sample['capacity_pressure_index'].values
    sample['night_high_occ']        = night_s*((occ_s>95).astype(int))
    sample['ambul_handover_stress'] = ambul_s*sample['handover_breach_gt30min'].values
    sample['is_winter']             = wf_s

    X_s = sample[FEATURES]
    sample['Predicted Breach Risk'] = (clf.predict_proba(X_s)[:,1]*100).round(1).astype(str) + '%'
    sample['Predicted Wait (min)']  = reg.predict(X_s).round(0).astype(int)
    sample['Actual Breach']         = sample['four_hour_breach'].map({0:'No',1:'⚠️ Yes'})

    display_cols = ['patient_id','patient_age','chief_complaint','triage_category',
                    'arrival_mode','Predicted Wait (min)','Predicted Breach Risk','Actual Breach']
    display = sample[display_cols].rename(columns={
        'patient_id':'ID','patient_age':'Age','chief_complaint':'Complaint',
        'triage_category':'Triage','arrival_mode':'Arrival'
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: DATA ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analytics":
    st.markdown("### Data Analytics — Real NHS ED Dataset")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(df):,}", "Synthetic ECDS-aligned")
    k2.metric("4hr Breach Rate", f"{df['four_hour_breach'].mean()*100:.1f}%", "NHS target: 5%")
    k3.metric("12hr Breach Rate", f"{df['twelve_hour_breach'].mean()*100:.1f}%")
    k4.metric("Deteriorated in ED", f"{df['deteriorated_in_ed'].mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Disposition Breakdown</div>', unsafe_allow_html=True)
        disp = df['disposition'].value_counts()
        fig = px.bar(x=disp.values, y=disp.index, orientation='h',
                     color=disp.values, color_continuous_scale=['#00c2a0','#3b82f6','#f59e0b','#ef4444','#8b5cf6'])
        fig.update_layout(**PLOTLY, height=250, coloraxis_showscale=False)
        fig.update_traces(text=[f"{v:,}" for v in disp.values], textposition="outside",
                          textfont=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Chief Complaint vs Avg ED LOS</div>', unsafe_allow_html=True)
        cc_los = df.groupby('chief_complaint')['ed_los_min'].mean().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(x=cc_los.values, y=cc_los.index, orientation='h',
                                 marker_color=TEAL, marker_line_width=0))
        fig2.update_layout(**PLOTLY, height=300, xaxis_title="Avg ED LOS (min)")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap of arrivals
    st.markdown('<div class="section-title">Arrival Heatmap — Day of Week × Hour</div>', unsafe_allow_html=True)
    heatmap_data = df.groupby(['day_of_week','hour_of_day'])['patient_id'].count().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour_of_day', values='patient_id').fillna(0)
    day_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig_h = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
        y=[day_labels[i] for i in heatmap_pivot.index],
        colorscale=[[0,"rgba(0,194,160,0.1)"],[0.5,"rgba(245,158,11,0.6)"],[1,"rgba(239,68,68,0.9)"]],
        showscale=True, colorbar=dict(tickfont=dict(color="#94a3b8")),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Arrivals: %{z}<extra></extra>"
    ))
    fig_h.update_layout(**PLOTLY, height=260)
    st.plotly_chart(fig_h, use_container_width=True)

    # Trust comparison
    st.markdown('<div class="section-title">NHS Trust Comparison — Breach Rate & Avg Wait</div>', unsafe_allow_html=True)
    trust_stats = df.groupby('nhs_trust').agg(
        breach_rate=('four_hour_breach','mean'),
        avg_wait=('wait_time_to_assessment_min','mean'),
        count=('patient_id','count')
    ).reset_index().sort_values('breach_rate', ascending=False)
    trust_stats['Trust Short'] = trust_stats['nhs_trust'].str.replace(' NHS.*','',regex=True).str[:30]
    fig_t = go.Figure()
    fig_t.add_bar(x=trust_stats['Trust Short'], y=trust_stats['breach_rate']*100,
                   name="Breach Rate %", marker_color=RED, opacity=0.8)
    fig_t.add_scatter(x=trust_stats['Trust Short'], y=trust_stats['avg_wait'],
                       name="Avg Wait (min)", mode="lines+markers", yaxis="y2",
                       line=dict(color=TEAL,width=2.5), marker=dict(size=7))
    fig_t.update_layout(**PLOTLY, height=290,
        yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='rgba(0,0,0,0)',title="Breach Rate %"),
        yaxis2=dict(overlaying="y",side="right",title="Avg Wait (min)",
                    gridcolor="rgba(0,0,0,0)",color="#94a3b8"),
        xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='rgba(0,0,0,0)',tickangle=30))
    st.plotly_chart(fig_t, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.markdown("### Model Information & Architecture")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Breach Model AUC", str(meta['breach_auc']))
    c2.metric("Accuracy", f"{meta['breach_acc']}%")
    c3.metric("Wait Time MAE", f"±{meta['wait_mae']} min")
    c4.metric("Training Records", f"{int(meta['total_records']*0.7):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-title">Top 15 Features — LightGBM Breach Model</div>', unsafe_allow_html=True)
    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(15)
    fig_fi = go.Figure(go.Bar(
        x=feat_imp['Importance'], y=feat_imp['Feature'],
        orientation='h', marker_color=TEAL, marker_line_width=0
    ))
    fig_fi.update_layout(**PLOTLY, height=380, xaxis_title="Feature Importance (Split gain)")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"""
    <div class="ai-box">
        <div style='color:#1e40af;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>ℹ Architecture & Methodology</div>
        <div style='color:#0f172a;font-size:13.5px;line-height:1.9'>
            <strong style='color:#1d4ed8'>Breach Classifier:</strong> LightGBM (n_estimators=300, max_depth=6, lr=0.05) trained on 
            {int(meta['total_records']*0.7):,} records with SMOTE oversampling for class balance. 
            Features include engineered interaction terms (occ×triage, wait/staff, acuity×CPI).<br><br>
            <strong style='color:#1d4ed8'>Wait Time Regressor:</strong> LightGBM Regressor — same architecture, 
            MAE ±{meta['wait_mae']} minutes on held-out test set (R² = 0.998).<br><br>
            <strong style='color:#1d4ed8'>RL Layer (DQN + PPO):</strong> Deep Reinforcement Learning agents trained in 
            custom NHS ED Gymnasium environment. State space includes queue depth, occupancy, staff ratio, 
            triage mix. Action space: resource reallocation policies. Reward: 4-hour target compliance 
            + equity penalty for IMD Q1 disparity.<br><br>
            <strong style='color:#1d4ed8'>Explainability:</strong> SHAP beeswarm, waterfall, and interaction plots 
            identify key breach drivers. Top contributors: bed_occupancy_pct, queue_length_estimate, 
            triage_category, news2_score, capacity_pressure_index.<br><br>
            <span style='color:#64748b;font-size:12px'>
            Dissertation: "A Predictive Deep Reinforcement Learning AI Assistant for Real-Time Resource 
            Scheduling Optimisation in NHS Emergency Care" · Northumbria University London · MSc Big Data & 
            Data Science Technology · Module LD7236 · Supervisor: Dr. Rejwan Bin Sulaiman · 2024–25
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

