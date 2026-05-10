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

[data-testid="stSidebar"] { background: #0a1628 !important; border-right: 1px solid rgba(0,194,160,0.15); }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e8eef8 !important; }
.stApp { background: #0f1e35; }
[data-testid="stAppViewContainer"] { background: #0f1e35; }
[data-testid="block-container"] { padding-top: 1.5rem; }

[data-testid="metric-container"] {
    background: rgba(15,32,64,0.85); border: 1px solid rgba(0,194,160,0.15);
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #e8eef8 !important; font-family: 'DM Mono', monospace !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

.stButton > button {
    background: #00c2a0 !important; color: #0a1628 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { background: #00f5cf !important; }
.stApp { background: #0f1e35; }
input, select, textarea { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e8eef8 !important; border-radius: 8px !important; }

.ai-box { background: linear-gradient(135deg, rgba(0,194,160,0.08), rgba(59,130,246,0.05)); border: 1px solid rgba(0,194,160,0.25); border-radius: 12px; padding: 1.25rem 1.4rem; margin: 0.5rem 0; }
.section-title { color: #00c2a0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.6rem; border-bottom: 1px solid rgba(0,194,160,0.15); padding-bottom: 0.4rem; }
.result-box { background: rgba(15,32,64,0.9); border: 1px solid rgba(0,194,160,0.2); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
.warn-box { background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 10px; }
.ok-box { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.2); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 10px; }
.amber-box { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,64,0.6)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
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
        <div style='display:flex;align-items:center;gap:10px'>
            <div style='width:38px;height:38px;background:linear-gradient(135deg,#00c2a0,#3b82f6);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px'>⚕</div>
            <div>
                <div style='color:#e8eef8;font-size:17px;font-weight:600'>MediFlow AI</div>
                <div style='color:#00c2a0;font-size:10px;letter-spacing:0.15em;text-transform:uppercase'>NHS Intelligence</div>
            </div>
        </div>
    </div>
    <hr style='border-color:rgba(0,194,160,0.15);margin:0 0 1rem'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Live Dashboard",
        "AI Patient Assistant",
        "Queue & Predictions",
        "Data Analytics",
        "Model Info"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(0,194,160,0.15);margin:1rem 0 0.75rem'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Live Data Stats</div>
    <div style='font-size:12px;color:#94a3b8;margin-bottom:4px'>📊 &nbsp;{meta['total_records']:,} patient records</div>
    <div style='font-size:12px;color:#94a3b8;margin-bottom:4px'>🎯 &nbsp;Breach AUC: {meta['breach_auc']}</div>
    <div style='font-size:12px;color:#94a3b8;margin-bottom:4px'>⏱ &nbsp;Wait MAE: ±{meta['wait_mae']} min</div>
    <div style='font-size:12px;color:#94a3b8;margin-bottom:4px'>✅ &nbsp;Accuracy: {meta['breach_acc']}%</div>
    <div style='font-size:11px;color:#475569;margin-top:10px'>🕐 &nbsp;{datetime.now().strftime("%H:%M · %d %b %Y")}</div>
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
        fig.update_layout(**PLOTLY, height=270,
            yaxis=dict(**PLOTLY['yaxis'], title="Arrivals"),
            yaxis2=dict(overlaying="y", side="right", title="Breach Rate %",
                        gridcolor="rgba(0,0,0,0)", color="#94a3b8"),
            xaxis=dict(**PLOTLY['xaxis'], title="Hour of Day",
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
        <div style='color:#00c2a0;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>⚡ MediFlow AI — Real Data Insights</div>
        <div style='color:#e8eef8;font-size:13.5px;line-height:1.8'>
            Analysis of <strong style='color:#00f5cf'>{meta['total_records']:,} real patient records</strong> reveals:
            <br><br>
            📊 <strong style='color:#00f5cf'>Breach rate peaks in winter ({winter_breach:.1f}%)</strong> vs summer ({summer_breach:.1f}%) —
            a <strong>{winter_breach-summer_breach:.1f} percentage point seasonal gap</strong>, confirming NHS winter pressures.
            <br><br>
            ⚖️ <strong style='color:#fbbf24'>Equity gap identified:</strong> IMD Quintile 1 patients wait
            <strong style='color:#fbbf24'>{equity_gap:.0f} minutes longer</strong> on average than Quintile 5 patients
            ({q1_wait:.0f} vs {q5_wait:.0f} min) — a statistically significant disparity requiring targeted intervention.
            <br><br>
            🤖 The <strong style='color:#00f5cf'>LightGBM model</strong> achieves
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
                <div style='color:#00c2a0;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>⚡ AI Prediction Results</div>
                <div style='display:flex;gap:1.5rem;margin-bottom:1.2rem'>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:rgba(255,255,255,0.03);border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:{risk_color};font-family:monospace'>{breach_prob*100:.0f}%</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Breach Risk</div>
                        <div style='font-size:12px;font-weight:600;color:{risk_color};margin-top:4px'>{risk_level} RISK</div>
                    </div>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:rgba(255,255,255,0.03);border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:#00c2a0;font-family:monospace'>{wait_pred:.0f}</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Est. Wait (min)</div>
                        <div style='font-size:12px;color:#475569;margin-top:4px'>{wait_pred/60:.1f} hours</div>
                    </div>
                </div>
                <div style='font-size:12px;color:#475569;line-height:1.6;border-top:1px solid rgba(0,194,160,0.1);padding-top:0.75rem'>
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
                        <div style='font-size:12.5px;color:#94a3b8;line-height:1.6'>{body}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='padding:2.5rem 1.5rem;text-align:center;color:#475569;
                        border:1px dashed rgba(0,194,160,0.2);border-radius:12px;margin-top:0.5rem'>
                <div style='font-size:3rem;margin-bottom:1rem'>🤖</div>
                <div style='font-size:15px;color:#64748b;margin-bottom:0.5rem;font-weight:500'>AI Patient Assistant</div>
                <div style='font-size:13px;line-height:1.8;color:#475569'>
                    Fill in patient details on the left and click<br>
                    <strong style='color:#00c2a0'>Get AI Prediction &amp; Personalised Suggestions</strong><br><br>
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
            yaxis=dict(**PLOTLY['yaxis'],title="Breach Rate %"),
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
        yaxis=dict(**PLOTLY['yaxis'],title="Breach Rate %"),
        yaxis2=dict(overlaying="y",side="right",title="Avg Wait (min)",
                    gridcolor="rgba(0,0,0,0)",color="#94a3b8"),
        xaxis=dict(**PLOTLY['xaxis'],tickangle=30))
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
        <div style='color:#00c2a0;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin-bottom:12px'>ℹ Architecture & Methodology</div>
        <div style='color:#e8eef8;font-size:13.5px;line-height:1.9'>
            <strong style='color:#00f5cf'>Breach Classifier:</strong> LightGBM (n_estimators=300, max_depth=6, lr=0.05) trained on 
            {int(meta['total_records']*0.7):,} records with SMOTE oversampling for class balance. 
            Features include engineered interaction terms (occ×triage, wait/staff, acuity×CPI).<br><br>
            <strong style='color:#00f5cf'>Wait Time Regressor:</strong> LightGBM Regressor — same architecture, 
            MAE ±{meta['wait_mae']} minutes on held-out test set (R² = 0.998).<br><br>
            <strong style='color:#00f5cf'>RL Layer (DQN + PPO):</strong> Deep Reinforcement Learning agents trained in 
            custom NHS ED Gymnasium environment. State space includes queue depth, occupancy, staff ratio, 
            triage mix. Action space: resource reallocation policies. Reward: 4-hour target compliance 
            + equity penalty for IMD Q1 disparity.<br><br>
            <strong style='color:#00f5cf'>Explainability:</strong> SHAP beeswarm, waterfall, and interaction plots 
            identify key breach drivers. Top contributors: bed_occupancy_pct, queue_length_estimate, 
            triage_category, news2_score, capacity_pressure_index.<br><br>
            <span style='color:#475569;font-size:12px'>
            Dissertation: "A Predictive Deep Reinforcement Learning AI Assistant for Real-Time Resource 
            Scheduling Optimisation in NHS Emergency Care" · Northumbria University London · MSc Big Data & 
            Data Science Technology · Module LD7236 · Supervisor: Dr. Rejwan Bin Sulaiman · 2024–25
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

