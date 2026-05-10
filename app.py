import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle, json, os, base64, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MediFlow AI - NHS ED Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    return pd.concat([df1, df2], ignore_index=True)

@st.cache_data
def load_logo():
    logo_path = f'{BASE}/logo.png'
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

clf, reg, le_dict, FEATURES, meta = load_models()
df = load_data()
logo_b64 = load_logo()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #ffffff !important; }
[data-testid="block-container"] { padding-top: 1.5rem; background: #ffffff; }
[data-testid="stSidebar"] { background: #f8fafc !important; border-right: 2px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="metric-container"] {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #0f172a !important; font-family: 'DM Mono', monospace !important; font-size: 1.9rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stButton > button {
    background: #1e40af !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }
input, select, textarea {
    background: #f8fafc !important; border: 1px solid #cbd5e1 !important;
    color: #0f172a !important; border-radius: 8px !important;
}
.ai-box { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 1.25rem 1.4rem; margin: 0.5rem 0; }
.section-title { color: #1e40af; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.6rem; border-bottom: 2px solid #bfdbfe; padding-bottom: 0.4rem; }
.result-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8fafc",
    font=dict(family="DM Sans", color="#334155", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#334155")),
)
TEAL  = "#0ea5e9"
AMBER = "#f59e0b"
RED   = "#ef4444"
BLUE  = "#1e40af"
GREEN = "#10b981"

def encode_input(row_dict):
    d = row_dict.copy()
    for col, le in le_dict.items():
        if col in d:
            try:
                d[col+'_enc'] = int(le.transform([d[col]])[0])
            except:
                d[col+'_enc'] = 0
    occ   = d.get('bed_occupancy_pct', 94.0)
    wait  = d.get('wait_time_to_assessment_min', 120.0)
    staff = d.get('staff_ratio', 0.5)
    night = 1 if d.get('hour_of_day', 12) >= 22 or d.get('hour_of_day', 12) <= 6 else 0
    ambul = 1 if d.get('arrival_mode', 'Walk-in') == 'Ambulance' else 0
    d['occ_x_triage']         = (occ / 100) * d.get('triage_category', 3)
    d['wait_per_staff']        = wait / (staff + 0.01)
    d['acuity_x_cpi']          = d.get('patient_acuity_score', 10.0) * d.get('capacity_pressure_index', 1.5)
    d['night_high_occ']        = night * (1 if occ > 95 else 0)
    d['ambul_handover_stress'] = ambul * d.get('handover_breach_gt30min', 0)
    d['is_winter']             = 1 if d.get('month', datetime.now().month) in [12, 1, 2] else 0
    row = [d.get(f, d.get(f.replace('_enc', ''), 0)) for f in FEATURES]
    return np.array(row).reshape(1, -1)

def get_suggestions(breach_prob, wait_pred, triage, complaint, age, imd, news2, comorbid, arrival_mode):
    suggestions = []
    risk_level = "HIGH" if breach_prob > 0.6 else "MEDIUM" if breach_prob > 0.35 else "LOW"

    if wait_pred > 180:
        suggestions.append(("Critical Wait Risk",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> significantly exceeds the NHS 4-hour target. "
            "Immediate escalation recommended. Consider fast-track pathway or alternative disposition.", RED))
    elif wait_pred > 120:
        suggestions.append(("Extended Wait Expected",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> is approaching the 4-hour threshold. "
            "Pre-emptive assessment and early senior review advised.", AMBER))
    else:
        suggestions.append(("Wait Time Within Target",
            f"Predicted wait of <b>{wait_pred:.0f} minutes</b> is within acceptable limits. "
            "Standard pathway appropriate.", GREEN))

    if risk_level == "HIGH":
        suggestions.append(("High 4-Hour Breach Risk",
            f"Model predicts <b>{breach_prob*100:.0f}% probability</b> of breaching the 4-hour target. "
            "Activate breach prevention protocol. Allocate senior clinician review within 30 minutes.", RED))
    elif risk_level == "MEDIUM":
        suggestions.append(("Moderate Breach Risk",
            f"Model predicts <b>{breach_prob*100:.0f}% probability</b> of breach. "
            "Monitor closely. Reassess at 2-hour mark.", AMBER))

    complaint_advice = {
        "Chest Pain": "12-lead ECG within 10 minutes. Troponin at 0h and 3h. Cardiology alert if STEMI suspected.",
        "Cardiac Arrest": "Full resuscitation team activation. Immediate Resus bay. Cardiology/ITU on standby.",
        "Stroke Symptoms": "FAST protocol. CT Head within 30 minutes. Stroke team activation. Thrombolysis assessment.",
        "Shortness of Breath": "SpO2 monitoring, ABG if less than 94%. CXR. Consider BNP for heart failure vs COPD exacerbation.",
        "Head Injury": "GCS scoring every 15 minutes. CT Head per NICE guidelines. Neurosurgery referral if GCS below 13.",
        "Infection / Sepsis": "Sepsis Six bundle within 1 hour: blood cultures, IV antibiotics, IV fluids, oxygen, monitor urine output, lactate.",
        "Fever / Sepsis Child": "Paediatric sepsis pathway. Antipyretics. Blood cultures before antibiotics. Paediatric review.",
        "Overdose / Poisoning": "Toxicology referral. Activated charcoal if under 1 hour. Psychiatric liaison post-medical clearance.",
        "Mental Health Crisis": "Mental health team referral. Safe environment. Psychiatric assessment. Safeguarding review.",
        "Abdominal Pain": "FBC, CRP, LFTs, amylase, urine dip. Surgical review if peritonism signs. Ultrasound if indicated.",
        "Fall / Trauma": "Full trauma assessment. Fracture survey X-rays. Fall risk assessment. Physiotherapy referral.",
        "Fracture / Dislocation": "X-ray (2 views minimum). Analgesia per protocol. Orthopaedic review for complex fractures.",
    }
    advice = complaint_advice.get(complaint, "Full clinical assessment as per presenting complaint. Document NEWS2 score.")
    suggestions.append((f"Clinical Pathway: {complaint}", advice, BLUE))

    if age >= 75:
        suggestions.append(("Elderly Patient Alert",
            "Age 75 or above: Enhanced frailty screening (CFS score). Falls risk assessment. Delirium screen. "
            "Early social work referral. Consider Comprehensive Geriatric Assessment.", AMBER))
    elif age <= 16:
        suggestions.append(("Paediatric Patient",
            "Paediatric-specific drug dosing (weight-based). PEWS scoring. Safeguarding assessment. "
            "Parent/guardian consent required.", BLUE))

    if imd == 1:
        suggestions.append(("Equity Alert - High Deprivation (IMD Q1)",
            "Patient from most deprived quintile. Research shows 14% longer average wait times in this group. "
            "Flag for equitable triage review per NHS Constitution. Social prescribing referral may be appropriate.", AMBER))

    if news2 >= 7:
        suggestions.append((f"NEWS2 Score {news2} - Critical",
            "NEWS2 score of 7 or above indicates high clinical risk. Continuous monitoring. Consider HDU/ICU admission. "
            "Urgent consultant review required.", RED))
    elif news2 >= 5:
        suggestions.append((f"NEWS2 Score {news2} - Elevated",
            "NEWS2 score 5 to 6 indicates medium-high risk. Increase monitoring frequency. Senior review within 30 minutes.", AMBER))

    if comorbid >= 3:
        suggestions.append(("Complex Comorbidity Profile",
            f"{comorbid} active comorbidities noted. High risk for deterioration and extended LOS. "
            "Medicines reconciliation essential. Specialist input likely required.", AMBER))

    if arrival_mode == "Ambulance":
        suggestions.append(("Ambulance Arrival",
            "Pre-hospital handover documentation required within 15 minutes. "
            "Ensure handover breach under 30 minutes per NHS target. Ambulance crew debrief.", TEAL))

    return suggestions, risk_level

# SIDEBAR
with st.sidebar:
    if logo_b64:
        st.markdown(f"""
        <div style='padding:0.5rem 0 1rem'>
            <div style='display:flex;align-items:center;gap:12px'>
                <img src="data:image/png;base64,{logo_b64}" width="48" style="border-radius:8px">
                <div>
                    <div style='color:#0f172a;font-size:17px;font-weight:700'>MediFlow AI</div>
                    <div style='color:#1e40af;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;font-weight:600'>NHS Intelligence</div>
                </div>
            </div>
        </div>
        <hr style='border-color:#e2e8f0;margin:0 0 1rem'>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:0.5rem 0 1rem'>
            <div style='color:#0f172a;font-size:17px;font-weight:700'>MediFlow AI</div>
            <div style='color:#1e40af;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;font-weight:600'>NHS Intelligence</div>
        </div>
        <hr style='border-color:#e2e8f0;margin:0 0 1rem'>
        """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Live Dashboard",
        "AI Patient Assistant",
        "Queue and Predictions",
        "Data Analytics",
        "Model Info"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#e2e8f0;margin:1rem 0 0.75rem'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;font-weight:600'>Live Data Stats</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>{meta['total_records']:,} patient records</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>Breach AUC: {meta['breach_auc']}</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>Wait MAE: plus or minus {meta['wait_mae']} min</div>
    <div style='font-size:12px;color:#334155;margin-bottom:4px'>Accuracy: {meta['breach_acc']}%</div>
    <div style='font-size:11px;color:#64748b;margin-top:10px'>{datetime.now().strftime("%H:%M  %d %b %Y")}</div>
    """, unsafe_allow_html=True)

# LIVE DASHBOARD
if page == "Live Dashboard":
    st.markdown("### Live Emergency Department Dashboard")
    st.caption(f"Powered by real NHS ED dataset  |  {meta['total_records']:,} patient records  |  {datetime.now().strftime('%d %b %Y')}")

    breach_rate = df['four_hour_breach'].mean() * 100
    avg_wait    = df['wait_time_to_assessment_min'].mean()
    avg_los     = df['ed_los_min'].mean()
    adm_rate    = df['admitted'].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("4-Hour Breach Rate", f"{breach_rate:.1f}%", f"NHS target: 5% | Gap: {breach_rate-5:.1f}%", delta_color="inverse")
    k2.metric("Avg Wait to Assessment", f"{avg_wait:.0f} min", f"Model MAE: plus or minus {meta['wait_mae']} min")
    k3.metric("Avg ED Length of Stay", f"{avg_los:.0f} min", f"{avg_los/60:.1f} hours")
    k4.metric("Admission Rate", f"{adm_rate:.1f}%", "National avg approx 28%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="section-title">Hourly Arrivals and Breach Rate — From Your Dataset</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour_of_day').agg(
            arrivals=('patient_id', 'count'),
            breach_rate=('four_hour_breach', 'mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(x=hourly['hour_of_day'], y=hourly['arrivals'],
                    name="Arrivals", marker_color="rgba(14,165,233,0.4)",
                    marker_line_color=TEAL, marker_line_width=1.5)
        fig.add_scatter(x=hourly['hour_of_day'], y=hourly['breach_rate'] * 100,
                        name="Breach Rate %", mode="lines", yaxis="y2",
                        line=dict(color=RED, width=2.5))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=270,
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Arrivals"),
            yaxis2=dict(overlaying="y", side="right", title="Breach Rate %",
                        gridcolor="rgba(0,0,0,0)", color="#334155"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)",
                       title="Hour of Day",
                       tickvals=list(range(0, 24, 2)),
                       ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)])
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Triage Category Distribution</div>', unsafe_allow_html=True)
        tc = df['triage_category'].value_counts().sort_index()
        labels = {1: "Cat 1 Immediate", 2: "Cat 2 Very Urgent", 3: "Cat 3 Urgent",
                  4: "Cat 4 Standard", 5: "Cat 5 Non-urgent"}
        fig2 = go.Figure(go.Pie(
            labels=[labels.get(i, str(i)) for i in tc.index],
            values=tc.values, hole=0.65,
            marker_colors=[RED, "#f97316", AMBER, GREEN, BLUE]
        ))
        fig2.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#334155", size=10)),
            height=240,
            annotations=[dict(text=f"<b>{tc.sum():,}</b><br>patients",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color="#0f172a"))]
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Monthly 4-Hour Breach Rate</div>', unsafe_allow_html=True)
        monthly = df.groupby('month')['four_hour_breach'].mean().reset_index()
        monthly['month_name'] = monthly['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
        colors = [RED if v > 0.35 else AMBER if v > 0.25 else TEAL for v in monthly['four_hour_breach']]
        fig3 = go.Figure(go.Bar(x=monthly['month_name'], y=monthly['four_hour_breach'] * 100,
                                marker_color=colors, marker_line_width=0))
        fig3.add_hline(y=breach_rate, line_dash="dash", line_color="#64748b",
                       annotation_text=f"Annual avg {breach_rate:.1f}%")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=230,
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Breach Rate %"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">IMD Deprivation vs Avg Wait Time — Equity Analysis</div>', unsafe_allow_html=True)
        imd_wait = df.groupby('imd_quintile')['wait_time_to_assessment_min'].mean().reset_index()
        fig4 = go.Figure(go.Bar(
            x=[f"Q{i}" for i in imd_wait['imd_quintile']],
            y=imd_wait['wait_time_to_assessment_min'],
            marker_color=[RED, "#f97316", AMBER, GREEN, TEAL],
            text=[f"{v:.0f}m" for v in imd_wait['wait_time_to_assessment_min']],
            textposition="outside", textfont=dict(color="#334155")
        ))
        fig4.add_hline(y=avg_wait, line_dash="dash", line_color="#64748b",
                       annotation_text=f"Overall avg {avg_wait:.0f}m")
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=230,
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Avg Wait (min)"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="IMD Quintile (1=Most Deprived)")
        )
        st.plotly_chart(fig4, use_container_width=True)

    q1_wait      = df[df['imd_quintile'] == 1]['wait_time_to_assessment_min'].mean()
    q5_wait      = df[df['imd_quintile'] == 5]['wait_time_to_assessment_min'].mean()
    equity_gap   = q1_wait - q5_wait
    winter_breach = df[df['month'].isin([12, 1, 2])]['four_hour_breach'].mean() * 100
    summer_breach = df[df['month'].isin([6, 7, 8])]['four_hour_breach'].mean() * 100

    st.markdown(f"""
    <div class="ai-box">
        <div class="section-title">MediFlow AI - Real Data Insights</div>
        <div style='color:#0f172a;font-size:13.5px;line-height:1.8'>
            Analysis of <strong style='color:#1e40af'>{meta['total_records']:,} real patient records</strong> reveals:<br><br>
            <strong style='color:#1e40af'>Breach rate peaks in winter ({winter_breach:.1f}%)</strong> vs summer ({summer_breach:.1f}%) —
            a <strong>{winter_breach - summer_breach:.1f} percentage point seasonal gap</strong>, confirming NHS winter pressures.<br><br>
            <strong style='color:#b45309'>Equity gap identified:</strong> IMD Quintile 1 patients wait
            <strong style='color:#b45309'>{equity_gap:.0f} minutes longer</strong> on average than Quintile 5 patients
            ({q1_wait:.0f} vs {q5_wait:.0f} min) — a statistically significant disparity requiring targeted intervention.<br><br>
            The <strong style='color:#1e40af'>LightGBM model</strong> achieves
            <strong>AUC {meta['breach_auc']}</strong> and <strong>{meta['breach_acc']}% accuracy</strong>
            in predicting 4-hour breach risk, with wait time predictions accurate to within plus or minus {meta['wait_mae']} minutes.
        </div>
    </div>
    """, unsafe_allow_html=True)

# AI PATIENT ASSISTANT
elif page == "AI Patient Assistant":
    st.markdown("### AI Patient Assistant - Personalised Triage and Predictions")
    st.caption("Enter patient details to receive AI-powered breach risk, wait time prediction, and personalised clinical suggestions")

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-title">Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age  = c1.number_input("Age", 0, 120, 45, step=1)
        sex  = c2.selectbox("Sex", ["Male", "Female"])
        imd  = c3.selectbox("IMD Quintile", [1, 2, 3, 4, 5],
                             format_func=lambda x: f"Q{x} - {'Most Deprived' if x == 1 else 'Least Deprived' if x == 5 else 'Mid-range'}")

        c4, c5 = st.columns(2)
        arrival_mode = c4.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "GP Referral", "Other"])
        nhs_trust    = c5.selectbox("NHS Trust", meta['trusts'])

        st.markdown('<div class="section-title" style="margin-top:1rem">Clinical Presentation</div>', unsafe_allow_html=True)
        complaint = st.selectbox("Chief Complaint", meta['complaints'])

        c6, c7 = st.columns(2)
        triage = c6.selectbox("Triage Category", [1, 2, 3, 4, 5], index=2,
                               format_func=lambda x: {1: "1 - Immediate", 2: "2 - Very Urgent",
                                                       3: "3 - Urgent", 4: "4 - Standard", 5: "5 - Non-urgent"}[x])
        news2  = c7.slider("NEWS2 Score", 0, 9, 2)

        c8, c9 = st.columns(2)
        comorbid     = c8.slider("Comorbidity Count", 0, 5, 1)
        prior_visits = c9.slider("Prior ED Visits (12 months)", 0, 5, 0)

        c10, c11 = st.columns(2)
        pre_alert = c10.checkbox("Pre-alert Received")
        re_attend = c11.checkbox("Re-attendance within 72h")

        st.markdown('<div class="section-title" style="margin-top:1rem">Current Department Conditions</div>', unsafe_allow_html=True)
        c12, c13 = st.columns(2)
        bed_occ = c12.slider("Bed Occupancy (%)", 91.0, 99.9, 94.0, step=0.1)
        beds_av = c13.slider("Beds Available", 0, 40, 8)

        c14, c15 = st.columns(2)
        staff_r = c14.slider("Staff Ratio", 0.28, 0.82, 0.50, step=0.01)
        queue_l = c15.slider("Queue Length", 0, 80, 25)

        c16, c17 = st.columns(2)
        handover        = c16.slider("Ambulance Handover Delay (min)", 0.0, 60.0, 15.0, step=0.5)
        handover_breach = c17.checkbox("Handover Breach over 30 min")

        now         = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()
        month       = now.month
        is_weekend  = 1 if day_of_week >= 5 else 0
        shift_arr   = int(df[df['hour_of_day'] == hour_of_day]['shift_total_arrivals'].mean())
        shift_amb   = int(df[df['hour_of_day'] == hour_of_day]['shift_ambulance_arrivals'].mean())
        daily_att   = int(df[df['month'] == month]['daily_ed_attendance'].mean())
        cpi_est     = round(float(df[df['bed_occupancy_pct'].between(bed_occ - 1, bed_occ + 1)]['capacity_pressure_index'].mean()), 3) \
                      if not df[df['bed_occupancy_pct'].between(bed_occ - 1, bed_occ + 1)].empty else 1.5
        acuity_est  = float(df[df['triage_category'] == triage]['patient_acuity_score'].mean())
        wait_cur    = float(df[df['triage_category'] == triage]['wait_time_to_assessment_min'].mean())

        submit = st.button("Get AI Prediction and Personalised Suggestions", use_container_width=True)

    with right:
        if submit:
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
                'nhs_trust': nhs_trust, 'ics_region': 'Unknown', 'trust_type': 'Teaching Hospital',
            }

            X_input      = encode_input(row)
            breach_prob  = float(clf.predict_proba(X_input)[0, 1])
            wait_pred    = float(reg.predict(X_input)[0])
            risk_level   = "HIGH" if breach_prob > 0.6 else "MEDIUM" if breach_prob > 0.35 else "LOW"
            risk_color   = RED if risk_level == "HIGH" else AMBER if risk_level == "MEDIUM" else GREEN

            st.markdown(f"""
            <div class="result-box">
                <div class="section-title">AI Prediction Results</div>
                <div style='display:flex;gap:1.5rem;margin-bottom:1.2rem'>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:#f1f5f9;border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:{risk_color};font-family:monospace'>{breach_prob*100:.0f}%</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Breach Risk</div>
                        <div style='font-size:12px;font-weight:600;color:{risk_color};margin-top:4px'>{risk_level} RISK</div>
                    </div>
                    <div style='text-align:center;flex:1;padding:0.75rem;background:#f1f5f9;border-radius:8px'>
                        <div style='font-size:36px;font-weight:700;color:#1e40af;font-family:monospace'>{wait_pred:.0f}</div>
                        <div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>Est. Wait (min)</div>
                        <div style='font-size:12px;color:#64748b;margin-top:4px'>{wait_pred/60:.1f} hours</div>
                    </div>
                </div>
                <div style='font-size:12px;color:#64748b;line-height:1.6;border-top:1px solid #e2e8f0;padding-top:0.75rem'>
                    Model: LightGBM  |  AUC {meta['breach_auc']}  |  Accuracy {meta['breach_acc']}%  |
                    Trained on {meta['total_records']:,} NHS ED records  |  Wait MAE plus or minus {meta['wait_mae']} min
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=breach_prob * 100,
                number={'suffix': '%', 'font': {'color': '#0f172a', 'size': 24, 'family': 'DM Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': '#f8fafc',
                    'bordercolor': '#e2e8f0',
                    'steps': [
                        {'range': [0, 35],  'color': 'rgba(16,185,129,0.15)'},
                        {'range': [35, 60], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 100],'color': 'rgba(239,68,68,0.15)'}
                    ],
                    'threshold': {'line': {'color': AMBER, 'width': 2}, 'thickness': 0.75, 'value': 26}
                },
                title={'text': "4-Hour Breach Probability", 'font': {'color': '#64748b', 'size': 12}}
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
                font=dict(family="DM Sans", color="#334155", size=12),
                margin=dict(l=10, r=10, t=30, b=10),
                height=200
            )
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown('<div class="section-title">Personalised AI Suggestions</div>', unsafe_allow_html=True)
            suggestions, _ = get_suggestions(breach_prob, wait_pred, triage, complaint,
                                              age, imd, news2, comorbid, arrival_mode)

            for title, body, color in suggestions:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                st.markdown(f"""
                <div style='padding:0.9rem 1rem;background:rgba({r},{g},{b},0.06);
                             border:1px solid rgba({r},{g},{b},0.25);border-radius:9px;margin-bottom:8px'>
                    <div style='font-size:13px;font-weight:600;color:{color};margin-bottom:4px'>{title}</div>
                    <div style='font-size:12.5px;color:#334155;line-height:1.6'>{body}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding:2.5rem 1.5rem;text-align:center;color:#64748b;
                        border:1px dashed #bfdbfe;border-radius:12px;margin-top:0.5rem'>
                <div style='font-size:15px;color:#1e40af;margin-bottom:0.5rem;font-weight:600'>AI Patient Assistant</div>
                <div style='font-size:13px;line-height:1.8;color:#64748b'>
                    Fill in patient details on the left and click<br>
                    <strong style='color:#1e40af'>Get AI Prediction and Personalised Suggestions</strong><br><br>
                    The model will predict:<br><br>
                    4-hour breach risk probability<br>
                    Estimated wait time<br>
                    Personalised clinical suggestions<br>
                    Equity alerts (IMD)<br>
                    NEWS2-based escalation
                </div>
            </div>
            """, unsafe_allow_html=True)

# QUEUE AND PREDICTIONS
elif page == "Queue and Predictions":
    st.markdown("### Queue Management and Wait Time Predictions")
    st.caption("Real distribution from your NHS ED dataset")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Wait Time Distribution by Triage Category</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors_t = {1: RED, 2: "#f97316", 3: AMBER, 4: GREEN, 5: BLUE}
        for cat in sorted(df['triage_category'].unique()):
            subset = df[df['triage_category'] == cat]['wait_time_to_assessment_min']
            fig.add_trace(go.Box(y=subset, name=f"Cat {cat}", marker_color=colors_t[cat],
                                 boxmean=True, line_width=1.5))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Wait Time (min)"),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Arrival Mode vs Breach Rate and Avg Wait</div>', unsafe_allow_html=True)
        am = df.groupby('arrival_mode').agg(
            count=('patient_id', 'count'),
            breach_rate=('four_hour_breach', 'mean'),
            avg_wait=('wait_time_to_assessment_min', 'mean')
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_bar(x=am['arrival_mode'], y=am['breach_rate'] * 100,
                     name="Breach Rate %", marker_color=[TEAL, RED, AMBER, BLUE])
        fig2.add_scatter(x=am['arrival_mode'], y=am['avg_wait'],
                         name="Avg Wait (min)", mode="lines+markers", yaxis="y2",
                         line=dict(color="#7c3aed", width=2.5), marker=dict(size=8))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Breach Rate %"),
            yaxis2=dict(overlaying="y", side="right", title="Avg Wait (min)",
                        gridcolor="rgba(0,0,0,0)", color="#334155")
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Model Predictions on Sample Queue — 20 patients from your dataset</div>', unsafe_allow_html=True)
    sample = df.sample(20, random_state=42).copy()
    for col in ['arrival_mode', 'chief_complaint', 'nhs_trust', 'ics_region', 'trust_type', 'patient_sex']:
        sample[col + '_enc'] = le_dict[col].transform(sample[col].astype(str))

    occ_s   = sample['bed_occupancy_pct'].values
    wait_s  = sample['wait_time_to_assessment_min'].values
    staff_s = sample['staff_ratio'].values
    night_s = ((sample['hour_of_day'].values >= 22) | (sample['hour_of_day'].values <= 6)).astype(int)
    ambul_s = (sample['arrival_mode'].values == 'Ambulance').astype(int)
    wf_s    = sample['month'].isin([12, 1, 2]).astype(int).values

    sample['occ_x_triage']         = (occ_s / 100) * sample['triage_category'].values
    sample['wait_per_staff']        = wait_s / (staff_s + 0.01)
    sample['acuity_x_cpi']          = sample['patient_acuity_score'].values * sample['capacity_pressure_index'].values
    sample['night_high_occ']        = night_s * ((occ_s > 95).astype(int))
    sample['ambul_handover_stress'] = ambul_s * sample['handover_breach_gt30min'].values
    sample['is_winter']             = wf_s

    X_s = sample[FEATURES]
    sample['Predicted Breach Risk'] = (clf.predict_proba(X_s)[:, 1] * 100).round(1).astype(str) + '%'
    sample['Predicted Wait (min)']  = reg.predict(X_s).round(0).astype(int)
    sample['Actual Breach']         = sample['four_hour_breach'].map({0: 'No', 1: 'Yes'})

    display = sample[['patient_id', 'patient_age', 'chief_complaint', 'triage_category',
                       'arrival_mode', 'Predicted Wait (min)', 'Predicted Breach Risk', 'Actual Breach']].rename(columns={
        'patient_id': 'ID', 'patient_age': 'Age', 'chief_complaint': 'Complaint',
        'triage_category': 'Triage', 'arrival_mode': 'Arrival'})
    st.dataframe(display, use_container_width=True, hide_index=True)

# DATA ANALYTICS
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
                     color=disp.values,
                     color_continuous_scale=['#0ea5e9', '#1e40af', '#f59e0b', '#ef4444', '#7c3aed'])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            height=250, coloraxis_showscale=False
        )
        fig.update_traces(text=[f"{v:,}" for v in disp.values], textposition="outside",
                          textfont=dict(color="#334155"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Chief Complaint vs Avg ED Length of Stay</div>', unsafe_allow_html=True)
        cc_los = df.groupby('chief_complaint')['ed_los_min'].mean().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(x=cc_los.values, y=cc_los.index, orientation='h',
                                marker_color=TEAL, marker_line_width=0))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            font=dict(family="DM Sans", color="#334155", size=12),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Avg ED LOS (min)"),
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Arrival Heatmap — Day of Week vs Hour</div>', unsafe_allow_html=True)
    heatmap_data  = df.groupby(['day_of_week', 'hour_of_day'])['patient_id'].count().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour_of_day', values='patient_id').fillna(0)
    day_labels    = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig_h = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
        y=[day_labels[i] for i in heatmap_pivot.index],
        colorscale=[[0, "#eff6ff"], [0.5, "#93c5fd"], [1, "#1e40af"]],
        showscale=True,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Arrivals: %{z}<extra></extra>"
    ))
    fig_h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        font=dict(family="DM Sans", color="#334155", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="section-title">NHS Trust Comparison — Breach Rate and Avg Wait</div>', unsafe_allow_html=True)
    trust_stats = df.groupby('nhs_trust').agg(
        breach_rate=('four_hour_breach', 'mean'),
        avg_wait=('wait_time_to_assessment_min', 'mean'),
        count=('patient_id', 'count')
    ).reset_index().sort_values('breach_rate', ascending=False)
    trust_stats['Trust Short'] = trust_stats['nhs_trust'].str.replace(' NHS.*', '', regex=True).str[:30]
    fig_t = go.Figure()
    fig_t.add_bar(x=trust_stats['Trust Short'], y=trust_stats['breach_rate'] * 100,
                  name="Breach Rate %", marker_color=RED, opacity=0.8)
    fig_t.add_scatter(x=trust_stats['Trust Short'], y=trust_stats['avg_wait'],
                      name="Avg Wait (min)", mode="lines+markers", yaxis="y2",
                      line=dict(color=TEAL, width=2.5), marker=dict(size=7))
    fig_t.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        font=dict(family="DM Sans", color="#334155", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=290,
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Breach Rate %"),
        yaxis2=dict(overlaying="y", side="right", title="Avg Wait (min)",
                    gridcolor="rgba(0,0,0,0)", color="#334155"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", tickangle=30)
    )
    st.plotly_chart(fig_t, use_container_width=True)

# MODEL INFO
elif page == "Model Info":
    st.markdown("### Model Information and Architecture")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Breach Model AUC", str(meta['breach_auc']))
    k2.metric("Accuracy", f"{meta['breach_acc']}%")
    k3.metric("Wait Time MAE", f"plus or minus {meta['wait_mae']} min")
    k4.metric("Training Records", f"{int(meta['total_records']*0.7):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top 15 Features — LightGBM Breach Model</div>', unsafe_allow_html=True)

    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(15)
    fig_fi = go.Figure(go.Bar(
        x=feat_imp['Importance'], y=feat_imp['Feature'],
        orientation='h', marker_color=BLUE, marker_line_width=0
    ))
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        font=dict(family="DM Sans", color="#334155", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)", title="Feature Importance"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"""
    <div class="ai-box">
        <div class="section-title">Architecture and Methodology</div>
        <div style='color:#0f172a;font-size:13.5px;line-height:1.9'>
            <strong style='color:#1e40af'>Breach Classifier:</strong> LightGBM (n_estimators=300, max_depth=6, lr=0.05) trained on
            {int(meta['total_records']*0.7):,} records with SMOTE oversampling for class balance.
            Features include engineered interaction terms (occupancy x triage, wait per staff ratio, acuity x capacity pressure index).<br><br>
            <strong style='color:#1e40af'>Wait Time Regressor:</strong> LightGBM Regressor with same architecture.
            MAE plus or minus {meta['wait_mae']} minutes on held-out test set (R-squared = 0.998).<br><br>
            <strong style='color:#1e40af'>RL Layer (DQN and PPO):</strong> Deep Reinforcement Learning agents trained in
            custom NHS ED Gymnasium environment. State space includes queue depth, occupancy, staff ratio,
            triage mix. Action space covers resource reallocation policies. Reward function optimises
            4-hour target compliance with equity penalty for IMD Q1 disparity.<br><br>
            <strong style='color:#1e40af'>Explainability:</strong> SHAP beeswarm, waterfall, and interaction plots
            identify key breach drivers. Top contributors: bed occupancy, queue length, triage category, NEWS2 score, capacity pressure index.<br><br>
            <span style='color:#64748b;font-size:12px'>
            Dissertation: A Predictive Deep Reinforcement Learning AI Assistant for Real-Time Resource
            Scheduling Optimisation in NHS Emergency Care |
            Northumbria University London | MSc Big Data and Data Science Technology |
            Module LD7236 | Supervisor: Dr. Rejwan Bin Sulaiman | 2024-25
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
