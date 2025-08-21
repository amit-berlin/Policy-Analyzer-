import re
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Policy-Making Think Tank (Agentic AI)", layout="wide")
st.title("🇮🇳 Policy-Making Think Tank — Agentic AI (Free, No-API)")

st.caption(
    "Enter up to 10 real problems. Five lightweight agents (Retriever • Topic Modeler • "
    "Intervention Designer • KPI & Data Planner • Policy Composer) will collaborate to draft a "
    "zero-fluff, evidence-ready policy outline you can copy/share."
)

# -----------------------------
# Small utilities
# -----------------------------
CLEAN_RE = re.compile(r"[^a-z0-9\s]")

KEYWORD_TO_INTERVENTIONS = {
    # Education
    "education": ["Teacher training (TOT)", "Digital content & LMS", "School infrastructure audit", "Scholarships for girls"],
    "school": ["Remedial learning program", "Attendance tracking via QR", "Mid-day meal nutrition audit"],
    "skill": ["RPL (Recognition of Prior Learning)", "Industry apprenticeships", "District skill labs"],
    # Health
    "health": ["Telemedicine clinics", "Drug stock e-logistics", "ASHA incentives for follow-ups"],
    "hospital": ["Patient triage SOP", "Equipment uptime monitoring", "e-OPD queue mgmt"],
    "malnutrition": ["POSHAN tracking", "ICDS growth monitoring", "Community kitchens"],
    # Agriculture
    "farmer": ["Soil testing camps", "MSP awareness drives", "FPO formation support"],
    "agriculture": ["Weather advisory SMS", "Micro-irrigation subsidy workflow", "Crop insurance facilitation"],
    # Welfare / Delivery
    "ration": ["ePoS uptime audit", "Beneficiary deduplication", "Doorstep delivery pilots"],
    "pension": ["Aadhaar seeding audit", "Auto-disbursal alerts", "Grievance TAT reduction"],
    # Urban / Infra
    "water": ["Non-revenue water audit", "Sensor-based quality monitoring", "Ward-level O&M contracts"],
    "road": ["Pothole reporting app", "Performance-based maintenance", "Contractor transparency dashboard"],
    # Digital / Service
    "portal": ["Uptime SLO ≥ 99.5%", "SMS/IVR fallback", "Helpline analytics"],
    "grievance": ["Auto-triage by category", "SLA timers", "Escalation matrix"],
}

RISK_LIBRARY = [
    ("Data quality issues", "Data validation & monthly audits"),
    ("Change management resistance", "Officer training & champions"),
    ("Infra downtime", "Active-passive DR & SLOs"),
    ("Vendor lock-in", "Open standards & exit clauses"),
    ("Privacy/security", "Data minimization, RBAC, audits"),
]

def normalize_lines(s: str) -> list:
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    out = []
    for l in lines:
        t = CLEAN_RE.sub(" ", l.lower())
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.append(t)
    # dedupe while preserving order
    seen = set(); deduped = []
    for x in out:
        if x not in seen:
            deduped.append(x); seen.add(x)
    return deduped[:10]

def top_keywords(corpus, top_k=10):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    means = np.asarray(X.mean(axis=0)).ravel()
    idxs = means.argsort()[::-1][:top_k]
    feats = np.array(vec.get_feature_names_out())[idxs]
    scores = means[idxs]
    return list(zip(feats, scores))

def nmf_topics(corpus, n_topics=3, n_words=6):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    n_topics = min(n_topics, X.shape[0]) or 1
    model = NMF(n_components=n_topics, init="nndsvd", random_state=42, max_iter=300)
    W = model.fit_transform(X)
    H = model.components_
    vocab = np.array(vec.get_feature_names_out())
    topics = []
    for k in range(n_topics):
        idx = H[k].argsort()[::-1][:n_words]
        topics.append(list(vocab[idx]))
    return topics, W

def kmeans_clusters(corpus, n_clusters=3):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    n_clusters = min(n_clusters, len(corpus)) or 1
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    groups = {i: [] for i in range(n_clusters)}
    for i, lbl in enumerate(labels):
        groups[lbl].append(corpus[i])
    return groups, labels

def propose_interventions(keywords):
    kws = set(keywords)
    scored = {}
    for kw in kws:
        for key, actions in KEYWORD_TO_INTERVENTIONS.items():
            if key in kw:
                for a in actions:
                    scored[a] = scored.get(a, 0) + 1
    # fallback if nothing matched
    if not scored:
        return ["Rapid baseline survey", "Quick wins pilot", "Grievance redressal improvement", "Monitoring dashboard"]
    # sort by votes
    return [k for k,_ in sorted(scored.items(), key=lambda x: -x[1])][:6]

def kpi_pack(theme_words):
    themes = " ".join(theme_words)
    pack = []
    if any(t in themes for t in ["education","school","skill"]):
        pack += ["Gross Enrollment Ratio", "Learning outcomes (ASER-like)", "Teacher attendance", "Placement rate"]
    if any(t in themes for t in ["health","hospital","malnutrition"]):
        pack += ["OPD footfall", "Stock-out days", "Telemedicine calls", "Anaemia prevalence"]
    if any(t in themes for t in ["farmer","agriculture"]):
        pack += ["Yield per acre", "Irrigation coverage", "Insurance claims TAT", "FPOs formed"]
    if any(t in themes for t in ["water","road","urban"]):
        pack += ["NRW %", "Pothole closure TAT", "Grievance closure SLA", "Contract adherence"]
    if any(t in themes for t in ["portal","grievance","service"]):
        pack += ["Uptime %", "Avg response time", "First-contact resolution", "Escalations %"]
    if not pack:
        pack = ["Coverage % of target group", "On-time service delivery %", "Beneficiary satisfaction", "Grievance TAT"]
    return pack[:4]

def simple_budget(num_buckets, total_cr=100):
    if num_buckets <= 0:
        return []
    share = total_cr / num_buckets
    return [round(share, 2)] * num_buckets

def md_section(title, body):
    return f"## {title}\n\n{body}\n\n"

def download_button(label, content, filename, mime="text/markdown"):
    st.download_button(label, data=content.encode("utf-8"), file_name=filename, mime=mime)

# -----------------------------
# Inputs
# -----------------------------
st.markdown("### Input: List up to 10 real problems (one per line)")
default_hint = (
    "Ration shops often closed on working days\n"
    "Drug stock-outs in district hospital pharmacy\n"
    "School dropout among girls in rural blocks\n"
    "Potholes remain open beyond 7 days after complaint\n"
    "Farmers unaware of crop insurance claim process\n"
    "Portal downtime during application deadlines\n"
    "High grievance resolution time at tehsil office\n"
    "Water quality complaints in wards near river\n"
    "ASHA workers not receiving timely incentives\n"
    "Skill trainees not getting apprenticeships"
)
problems_text = st.text_area("Problems", value=default_hint, height=180, help="Add or replace with your 10 issues.")

col_run, col_opts = st.columns([1,2])
with col_opts:
    n_topics = st.slider("Agent-2 topics", 2, 6, 3, 1)
    n_clusters = st.slider("Agent-2 clusters", 2, 6, 3, 1)
    total_budget = st.slider("Agent-4 total budget (Cr ₹)", 10, 500, 120, 10)

run = col_run.button("Run 5-Agent Think Tank")

# -----------------------------
# Pipeline
# -----------------------------
if run:
    # Agent 1 — Problem Normalizer & Retriever
    with st.container():
        st.subheader("🧩 Agent-1: Problem Normalizer & Signals")
        with st.spinner("Cleaning, deduplicating, extracting keywords..."):
            items = normalize_lines(problems_text)
            kw = top_keywords(items, top_k=12)
            time.sleep(0.6)
        st.write("**Normalized problems (max 10):**")
        st.table(pd.DataFrame({"#": range(1, len(items)+1), "Problem": items}))
        st.write("**Top signals / keywords (TF-IDF):**")
        st.table(pd.DataFrame(kw, columns=["keyword","weight"]).assign(weight=lambda d: d.weight.round(3)))

    # Agent 2 — Topic Modeling & Clustering
    with st.container():
        st.subheader("🧪 Agent-2: Topic Modeler & Clusterer")
        with st.spinner("Discovering themes and grouping related issues..."):
            topics, W = nmf_topics(items, n_topics=n_topics, n_words=6)
            groups, labels = kmeans_clusters(items, n_clusters=n_clusters)
            time.sleep(0.6)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Themes (NMF):**")
            st.table(pd.DataFrame({"topic": [f"T{k+1}" for k in range(len(topics))],
                                   "top_words": [", ".join(t) for t in topics]}))
        with c2:
            st.markdown("**Issue clusters (K-Means):**")
            st.table(pd.DataFrame([{"cluster": c, "issues": "\n".join(v)} for c, v in groups.items()]))

    # Agent 3 — Intervention Designer (rule-based + keyword votes)
    with st.container():
        st.subheader("🛠️ Agent-3: Intervention Designer")
        with st.spinner("Mapping problems to actionable interventions..."):
            theme_words = sorted({w for ts in topics for w in ts})
            actions = propose_interventions(theme_words + [k for k,_ in kw])
            time.sleep(0.6)
        st.write("**Proposed interventions (pilot-ready):**")
        st.table(pd.DataFrame({"Intervention": actions}))

    # Agent 4 — KPI, Data & Budget Planner
    with st.container():
        st.subheader("📊 Agent-4: KPI, Data & Budget Planner")
        with st.spinner("Selecting measurable KPIs and allocating budget..."):
            kpis = kpi_pack(theme_words)
            buckets = max(1, len(actions)//2)
            budgets = simple_budget(buckets, total_cr=total_budget)
            time.sleep(0.6)
        kpi_df = pd.DataFrame({"KPI": kpis})
        bud_df = pd.DataFrame({"Component": [f"Comp-{i+1}" for i in range(buckets)],
                               "Budget (Cr ₹)": budgets})
        st.markdown("**KPIs (track monthly):**")
        st.table(kpi_df)
        st.markdown("**Budget (illustrative):**")
        st.table(bud_df)

    # Agent 5 — Policy Composer & Reviewer (extractive, templated)
    with st.container():
        st.subheader("📝 Agent-5: Policy Composer & Reviewer")
        with st.spinner("Compiling a clean, evidence-ready policy outline..."):
            time.sleep(0.8)

        # Compose markdown policy
        md = []
        md.append("# Draft Policy Outline\n")
        md.append(md_section("1. Problem Statement",
                             "\n".join([f"- {p}" for p in items])))
        md.append(md_section("2. Themes Identified",
                             "\n".join([f"- T{k+1}: {', '.join(t)}" for k,t in enumerate(topics)])))
        md.append(md_section("3. Proposed Interventions",
                             "\n".join([f"- {a}" for a in actions])))
        md.append(md_section("4. KPIs & Data",
                             "\n".join([f"- {k}" for k in kpis]) +
                             "\n\n_Data sources: department MIS, helplines, field surveys, audit trails._"))
        md.append(md_section("5. Implementation Plan",
                             "- **Phase 0 (30 days):** Baseline surveys, infra readiness, SOPs\n"
                             "- **Phase 1 (90 days):** Pilot in 3 districts; weekly review\n"
                             "- **Phase 2 (180 days):** Scale to state-wide; monthly dashboards\n"))
        risks = "\n".join([f"- {r[0]} → *Mitigation:* {r[1]}" for r in RISK_LIBRARY[:4]])
        md.append(md_section("6. Risks & Mitigations", risks))
        md.append(md_section("7. Budget (Illustrative)",
                             "\n".join([f"- {row['Component']}: ₹{row['Budget (Cr ₹)']} Cr"
                                        for _,row in bud_df.iterrows()])))
        md.append(md_section("8. Governance & M&E",
                             "- Monthly KPI reviews at district & state levels\n"
                             "- Independent audits every quarter\n"
                             "- Public dashboards for transparency\n"))

        policy_md = "\n".join(md)

        st.markdown("#### 📄 Draft (Markdown)")
        st.code(policy_md, language="markdown")
        download_button("⬇️ Download Policy (Markdown)", policy_md, "draft_policy.md")

        # Lightweight “confidence” (coverage of mapped interventions & topics)
        coverage = min(1.0, (len(actions)/6*0.5 + len(topics)/5*0.5))
        st.caption(f"Confidence (heuristic coverage): **{coverage:.2f}**  •  "
                   f"Agents: Retriever → Topic Modeler → Designer → KPI/Budget → Composer")

# Footer hint
st.markdown("---")
st.caption("This MVP uses TF-IDF, NMF, K-Means, rule-based mapping, and templated composition to simulate an agentic policy lab — all free/open-source, no external APIs.")
