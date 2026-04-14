import sys
import os

# Ensure the project root is in the Python path for imports
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.preprocessing import (
    load_data,
    prepare_text,
    vectorize_text,
    create_similarity_matrix
)

from src.scoring import (
    get_top_matches
)

# Page configuration
st.set_page_config(
    page_title="Intelligent Profile Matching",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&family=Montserrat:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

:root {
    --gold:        #b8922a;
    --gold-light:  #d4a843;
    --gold-pale:   #f0d98c;
    --cream:       #fdf9f2;
    --cream-dark:  #f5efe2;
    --warm-white:  #fffdf8;
    --text-dark:   #1a1409;
    --text-mid:    #5a4e38;
    --text-muted:  #9e8e72;
    --border:      rgba(184,146,42,0.22);
    --border-strong: rgba(184,146,42,0.50);
}

html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

.stApp { background: var(--cream); color: var(--text-dark); }

/* Hide Streamlit chrome and ALL anchor/link icons */
#MainMenu, footer, header { visibility: hidden; }
a[href^="#"], .css-zt5igj a, [data-testid="stMarkdownContainer"] a,
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
svg[data-testid="stMarkdownLinkIcon"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--warm-white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-mid) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--cream) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #b8922a 0%, #d4a843 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 18px rgba(184,146,42,0.35) !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88 !important; }

/* ── Top header bar ── */
.top-header {
    background: linear-gradient(135deg, #1a1409 0%, #2d2210 60%, #1a1409 100%);
    padding: 0.65rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--gold);
}
.top-header-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--gold-light);
    letter-spacing: 0.06em;
}
.top-header-nav {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(212,168,67,0.55);
    display: flex;
    gap: 2.2rem;
}

/* ── Hero ── */
.hero-wrapper {
    background: var(--warm-white);
    text-align: center;
    padding: 3.5rem 2rem 3rem;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 50% -10%, rgba(184,146,42,0.10) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(184,146,42,0.04) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 79px, rgba(184,146,42,0.03) 80px);
    pointer-events: none;
}
.hero-ornament {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
    opacity: 0.85;
}
.hero-ornament::before { content: '— '; }
.hero-ornament::after  { content: ' —'; }
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 700;
    color: var(--text-dark);
    letter-spacing: -0.01em;
    line-height: 1.1;
    margin: 0 0 0.6rem;
}
.hero-title em { font-style: italic; color: var(--gold); }
.hero-sub {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0;
}
.hero-line {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.5rem auto 0;
}

/* ── Stat tiles ── */
.stat-tile {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1.3rem 1.2rem 1.1rem;
    text-align: center;
}
.stat-tile-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.stat-tile-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.35rem;
}

/* ── Section label ── */
.section-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Profile card ── */
.profile-card {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: 6px;
    padding: 1.8rem;
    box-shadow: 0 2px 20px rgba(184,146,42,0.06);
}
.profile-card-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-dark);
    margin: 0 0 0.2rem;
}
.profile-card-id {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
}
.profile-pill {
    display: inline-block;
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 0.22rem 0.7rem;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-mid);
    background: var(--cream);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── Match card ── */
.match-card {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 12px rgba(184,146,42,0.04);
    transition: border-color 0.25s, box-shadow 0.25s;
}
.match-card:hover { border-color: var(--gold); box-shadow: 0 4px 24px rgba(184,146,42,0.13); }
.match-rank {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.32rem;
}
.match-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 0.18rem;
}
.match-detail { font-family: 'Montserrat', sans-serif; font-size: 0.74rem; color: var(--text-muted); }
.match-score-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: right;
    margin-bottom: 0.18rem;
}
.match-score-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
    text-align: right;
    line-height: 1;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #b8922a 0%, #f0d98c 100%) !important;
    border-radius: 2px !important;
}
[data-testid="stProgress"] > div {
    background: var(--cream-dark) !important;
    border-radius: 2px !important;
    height: 4px !important;
}

/* ── Gold divider ── */
.gold-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-strong) 30%, var(--border-strong) 70%, transparent);
    margin: 1.8rem 0;
}

/* ── Footer ── */
.luxury-footer {
    background: linear-gradient(135deg, #1a1409 0%, #2d2210 60%, #1a1409 100%);
    border-top: 2px solid var(--gold);
    padding: 2.2rem 3rem;
    margin-top: 3rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.7rem;
}
.footer-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--gold-light);
    letter-spacing: 0.1em;
}
.footer-divider-line { width: 60px; height: 1px; background: var(--gold); opacity: 0.45; }
.footer-copy {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.66rem;
    font-weight: 400;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(212,168,67,0.48);
    text-align: center;
}

/* ── Sidebar info ── */
.sidebar-info {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.1rem;
    margin-top: 0.9rem;
}
.sidebar-info-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 0.45rem;
}
.sidebar-info-body {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.77rem;
    color: var(--text-muted) !important;
    line-height: 1.65;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 5rem 2rem; }
.empty-ornament {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    color: var(--gold);
    opacity: 0.3;
    line-height: 1;
    margin-bottom: 1.2rem;
}
.empty-text {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: var(--text-muted);
}
.empty-text strong { color: var(--gold); }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_and_prepare():
    users, feedback = load_data()
    users = prepare_text(users)
    tfidf_matrix = vectorize_text(users)
    similarity_matrix = create_similarity_matrix(tfidf_matrix)
    return users, feedback, similarity_matrix

users, feedback, similarity_matrix = load_and_prepare()

# Top Header
st.markdown("""
<div class="top-header">
    <div class="top-header-brand">✦ &nbsp; ProfileMatch </div>
    <div class="top-header-nav">
        <span>Dashboard</span>
        <span>Profiles</span>
        <span>Analytics</span>
        <span>ML Project</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.6rem;text-align:center">
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.65rem;font-weight:700;
                    color:#b8922a;letter-spacing:0.04em;line-height:1.2">
            ✦ Profile<br>Match
        </div>
        <div style="width:40px;height:1px;background:#b8922a;margin:0.8rem auto;opacity:0.45"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:Montserrat,sans-serif;font-size:0.63rem;font-weight:700;'
                'letter-spacing:0.2em;text-transform:uppercase;color:#b8922a;margin-bottom:0.4rem">'
                'Select Profile</p>', unsafe_allow_html=True)

    user_names = (
        [f"{row.get('name', f'User {i}')}" for i, row in users.iterrows()]
        if hasattr(users, 'iterrows')
        else [f"User {i}" for i in range(len(users))]
    )

    user_index = st.selectbox(
        "Choose a user",
        options=range(len(users)),
        format_func=lambda i: user_names[i],
        label_visibility="collapsed"
    )

    find_btn = st.button("✦  Find Best Matches", use_container_width=True)

    st.markdown("""
    <div class="sidebar-info">
        <div class="sidebar-info-title">How it works</div>
        <div class="sidebar-info-body">
            Select any profile from the dropdown, then click <em>Find Best Matches</em>.
            The engine uses TF-IDF vectorisation and cosine similarity to rank
            the top 5 most compatible profiles in real time.
        </div>
    </div>
    <div class="sidebar-info">
        <div class="sidebar-info-title">About</div>
        <div class="sidebar-info-body">
            ProfileMatch · v1.0<br>
            Intelligent compatibility engine <br> Real-time profile recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-ornament">AI-Powered Recommendation Engine</div>
    <h1 class="hero-title">Intelligent Profile <em>Matching</em> System</h1>
    <p class="hero-sub">Content-based compatibility engine &nbsp;·&nbsp; TF-IDF &nbsp;·&nbsp; Cosine Similarity</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# Stats Tiles
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
cols = st.columns(4)
for col, val, label in [
    (cols[0], len(users), "Total Profiles"),
    (cols[1], "5",        "Top Matches"),
    (cols[2], "TF-IDF",   "Vectoriser"),
    (cols[3], "Cosine",   "Similarity"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-tile-value">{val}</div>
            <div class="stat-tile-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

# Main
if find_btn:
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-label">✦ Selected Profile</div>', unsafe_allow_html=True)
        row        = users.iloc[user_index]
        name       = row.get("name",       "—")
        profession = row.get("profession", "—")
        location   = row.get("location",   "—")
        age        = row.get("age",         "—")
        gender     = row.get("gender",      "—")

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-card-name">👤 {name}</div>
            <div class="profile-card-id">Profile #{user_index}</div>
            <span class="profile-pill">💼 {profession}</span>
            <span class="profile-pill">📍 {location}</span>
            <span class="profile-pill">🎂 {age}</span>
            <span class="profile-pill">⚧ {gender}</span>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-label">✦ Top 5 Compatible Profiles</div>', unsafe_allow_html=True)

        with st.spinner("Discovering best matches…"):
            matches = get_top_matches(users, similarity_matrix, user_index)

        ranks  = ["Best Match", "2nd Match", "3rd Match", "4th Match", "5th Match"]
        medals = ["🥇", "🥈", "🥉", "④", "⑤"]

        for i, match in enumerate(matches):
            m_name  = match.get("name",       "—")
            m_prof  = match.get("profession", "—")
            m_loc   = match.get("location",   "—")
            m_score = match.get("score",       0)
            frac    = min(max(float(m_score) / 100.0, 0.0), 1.0)

            st.markdown(f"""
            <div class="match-card">
                <div>
                    <div class="match-rank">{medals[i]} &nbsp; {ranks[i]}</div>
                    <div class="match-name">{m_name}</div>
                    <div class="match-detail">💼 {m_prof} &nbsp;·&nbsp; 📍 {m_loc}</div>
                </div>
                <div>
                    <div class="match-score-label">Compatibility</div>
                    <div class="match-score-value">{m_score}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(frac)
            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-ornament">✦</div>
        <div class="empty-text">
            Select a profile from the sidebar and click
            <strong>Find Best Matches</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="luxury-footer">
    <div class="footer-logo">✦ &nbsp; Intelligent Profile Matching System</div>
    <div class="footer-divider-line"></div>
    <div class="footer-copy">
        Developed by Debabrata Kuiry &nbsp;·&nbsp;
        ProfileMatch &nbsp;·&nbsp; v1.0
    </div>
</div>
""", unsafe_allow_html=True)