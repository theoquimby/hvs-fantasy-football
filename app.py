import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HVS Fantasy League",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. BESPOKE STYLING & FONTS ---
# We are importing 'Bangers' font to give it that bespoke, arcade/fantasy feel.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Roboto:wght@400;700&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Roboto', sans-serif;
    }
    
    /* BESPOKE HEADERS - READABILITY UPDATE */
    h1, h2, h3, h4 {
        font-family: 'Bangers', cursive;
        letter-spacing: 2px;
        text-shadow: 3px 3px 0px rgba(0,0,0,0.6);
        line-height: 1.2;
    }
    h1 { color: #fbbf24 !important; font-size: 3.5rem !important; }
    h2 { color: #38bdf8 !important; }
    h3 { color: #f472b6 !important; }
    h4 { color: #cbd5e1 !important; letter-spacing: 1px; }
    
    /* Navigation Bar */
    div[role="radiogroup"] > label > div:first-of-type { display: none; }
    div[role="radiogroup"] {
        flex-direction: row; justify-content: center; gap: 8px; width: 100%;
        background: rgba(255,255,255,0.05); padding: 8px; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1); margin-bottom: 25px;
    }
    div[role="radiogroup"] label {
        background: transparent; border: 1px solid transparent; padding: 8px 16px;
        border-radius: 6px; transition: 0.3s; color: #94a3b8; font-weight: 600; font-size: 14px;
        font-family: 'Roboto', sans-serif;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.1); color: white;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: #38bdf8 !important; color: #0f172a !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); height: 100%;
        display: flex; flex-direction: column; justify-content: center;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-5px); border-color: #fbbf24; }
    .stat-value { font-size: 32px; font-weight: 800; color: #38bdf8; margin: 5px 0; font-family: 'Bangers', cursive; letter-spacing: 1px; }
    .stat-label { font-size: 12px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px; }
    .stat-sub { font-size: 11px; color: #64748b; margin-top: 4px;}

    /* Manager Profile - UPDATED FOR HEIGHT AND LAYOUT */
    .manager-profile {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border: 2px solid #fbbf24; padding: 25px; border-radius: 15px;
        color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        min-height: 750px; /* Force height to match charts */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .profile-header {
        color: #fbbf24; border-bottom: 2px solid #fbbf24; padding-bottom: 10px; margin-top: 0;
        font-family: 'Bangers', cursive; font-size: 36px; letter-spacing: 1px;
    }
    
    /* Rivalry Card */
    .rivalry-container {
        background: rgba(0,0,0,0.4); border: 1px solid #334155; padding: 25px; 
        border-radius: 15px; margin-bottom: 20px;
    }
    
    /* Tale of the Tape Row */
    .tale-row {
        display: flex; justify-content: space-between; padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        font-size: 15px;
    }
    
    /* Badge styling */
    .badge {
        background: rgba(251, 191, 36, 0.15); border: 1px solid #fbbf24;
        color: #fbbf24; padding: 6px 12px; border-radius: 12px;
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        display: block; 
        margin: 0;
    }
    
    /* Trophy Case */
    .trophy-row {
        display: flex; align-items: center; margin-bottom: 10px;
        background: rgba(255,255,255,0.05); padding: 8px; border-radius: 8px;
    }
    .trophy-icon { font-size: 20px; margin-right: 10px; }
    .trophy-text { font-size: 12px; color: #e2e8f0; }
    .trophy-owner { color: #fbbf24; font-weight: bold; margin-left: auto; font-size: 12px; }
    
    /* Dataframe overrides */
    [data-testid="stDataFrame"] { background-color: rgba(0,0,0,0.2); }
    
    /* Chart Container Spacer */
    .chart-box {
        margin-top: 20px;
        margin-bottom: 20px;
        padding: 10px;
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ROBUST DATA PROCESSING ---
@st.cache_data(ttl=600)
def process_data(df):
    try:
        df['Points For'] = df['Points For'].astype(str).str.replace(',', '', regex=False)
        df['Points For'] = pd.to_numeric(df['Points For'], errors='coerce').fillna(0.0)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce').fillna(0).astype(int)
        df['Losses'] = pd.to_numeric(df['Losses'], errors='coerce').fillna(0).astype(int)
        df['Games'] = df['Wins'] + df['Losses']
        df['Win %'] = df.apply(lambda x: x['Wins'] / x['Games'] if x['Games'] > 0 else 0.0, axis=1)
        
        # ADVANCED STAT: Z-SCORE (Era Adjusted Dominance)
        df['Z-Score'] = df.groupby('Year')['Points For'].transform(lambda x: (x - x.mean()) / x.std())
        
        return df
    except Exception:
        return pd.DataFrame()

# --- 4. SIDEBAR ---
st.sidebar.markdown("# 🏈 LEAGUE OFFICE")
with st.sidebar.expander("⚙️ Data Settings", expanded=False):
    source = st.radio("Source:", ["Default CSV", "Upload File", "Google Sheet URL"], key="data_source_radio")
    df = None
    if source == "Default CSV":
        try: df = pd.read_csv('hvs_league_data_cleaned.csv')
        except: st.error("Default CSV not found.")
    elif source == "Upload File":
        up = st.file_uploader("Upload CSV", type=['csv'], key="uploader")
        if up: df = pd.read_csv(up)
    elif source == "Google Sheet URL":
        url = st.text_input("Sheet URL", key="gsheet")
        if url:
            try:
                sid = url.split('/d/')[1].split('/')[0]
                df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv")
            except: st.error("Invalid URL")

if df is None: st.stop()
df = process_data(df)
all_years = sorted(df['Year'].unique(), reverse=True)
all_mgrs = sorted(df['Manager'].unique())

# --- DYNAMIC TROPHY CASE ---
st.sidebar.divider()
st.sidebar.markdown("### 🏆 TROPHY ROOM")
high_score = df.loc[df['Points For'].idxmax()]
best_record = df.loc[df['Win %'].idxmax()]
most_wins_career = df.groupby('Manager')['Wins'].sum().idxmax()

trophies_html = f"""
<div class="trophy-row">
    <div class="trophy-icon">👟</div>
    <div class="trophy-text">GOLDEN BOOT<br><span style="color:#64748b; font-size:10px;">Highest Season Score</span></div>
    <div class="trophy-owner">{high_score['Manager']}<br>{high_score['Points For']:.0f}</div>
</div>
<div class="trophy-row">
    <div class="trophy-icon">🛡️</div>
    <div class="trophy-text">THE SHIELD<br><span style="color:#64748b; font-size:10px;">Best Win %</span></div>
    <div class="trophy-owner">{best_record['Manager']}<br>{best_record['Win %']:.1%}</div>
</div>
<div class="trophy-row">
    <div class="trophy-icon">🐐</div>
    <div class="trophy-text">THE GOAT<br><span style="color:#64748b; font-size:10px;">Career Wins Leader</span></div>
    <div class="trophy-owner">{most_wins_career}</div>
</div>
"""
st.sidebar.markdown(trophies_html, unsafe_allow_html=True)

st.sidebar.divider()
sel_years = st.sidebar.multiselect("Seasons", all_years, default=all_years, key="filter_years")
sel_mgrs = st.sidebar.multiselect("Managers", all_mgrs, default=all_mgrs, key="filter_mgrs")
df_filtered = df[(df['Year'].isin(sel_years)) & (df['Manager'].isin(sel_mgrs))]

# --- 5. MAIN HEADER ---
st.title("🏆 HVS FANTASY LEAGUE")

with st.expander("ℹ️ **User Guide (Click to Expand)**"):
    st.markdown("""
    * **📊 Standings:** Aggregate stats and season-by-season leaderboards.
    * **🔥 Heatmaps:** Visual scoring history with **Era Adjusted** modes.
    * **⚔️ Rivalry:** Deep head-to-head analysis.
    * **📈 Trends:** "Luck Matrix", consistency, and dynasty tracking.
    * **🏛️ Hall of Fame:** Manager cards, radar charts, and DNA plots.
    """)

# NAVIGATION
nav_options = ["📊 Standings", "🔥 Heatmaps", "⚔️ Rivalry", "📈 Trends", "🏛️ Hall of Fame"]
active_tab = st.radio("Nav", nav_options, horizontal=True, label_visibility="collapsed", key="main_nav")

# METRICS HELPER
def render_metric_card(label, value, sub):
    html = '<div class="stat-card">'
    html += f'<div class="stat-label">{label}</div>'
    html += f'<div class="stat-value">{value}</div>'
    html += f'<div class="stat-sub">{sub}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# TOP METRICS
if active_tab == "📊 Standings":
    tot = df_filtered['Points For'].sum()
    avg_p = df_filtered['Points For'].mean()
    high = df_filtered.loc[df_filtered['Points For'].idxmax()]
    w_mgr = df_filtered.groupby('Manager')['Wins'].sum().idxmax()
    w_cnt = df_filtered.groupby('Manager')['Wins'].sum().max()

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("LEAGUE POINTS", f"{tot:,.0f}", "Filtered History")
    with c2: render_metric_card("AVG SEASON", f"{avg_p:,.0f}", "Points/Year")
    with c3: render_metric_card("SCORING KING", f"{high['Points For']:,.0f}", f"{high['Manager']} ({high['Year']})")
    with c4: render_metric_card("WINNINGEST", f"{w_cnt}", f"{w_mgr}")
    st.markdown("<br>", unsafe_allow_html=True)

# HELPER: COMMON PLOTLY LAYOUT UPDATE
def update_plot_style(fig, title, height=500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='white', family="Bangers, cursive"), x=0, y=0.95),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        height=height,
        margin=dict(t=80, b=50, l=50, r=50),
        font=dict(family="Roboto, sans-serif", size=12, color="#cbd5e1"),
        hovermode="x unified"
    )
    return fig

# --- 6. PAGE LOGIC ---

# === PAGE 1: STANDINGS ===
if active_tab == "📊 Standings":
    st.subheader("Leaderboards")
    mode = st.radio("View:", ["Aggregate (All-Time)", "Single Season"], horizontal=True, key="std_mode")
    
    if mode == "Aggregate (All-Time)":
        agg = df_filtered.groupby('Manager').agg({'Wins':'sum','Losses':'sum','Points For':'sum','Year':'count'}).reset_index()
        agg['Win %'] = agg['Wins']/(agg['Wins']+agg['Losses'])
        agg = agg.sort_values('Wins', ascending=False)
        st.dataframe(agg, column_config={
            "Win %": st.column_config.ProgressColumn("Win %", format="%.3f", min_value=0, max_value=1),
            "Points For": st.column_config.NumberColumn("Points", format="%.0f")
        }, hide_index=True, use_container_width=True)
    else:
        y = st.selectbox("Select Season", sel_years, key="std_year")
        s = df[df['Year']==y].sort_values('Wins', ascending=False)
        st.dataframe(s[['Manager','Record','Wins','Losses','Points For']], hide_index=True, use_container_width=True)
    
    with st.expander("🎓 Stats Class: Understanding the Leaderboard"):
        st.markdown("""
        * **How to Read:** This table allows you to sort by Wins, Win %, or Points. Click a column header to toggle sort order.
        * **Fantasy Context:**
            * **Win %:** The true measure of a manager's career consistency. Anything above `.550` is Hall of Fame material.
            * **Points For:** Indicates offensive firepower. If you have high Points but low Wins, you have "Bad Luck" (tough schedule).
        """)

# === PAGE 2: HEATMAPS ===
elif active_tab == "🔥 Heatmaps":
    c_head, c_tog = st.columns([3, 1])
    with c_head: st.subheader("Scoring Intensity")
    with c_tog:
        view_type = st.selectbox("Metric:", ["Raw Points", "Era Adjusted (Z-Score)"])
    
    if view_type == "Raw Points":
        val_col = 'Points For'
        colors = "Viridis"
        title = "Annual Scoring Heatmap (Raw Pts)"
    else:
        val_col = 'Z-Score'
        colors = "RdBu"
        title = "Dominance Index (Std Dev from League Avg)"
        st.caption("ℹ️ *Z-Score shows how dominant a manager was relative to that specific year's environment. Red is above average, Blue is below.*")

    piv = df.pivot_table(index='Manager', columns='Year', values=val_col)
    fig = px.imshow(piv, labels=dict(color="Val"), color_continuous_scale=colors, template="plotly_dark")
    fig = update_plot_style(fig, title, height=650)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🎓 Stats Class: Decoding the Heatmap"):
        st.markdown("""
        * **How to Read:** The X-axis is the Season, the Y-axis is the Manager. The color brightness indicates performance.
        * **Fantasy Context:**
            * **Raw Points:** Good for seeing who had the "Mega Team" in a specific year.
            * **Z-Score (Dominance):** This is the **fair** way to compare eras. Scoring 1500 in 2012 might be harder than 1500 in 2024 due to NFL rule changes. A high Z-Score (Red) means you crushed *your specific peers* that year.
        """)

# === PAGE 3: RIVALRY ===
elif active_tab == "⚔️ Rivalry":
    st.subheader("⚔️ The War Room")
    
    c1, c2 = st.columns(2)
    m1 = c1.selectbox("Manager 1", all_mgrs, index=0, key="riv_m1")
    default_m2_idx = 1 if len(all_mgrs) > 1 else 0
    m2 = c2.selectbox("Manager 2", all_mgrs, index=default_m2_idx, key="riv_m2")
    
    if m1 == m2:
        st.warning("Please select two different managers to analyze.")
    else:
        d1 = df[df['Manager'] == m1].set_index('Year')
        d2 = df[df['Manager'] == m2].set_index('Year')
        common_years = sorted(list(set(d1.index) & set(d2.index)), reverse=True)
        
        if len(common_years) == 0:
            st.info(f"No common seasons found between {m1} and {m2}.")
        else:
            comp_data = []
            for year in common_years:
                p1 = d1.loc[year, 'Points For']
                p2 = d2.loc[year, 'Points For']
                w1 = d1.loc[year, 'Wins']
                w2 = d2.loc[year, 'Wins']
                comp_data.append({
                    "Year": year,
                    f"{m1} Pts": p1,
                    f"{m2} Pts": p2,
                    "Diff": p1 - p2,
                })
            
            comp_df = pd.DataFrame(comp_data)
            comp_df['Better Year'] = np.where(comp_df[f"{m1} Pts"] > comp_df[f"{m2} Pts"], m1, m2)
            wins_m1 = len(comp_df[comp_df['Better Year'] == m1])
            wins_m2 = len(comp_df[comp_df['Better Year'] == m2])
            
            avg1 = comp_df[f"{m1} Pts"].mean()
            avg2 = comp_df[f"{m2} Pts"].mean()
            high1 = comp_df[f"{m1} Pts"].max()
            high2 = comp_df[f"{m2} Pts"].max()
            prob1 = wins_m1 / len(common_years)
            prob2 = wins_m2 / len(common_years)
            
            html_tape = f"""
<div class="rivalry-container">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
<div style="text-align: center; width: 30%;">
<h2 style="color:#38bdf8; margin:0; text-shadow: 0 0 10px rgba(56,189,248,0.5);">{m1}</h2>
<h1 style="font-size: 60px; margin:0; line-height: 1;">{wins_m1}</h1>
<p style="color:#94a3b8; font-weight:bold; letter-spacing:1px; font-size:12px;">BETTER SEASONS</p>
</div>
<div style="text-align: center; font-size: 40px; font-weight: 900; color: #64748b; font-style:italic;">VS</div>
<div style="text-align: center; width: 30%;">
<h2 style="color:#f472b6; margin:0; text-shadow: 0 0 10px rgba(244,114,182,0.5);">{m2}</h2>
<h1 style="font-size: 60px; margin:0; line-height: 1;">{wins_m2}</h1>
<p style="color:#94a3b8; font-weight:bold; letter-spacing:1px; font-size:12px;">BETTER SEASONS</p>
</div>
</div>
<div class="tale-row">
<span style="color:#38bdf8; font-weight:bold; font-size:18px;">{avg1:,.1f}</span> 
<span style="color:#cbd5e1; font-size:12px; letter-spacing:1px;">AVG SEASON SCORE</span> 
<span style="color:#f472b6; font-weight:bold; font-size:18px;">{avg2:,.1f}</span>
</div>
<div class="tale-row">
<span style="color:#38bdf8; font-weight:bold; font-size:18px;">{high1:,.0f}</span> 
<span style="color:#cbd5e1; font-size:12px; letter-spacing:1px;">BEST SEASON</span> 
<span style="color:#f472b6; font-weight:bold; font-size:18px;">{high2:,.0f}</span>
</div>
<div class="tale-row">
<span style="color:#38bdf8">{prob1:.0%}</span> 
<span style="color:#cbd5e1; font-size:12px; letter-spacing:1px;">HISTORICAL DOMINANCE</span> 
<span style="color:#f472b6">{prob2:.0%}</span>
</div>
</div>
"""
            st.markdown(html_tape, unsafe_allow_html=True)
            
            with st.expander("🎓 Stats Class: Reading the Tape"):
                st.markdown("""
                * **How to Read:** This compares aggregate stats across all shared seasons.
                * **Fantasy Context:** "Historical Dominance" is the percentage of seasons where one manager outscored the other. If this is 75%+, one manager effectively "owns" the other.
                """)
            
            st.markdown("#### 📊 Visual Comparison")
            tab_chart1, tab_chart2 = st.tabs(["Yearly Scoring", "Point Differential"])
            
            with tab_chart1:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=comp_df['Year'], y=comp_df[f"{m1} Pts"], name=m1, marker_color='#38bdf8'))
                fig_bar.add_trace(go.Bar(x=comp_df['Year'], y=comp_df[f"{m2} Pts"], name=m2, marker_color='#f472b6'))
                fig_bar.update_layout(barmode='group')
                fig_bar = update_plot_style(fig_bar, "Points Scored per Season")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                with st.expander("🎓 How to Read: Grouped Bars"):
                    st.markdown("**Context:** A simple side-by-side comparison. Look for 'sweeps' where one color dominates several years in a row.")
                
            with tab_chart2:
                comp_df['Cumulative Diff'] = comp_df['Diff'].cumsum()
                fig_line = px.line(comp_df, x='Year', y='Cumulative Diff', markers=True, template="plotly_dark")
                fig_line.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                fig_line.update_traces(line_color='#fbbf24', line_width=4)
                fig_line = update_plot_style(fig_line, f"Cumulative Point Gap ({m1} vs {m2})")
                st.plotly_chart(fig_line, use_container_width=True)
                
                with st.expander("🎓 How to Read: Cumulative Gap"):
                    st.markdown("**Context:** Think of this as a 'Tug of War'. If the line goes UP, Manager 1 is pulling away. If it crosses the dotted line (Zero), the lead in the rivalry has flipped.")

            st.markdown("#### 📜 Season-by-Season Log")
            st.dataframe(
                comp_df[['Year', f"{m1} Pts", f"{m2} Pts", "Diff"]],
                column_config={
                    "Year": st.column_config.NumberColumn(format="%d"),
                    f"{m1} Pts": st.column_config.NumberColumn(format="%.1f"),
                    f"{m2} Pts": st.column_config.NumberColumn(format="%.1f"),
                    "Diff": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
                use_container_width=True
            )

# === PAGE 4: TRENDS ===
elif active_tab == "📈 Trends":
    st.markdown("### League Performance Trends")
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown("#### 🍀 The Luck Matrix (Wins vs. Points)")
    st.caption("Top Left: Lucky (Bad scoring, High wins). Bottom Right: Unlucky (High scoring, Low wins).")
    
    fig_luck = px.scatter(df_filtered, x="Points For", y="Wins", color="Manager", 
                          hover_data=["Year"], text="Year",
                          template="plotly_dark")
    fig_luck.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='white')))
    fig_luck = update_plot_style(fig_luck, "The 'Luck Matrix' (Wins vs Points)")
    st.plotly_chart(fig_luck, use_container_width=True)
    
    with st.expander("🎓 Stats Class: The Luck Matrix"):
        st.markdown("""
        * **How to Read:** Each dot is a Manager's season.
            * **Top Left:** "The Fraud Zone." Low points but High wins.
            * **Bottom Right:** "The Pain Zone." High points but Low wins.
            * **Top Right:** "The Juggernaut." High points and High wins (Good team + Good luck).
        * **Fantasy Context:** Fantasy is 60% skill (points) and 40% luck (schedule/wins). This chart proves who has the right to complain.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig_box = px.box(df_filtered, x='Manager', y='Points For', color='Manager', template="plotly_dark")
    fig_box = update_plot_style(fig_box, "Consistency Analysis (Distribution)")
    st.plotly_chart(fig_box, use_container_width=True)
    
    with st.expander("🎓 Stats Class: Box Plots"):
        st.markdown("""
        * **How to Read:** The colored box represents the middle 50% of a manager's seasons. The lines (whiskers) show their absolute best and worst.
        * **Fantasy Context:**
            * **Short Box:** A consistent manager. You know what you're getting.
            * **Tall Box:** A "Boom or Bust" manager. They either win the league or finish last.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig_line = px.line(df_filtered, x='Year', y='Points For', color='Manager', markers=True, template="plotly_dark")
    fig_line = update_plot_style(fig_line, "Scoring History (Points Scored)")
    st.plotly_chart(fig_line, use_container_width=True)
    
    with st.expander("🎓 Stats Class: Trend Lines"):
        st.markdown("""
        * **How to Read:** Tracks total points scored over time.
        * **Fantasy Context:** Look for trajectories. Is a manager "Washing Up" (steady decline) or "Peaking" (steady incline)?
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    rank_df = df_filtered.copy()
    rank_df['Rank'] = rank_df.groupby('Year')['Points For'].rank(ascending=False, method='min')
    fig_bump = px.line(rank_df, x='Year', y='Rank', color='Manager', markers=True, template="plotly_dark")
    fig_bump.update_yaxes(autorange="reversed") 
    fig_bump = update_plot_style(fig_bump, "The 'Dynasty' Chart (Rankings Over Time)", height=600)
    st.plotly_chart(fig_bump, use_container_width=True)
    
    with st.expander("🎓 Stats Class: The Bump Chart"):
        st.markdown("""
        * **How to Read:** This tracks *Rank*, not points. Being at the top (#1) is best.
        * **Fantasy Context:** This visualizes "Dynasties." If a manager's line stays near the top for 3+ years, they are the 'Patriots' of your league.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# === PAGE 5: HALL OF FAME ===
elif active_tab == "🏛️ Hall of Fame":
    st.subheader("Manager Trading Cards")
    mgr = st.selectbox("Select Profile", all_mgrs, key="hof_mgr")
    stats = df[df['Manager'] == mgr]
    
    if not stats.empty:
        wins = stats['Wins'].sum()
        loss = stats['Losses'].sum()
        pts = stats['Points For'].sum()
        best = stats.loc[stats['Points For'].idxmax()]
        win_pct = wins/(wins+loss) if (wins+loss)>0 else 0
        
        # Calculate extra stats for the longer card
        ranks = df.copy()
        ranks['Rank'] = ranks.groupby('Year')['Points For'].rank(ascending=False)
        mgr_ranks = ranks[ranks['Manager'] == mgr]['Rank']
        
        best_rank = mgr_ranks.min()
        worst_rank = mgr_ranks.max()
        avg_rank = mgr_ranks.mean()
        
        # Archetype Logic
        std_dev = stats['Points For'].std()
        league_std = df['Points For'].std()
        
        if win_pct > 0.60: archetype = "THE DOMINATOR"
        elif pts > df.groupby('Manager')['Points For'].sum().mean() + 200: archetype = "THE SCORER"
        elif std_dev > league_std * 1.1: archetype = "THE WILD CARD"
        elif win_pct < 0.40: archetype = "THE REBUILDER"
        else: archetype = "THE GRINDER"
        
        badges = []
        if len(stats)>=5: badges.append("🛡️ VETERAN")
        if win_pct>0.55: badges.append("🏆 WINNER")
        if stats['Points For'].mean() > df['Points For'].mean() + 80: badges.append("🚀 HIGH ROLLER")
        if wins > 50: badges.append("🗿 DYNASTY")
        if win_pct < 0.40 and len(stats)>3: badges.append("❤️ HEART")
        
        badge_html = "".join([f'<span class="badge">{b}</span>' for b in badges])

        cL, cR = st.columns([1, 1.5])
        
        with cL:
            # EXTENDED HTML CARD WITH NEW SECTIONS
            html_card = '<div class="manager-profile">'
            html_card += f'<div><div class="profile-header">{mgr}</div>'
            html_card += f'<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:15px;">{badge_html}</div>'
            html_card += f'<div style="font-family:Bangers, cursive; font-size:24px; color:#38bdf8; margin-bottom:15px;">{archetype}</div></div>'
            
            html_card += f'<div><div class="tale-row"><span>Seasons Played</span> <strong>{len(stats)}</strong></div>'
            html_card += f'<div class="tale-row"><span>Career Record</span> <strong>{wins}-{loss}</strong></div>'
            html_card += f'<div class="tale-row"><span>Win Percentage</span> <strong>{win_pct:.1%}</strong></div>'
            html_card += f'<div class="tale-row"><span>Total Points</span> <strong>{pts:,.0f}</strong></div>'
            
            html_card += f'<hr style="border-color:rgba(255,255,255,0.1); margin: 15px 0;">'
            html_card += f'<div style="color:#cbd5e1; font-size:12px; letter-spacing:1px; margin-bottom:5px;">CAREER VITALS</div>'
            html_card += f'<div class="tale-row"><span>Best Rank</span> <strong style="color:#38bdf8">#{best_rank:.0f}</strong></div>'
            html_card += f'<div class="tale-row"><span>Worst Rank</span> <strong style="color:#f472b6">#{worst_rank:.0f}</strong></div>'
            html_card += f'<div class="tale-row"><span>Avg Rank</span> <strong>#{avg_rank:.1f}</strong></div></div>'
            
            html_card += f"""<div style="background:rgba(255,255,255,0.1); padding:15px; border-radius:10px; margin-top:20px; text-align:center; border: 1px solid rgba(251, 191, 36, 0.3);">
<div style="font-size:10px; color:#fbbf24; letter-spacing:1px; font-weight:bold;">PEAK SEASON</div>
<div style="font-size:24px; font-weight:800;">{best['Year']}</div>
<div style="color:#cbd5e1; font-size:16px;">{best['Points For']:,.0f} PTS</div>
</div></div>"""
            st.markdown(html_card, unsafe_allow_html=True)
            
        with cR:
            # CHART 1: RADAR
            l_max_p = df['Points For'].max()
            l_max_v = df.groupby('Manager')['Points For'].std().max() or 1
            
            s_peak = (stats['Points For'].max() / l_max_p)*100
            s_cons = 100 - ((stats['Points For'].std()/l_max_v)*100)
            s_win = (stats['Win %'].mean())*100
            s_exp = min(len(stats)*10, 100)
            
            fig_rad = go.Figure(go.Scatterpolar(
                r=[s_win, s_peak, s_cons, s_exp],
                theta=['Win Rate', 'Peak Ability', 'Consistency', 'Experience'],
                fill='toself', line_color='#fbbf24', fillcolor='rgba(251, 191, 36, 0.2)'
            ))
            fig_rad = update_plot_style(fig_rad, "Manager Attributes", height=350)
            fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], showticklabels=False), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_rad, use_container_width=True)
            
            with st.expander("🎓 Stats Class: The Scouting Report"):
                st.markdown("""
                * **How to Read:** The larger the colored area, the better the manager.
                * **Fantasy Context:** * **Peak Ability:** Can they put up massive points when it counts?
                    * **Consistency:** Are they reliable week-to-week, or volatile?
                """)
            
            # CHART 2: VIOLIN
            st.markdown("##### 🧬 Scoring DNA (Distribution)")
            mgr_scores = stats[['Points For']].copy()
            mgr_scores['Group'] = mgr
            lg_scores = df[['Points For']].copy()
            lg_scores['Group'] = 'League Avg'
            comb = pd.concat([mgr_scores, lg_scores])
            
            fig_vio = go.Figure()
            fig_vio.add_trace(go.Violin(x=comb['Group'][comb['Group']==mgr], y=comb['Points For'][comb['Group']==mgr], 
                                        name=mgr, box_visible=True, meanline_visible=True, line_color='#fbbf24'))
            fig_vio.add_trace(go.Violin(x=comb['Group'][comb['Group']=='League Avg'], y=comb['Points For'][comb['Group']=='League Avg'], 
                                        name='League', box_visible=True, meanline_visible=True, line_color='#94a3b8'))
            fig_vio = update_plot_style(fig_vio, "", height=300)
            fig_vio.update_layout(showlegend=False, margin=dict(t=10, b=40, l=40, r=40))
            st.plotly_chart(fig_vio, use_container_width=True)
            
            with st.expander("🎓 Stats Class: Violin Plots"):
                st.markdown("""
                * **How to Read:** The width of the shape shows how often that score happens.
                * **Fantasy Context:**
                    * **Fat Bottom:** You have a lot of low-scoring seasons (High floor/Low ceiling).
                    * **Fat Top:** You frequently put up monster seasons (High ceiling).
                """)

        # --- ADDITIVE SECTION: CAREER DEEP DIVE ---
        st.divider()
        st.markdown("### 📅 SEASON-BY-SEASON BREAKDOWN")
        
        c_hist1, c_hist2 = st.columns(2)
        
        with c_hist1:
            # CHART 3: YEARLY WINS (BAR)
            fig_wins = px.bar(stats, x='Year', y='Wins', text='Wins', template="plotly_dark")
            fig_wins.update_traces(marker_color='#38bdf8', textposition='outside')
            fig_wins = update_plot_style(fig_wins, "The Win Column (Wins per Year)")
            st.plotly_chart(fig_wins, use_container_width=True)
            
            with st.expander("🎓 Stats Class: Win History"):
                st.markdown("**Context:** A simple view of success. Look for 'Championship Windows' (clusters of high win seasons).")
            
        with c_hist2:
            # CHART 4: POINTS VS LEAGUE AVG (LINE)
            league_avg = df.groupby('Year')['Points For'].mean().reset_index()
            merged_stats = stats.merge(league_avg, on='Year', suffixes=('', '_Avg'))
            
            fig_vs = go.Figure()
            fig_vs.add_trace(go.Scatter(x=merged_stats['Year'], y=merged_stats['Points For'], name=mgr, line=dict(color='#fbbf24', width=4)))
            fig_vs.add_trace(go.Scatter(x=merged_stats['Year'], y=merged_stats['Points For_Avg'], name="League Avg", line=dict(color='white', dash='dash')))
            fig_vs = update_plot_style(fig_vs, "Versus The World (Manager Pts vs Avg)")
            st.plotly_chart(fig_vs, use_container_width=True)
            
            with st.expander("🎓 Stats Class: Performance vs Average"):
                st.markdown("**Context:** Being above the dotted line means you were better than average. If you are consistently above it, you are a top-tier manager.")

        # DATA TABLE: CAREER LOG
        st.markdown("#### 📜 Career Log")
        
        rank_data = df.copy()
        rank_data['League Rank'] = rank_data.groupby('Year')['Points For'].rank(ascending=False, method='min')
        final_log = rank_data[rank_data['Manager'] == mgr].sort_values('Year', ascending=False)
        
        st.dataframe(
            final_log[['Year', 'Record', 'Points For', 'League Rank', 'Win %']],
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "Win %": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                "Points For": st.column_config.NumberColumn(format="%.1f"),
                "League Rank": st.column_config.NumberColumn(format="#%d")
            },
            hide_index=True,
            use_container_width=True
        )