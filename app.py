import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HVS Fantasy League",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DARK MODE CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Glass Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stat-value { font-size: 28px; font-weight: bold; color: #38bdf8; margin: 5px 0; }
    .stat-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; }
    
    /* Manager Profile Card */
    .manager-profile {
        background: linear-gradient(180deg, #1e3a8a 0%, #000000 100%);
        border: 2px solid #fbbf24;
        padding: 25px;
        border-radius: 15px;
        color: white;
        height: 100%;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }
    .badge {
        background: rgba(251, 191, 36, 0.2);
        border: 1px solid #fbbf24;
        color: #fbbf24;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING (Restored Logic) ---
@st.cache_data(ttl=600)
def process_data(df):
    try:
        # Clean numeric columns strictly
        df['Points For'] = df['Points For'].astype(str).str.replace(',', '', regex=False)
        df['Points For'] = pd.to_numeric(df['Points For'], errors='coerce').fillna(0.0)
        
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce').fillna(0).astype(int)
        df['Losses'] = pd.to_numeric(df['Losses'], errors='coerce').fillna(0).astype(int)
        
        df['Games'] = df['Wins'] + df['Losses']
        df['Win %'] = df.apply(lambda x: x['Wins'] / x['Games'] if x['Games'] > 0 else 0.0, axis=1)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- 4. SIDEBAR & DATA INPUT ---
st.sidebar.markdown("### 🏈 League Office")

# Data Source Switcher
with st.sidebar.expander("⚙️ Data Source", expanded=True):
    source = st.radio("Load Data From:", ["Default CSV", "Upload File", "Google Sheet URL"])
    
    df = None
    if source == "Default CSV":
        try:
            df = pd.read_csv('hvs_league_data_cleaned.csv')
        except:
            st.error("Default file not found.")
            
    elif source == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
    elif source == "Google Sheet URL":
        url = st.text_input("Paste Google Sheet URL")
        if url:
            try:
                # Convert edit URL to export URL
                sheet_id = url.split('/d/')[1].split('/')[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(csv_url)
            except:
                st.error("Invalid Google Sheet URL.")

if df is None:
    st.info("👈 Please select a data source to begin.")
    st.stop()

# Process the loaded data
df = process_data(df)

# Filters
all_years = sorted(df['Year'].unique(), reverse=True)
all_managers = sorted(df['Manager'].unique())

st.sidebar.divider()
selected_years = st.sidebar.multiselect("Seasons", all_years, default=all_years)
selected_managers = st.sidebar.multiselect("Managers", all_managers, default=all_managers)

df_filtered = df[(df['Year'].isin(selected_years)) & (df['Manager'].isin(selected_managers))]

if df_filtered.empty:
    st.warning("No data found for these filters.")
    st.stop()

# --- 5. HOME PAGE BLURB ---
st.title("🏆 HVS Fantasy League Hub")

with st.expander("ℹ️ **Welcome! How to use this Dashboard**", expanded=False):
    st.markdown("""
    **Welcome to the official analytics hub of the HVS League!** This tool allows you to explore our league's history with interactive charts and deep-dive stats.
    
    * **📊 Standings:** See who ruled the regular season each year or view the all-time aggregate table.
    * **🔥 Heatmaps:** Visualize scoring eras. Darker colors = More points.
    * **⚔️ Rivalry:** Pick two managers and see their head-to-head scoring history.
    * **🏛️ Hall of Fame:** The ultimate profile card. View detailed career stats, award badges, and attribute radars.
    
    *Use the **Sidebar** on the left to filter by specific years or managers!*
    """)

st.divider()

# --- 6. TOP METRICS ---
def card(label, value, subtext=""):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div style="font-size: 12px; color: #64748b;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

total_pts = df_filtered['Points For'].sum()
avg_score = df_filtered['Points For'].mean()
highest_score_row = df_filtered.loc[df_filtered['Points For'].idxmax()]
most_wins_row = df_filtered.groupby('Manager')['Wins'].sum().idxmax()
most_wins_val = df_filtered.groupby('Manager')['Wins'].sum().max()

c1, c2, c3, c4 = st.columns(4)
with c1: card("Total Points", f"{total_pts:,.0f}", "League History")
with c2: card("Avg Season Score", f"{avg_score:,.0f}", "Per Manager")
with c3: card("Scoring King", f"{highest_score_row['Manager']}", f"{highest_score_row['Points For']:,.0f} pts (Single Season)")
with c4: card("Winningest", f"{most_wins_row}", f"{most_wins_val} Total Wins")

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Standings", "🔥 Heatmaps", "⚔️ Rivalry", "📈 Trends", "🏛️ Hall of Fame"
])

# === TAB 1: STANDINGS ===
with tab1:
    view_mode = st.radio("View Mode:", ["Aggregate (All-Time)", "Single Season"], horizontal=True)
    if view_mode == "Aggregate (All-Time)":
        agg = df_filtered.groupby('Manager').agg({
            'Wins': 'sum', 'Losses': 'sum', 'Points For': 'sum', 'Year': 'count'
        }).reset_index()
        agg['Win %'] = agg['Wins'] / (agg['Wins'] + agg['Losses'])
        agg = agg.sort_values('Wins', ascending=False)
        st.dataframe(agg, column_config={"Win %": st.column_config.ProgressColumn("Win %", format="%.3f", min_value=0, max_value=1), "Points For": st.column_config.NumberColumn("Total Points", format="%.0f")}, hide_index=True, use_container_width=True)
    else:
        y = st.selectbox("Select Season", selected_years)
        s_df = df[df['Year'] == y].sort_values('Wins', ascending=False)
        st.dataframe(s_df[['Manager', 'Record', 'Wins', 'Losses', 'Points For']], hide_index=True, use_container_width=True)

# === TAB 2: HEATMAPS ===
with tab2:
    st.subheader("Scoring Eras")
    pivot_df = df.pivot_table(index='Manager', columns='Year', values='Points For')
    fig = px.imshow(pivot_df, labels=dict(x="Year", y="Manager", color="Points"), color_continuous_scale="Viridis", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 3: RIVALRY ===
with tab3:
    st.subheader("Head-to-Head Comparison")
    rc1, rc2 = st.columns(2)
    m1 = rc1.selectbox("Home Manager", all_managers, index=0)
    m2 = rc2.selectbox("Away Manager", all_managers, index=1 if len(all_managers) > 1 else 0)
    
    if m1 == m2:
        st.error("Pick two different managers.")
    else:
        m1_data = df[df['Manager'] == m1].set_index('Year')['Points For']
        m2_data = df[df['Manager'] == m2].set_index('Year')['Points For']
        common_years = m1_data.index.intersection(m2_data.index)
        
        if len(common_years) == 0:
            st.info("No common seasons found.")
        else:
            comp = pd.DataFrame({'Year': common_years, m1: m1_data[common_years], m2: m2_data[common_years]})
            comp['Winner'] = np.where(comp[m1] > comp[m2], m1, m2)
            w1 = len(comp[comp['Winner'] == m1])
            w2 = len(comp[comp['Winner'] == m2])
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 15px; border: 1px solid #334155;">
                <div style="text-align: center; width: 40%;"><h2 style="color: #38bdf8; margin:0;">{m1}</h2><h1 style="font-size: 50px; margin:0;">{w1}</h1><p style="color: #94a3b8;">SEASONS WON</p></div>
                <div style="text-align: center; width: 20%; font-size: 30px; font-weight: bold; color: #64748b;">VS</div>
                <div style="text-align: center; width: 40%;"><h2 style="color: #f472b6; margin:0;">{m2}</h2><h1 style="font-size: 50px; margin:0;">{w2}</h1><p style="color: #94a3b8;">SEASONS WON</p></div>
            </div><br>""", unsafe_allow_html=True)
            
            fig_riv = go.Figure()
            fig_riv.add_trace(go.Bar(x=comp['Year'], y=comp[m1], name=m1, marker_color='#38bdf8'))
            fig_riv.add_trace(go.Bar(x=comp['Year'], y=comp[m2], name=m2, marker_color='#f472b6'))
            fig_riv.update_layout(template="plotly_dark", barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_riv, use_container_width=True)

# === TAB 4: TRENDS ===
with tab4:
    fig_trend = px.line(df_filtered, x='Year', y='Points For', color='Manager', markers=True, template="plotly_dark")
    fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

# === TAB 5: HALL OF FAME (ENHANCED) ===
with tab5:
    st.subheader("Manager Trading Cards")
    
    target_mgr = st.selectbox("View Manager Profile", all_managers)
    stats = df[df['Manager'] == target_mgr]
    
    if not stats.empty:
        wins = stats['Wins'].sum()
        losses = stats['Losses'].sum()
        pts = stats['Points For'].sum()
        games = wins + losses
        win_pct = wins / games if games > 0 else 0
        best_yr = stats.loc[stats['Points For'].idxmax()]
        avg_pts = stats['Points For'].mean()
        
        # --- BADGE LOGIC ---
        badges = []
        if len(stats) >= 5: badges.append("🛡️ Veteran")
        if win_pct > 0.60: badges.append("🏆 Winner")
        if avg_pts > df['Points For'].mean() + 100: badges.append("🚀 High Roller")
        if stats['Points For'].std() > 200: badges.append("🎰 Wildcard")
        if wins + losses > 50: badges.append("🗿 Dynasty")
        
        badge_html = "".join([f'<span class="badge">{b}</span>' for b in badges])
        
        # --- LAYOUT ---
        # Using [1, 1.3] to bring plot closer and make it bigger relative to card
        c_left, c_right = st.columns([1, 1.3])
        
        with c_left:
            st.markdown(f"""
            <div class="manager-profile">
                <h2 style="color: #fbbf24 !important; border-bottom: 1px solid #fbbf24; padding-bottom: 10px; margin-top:0;">{target_mgr}</h2>
                <div class="badge-container">{badge_html}</div>
                <br>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Seasons:</span> <strong>{len(stats)}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Record:</span> <strong>{wins}-{losses} ({win_pct:.1%})</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Career Pts:</span> <strong>{pts:,.0f}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Pts/Game (Est):</span> <strong>{(pts/games if games else 0):.1f}</strong>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
                    <span style="font-size: 12px; color: #fbbf24; letter-spacing: 1px;">PEAK PERFORMANCE</span><br>
                    <span style="font-size: 24px; font-weight: bold;">{best_yr['Year']}</span><br>
                    <span style="font-size: 18px; color: #e2e8f0;">{best_yr['Points For']:,.0f} Pts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_right:
            # Enhanced Radar Chart
            league_max_pts = df['Points For'].max()
            league_max_vol = df.groupby('Manager')['Points For'].std().max() or 1
            
            score_peak = (stats['Points For'].max() / league_max_pts) * 100
            score_const = 100 - ((stats['Points For'].std() / league_max_vol) * 100)
            score_win = (stats['Win %'].mean()) * 100
            score_exp = min(len(stats)*10, 100)
            
            # Additional metric: Efficiency (Pts relative to games played)
            
            fig_rad = go.Figure(go.Scatterpolar(
                r=[score_win, score_peak, score_const, score_exp],
                theta=['Win Rate', 'Peak Ability', 'Consistency', 'Experience'],
                fill='toself',
                line_color='#fbbf24',
                fillcolor='rgba(251, 191, 36, 0.3)'
            ))
            fig_rad.update_layout(
                template="plotly_dark",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=30),
                height=500 # Taller chart!
            )
            st.plotly_chart(fig_rad, use_container_width=True)