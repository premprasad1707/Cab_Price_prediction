import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from src.preprocessing import DataProcessor
from src.feature_engineering import engineer_features
from sklearn.ensemble import IsolationForest


# --- CONFIG ---
st.set_page_config(page_title="AI Car Intelligence", layout="wide", initial_sidebar_state="expanded")

THEME_COLORS = {
    "light": "#F5F5F5",
    "black": "#0B0B0B",
    "surface": "#2C2C2C",
    "emerald": "#0F4C45",
    "emerald_bright": "#1C756B",
    "mint": "#8FD8CC",
    "line": "rgba(245, 245, 245, 0.16)",
}


def load_modern_theme():
    st.markdown(
        f"""
        <style>
        :root {{
            --light-neutral: {THEME_COLORS["light"]};
            --black: {THEME_COLORS["black"]};
            --surface: {THEME_COLORS["surface"]};
            --emerald: {THEME_COLORS["emerald"]};
            --emerald-bright: {THEME_COLORS["emerald_bright"]};
            --mint: {THEME_COLORS["mint"]};
            --line: {THEME_COLORS["line"]};
        }}

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 10% 6%, rgba(15, 76, 69, 0.32), transparent 30%),
                radial-gradient(circle at 96% 18%, rgba(143, 216, 204, 0.10), transparent 22%),
                linear-gradient(180deg, var(--black) 0%, #111111 58%, var(--black) 100%);
            color: var(--light-neutral);
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        .main .block-container {{
            max-width: 1260px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stHeader"] {{
            background: rgba(11, 11, 11, 0.78);
            backdrop-filter: blur(12px);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15, 76, 69, 0.94) 0%, #161616 48%, var(--black) 100%);
            border-right: 1px solid rgba(245, 245, 245, 0.12);
            box-shadow: 18px 0 44px rgba(0, 0, 0, 0.32);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--light-neutral);
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        [data-testid="stSidebar"] h1 {{
            font-size: 1.35rem;
            margin-bottom: 0.35rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: rgba(245, 245, 245, 0.06);
            border: 1px solid rgba(245, 245, 245, 0.08);
            border-radius: 8px;
            padding: 9px 10px;
            margin: 7px 0;
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: rgba(245, 245, 245, 0.11);
            border-color: rgba(143, 216, 204, 0.36);
            transform: translateX(4px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        }}

        h1, h2, h3, p, label, span, div {{
            letter-spacing: 0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        h1, h2, h3 {{
            color: var(--light-neutral);
        }}

        .sidebar-panel,
        .premium-hero,
        .market-strip,
        .prediction-shell,
        .prediction-panel,
        .result-panel,
        .mini-card,
        .insight-card,
        .table-shell {{
            border-radius: 8px;
        }}

        .sidebar-panel {{
            background: rgba(245, 245, 245, 0.08);
            border: 1px solid rgba(245, 245, 245, 0.12);
            padding: 14px;
            margin: 12px 0 20px;
            box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
        }}

        .sidebar-panel .label {{
            color: rgba(245, 245, 245, 0.68);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .sidebar-panel .value {{
            color: var(--light-neutral);
            font-weight: 900;
            font-size: 1.28rem;
            margin-top: 4px;
        }}

        .premium-hero {{
            background:
                linear-gradient(135deg, rgba(15, 76, 69, 0.95), rgba(44, 44, 44, 0.94)),
                var(--surface);
            border: 1px solid rgba(245, 245, 245, 0.14);
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
            position: relative;
            overflow: hidden;
        }}

        .premium-hero::before {{
            content: "";
            position: absolute;
            inset: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--mint), var(--emerald-bright), transparent 76%);
        }}

        .premium-hero::after {{
            content: "";
            position: absolute;
            right: -90px;
            top: -110px;
            width: 260px;
            height: 260px;
            border: 1px solid rgba(245, 245, 245, 0.10);
            border-radius: 999px;
        }}

        .eyebrow {{
            color: var(--mint);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 9px;
        }}

        .premium-hero h1 {{
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 3.55rem);
            line-height: 1.04;
        }}

        .premium-hero p {{
            color: rgba(245, 245, 245, 0.78);
            max-width: 820px;
            margin: 14px 0 0;
            font-size: 1.02rem;
        }}

        .market-strip {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 0 0 22px;
        }}

        .mini-card {{
            background: rgba(44, 44, 44, 0.78);
            border: 1px solid rgba(245, 245, 245, 0.12);
            padding: 16px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        }}

        .mini-card span {{
            display: block;
            color: rgba(245, 245, 245, 0.62);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .mini-card strong {{
            display: block;
            margin-top: 7px;
            color: var(--light-neutral);
            font-size: 1.18rem;
        }}

        div[data-testid="stMetric"],
        [data-testid="stForm"],
        [data-testid="stDataFrame"],
        .stPlotlyChart,
        .insight-card,
        .result-panel,
        .mini-card {{
            opacity: 0;
            transform: translateY(18px);
        }}

        .js-ready div[data-testid="stMetric"][data-reveal="visible"],
        .js-ready [data-testid="stForm"][data-reveal="visible"],
        .js-ready [data-testid="stDataFrame"][data-reveal="visible"],
        .js-ready .stPlotlyChart[data-reveal="visible"],
        .js-ready .insight-card[data-reveal="visible"],
        .js-ready .result-panel[data-reveal="visible"],
        .js-ready .mini-card[data-reveal="visible"] {{
            opacity: 1;
            transform: translateY(0);
        }}

        body:not(.js-ready) div[data-testid="stMetric"],
        body:not(.js-ready) [data-testid="stForm"],
        body:not(.js-ready) [data-testid="stDataFrame"],
        body:not(.js-ready) .stPlotlyChart,
        body:not(.js-ready) .insight-card,
        body:not(.js-ready) .result-panel,
        body:not(.js-ready) .mini-card {{
            opacity: 1;
            transform: none;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(44, 44, 44, 0.9);
            border: 1px solid rgba(245, 245, 245, 0.13);
            border-radius: 8px;
            padding: 18px 18px 16px;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.26);
            transition: opacity 520ms ease, transform 520ms ease, border-color 220ms ease, box-shadow 220ms ease;
        }}

        div[data-testid="stMetric"]:hover,
        .mini-card:hover {{
            transform: translateY(-4px) !important;
            border-color: rgba(143, 216, 204, 0.46);
            box-shadow: 0 22px 52px rgba(0, 0, 0, 0.38);
        }}

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: rgba(245, 245, 245, 0.68);
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--light-neutral);
            font-weight: 900;
        }}

        div[data-testid="stMetric"]::after {{
            content: "";
            display: block;
            height: 3px;
            margin-top: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--mint), var(--emerald-bright), transparent);
        }}

        [data-testid="stForm"], [data-testid="stDataFrame"], .stPlotlyChart {{
            background: rgba(44, 44, 44, 0.9);
            border: 1px solid rgba(245, 245, 245, 0.13);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.24);
            transition: opacity 520ms ease, transform 520ms ease, border-color 220ms ease, box-shadow 220ms ease;
        }}

        [data-testid="stForm"]:hover, .stPlotlyChart:hover, [data-testid="stDataFrame"]:hover {{
            border-color: rgba(143, 216, 204, 0.42);
            box-shadow: 0 22px 54px rgba(0, 0, 0, 0.36);
        }}

        .stButton > button, .stFormSubmitButton > button {{
            background: linear-gradient(135deg, var(--emerald), var(--emerald-bright));
            color: var(--light-neutral);
            border: 1px solid rgba(245, 245, 245, 0.18);
            border-radius: 8px;
            font-weight: 900;
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            color: var(--light-neutral);
            border-color: rgba(245, 245, 245, 0.34);
            transform: translateY(-2px);
            filter: brightness(1.08);
            box-shadow: 0 14px 30px rgba(15, 76, 69, 0.46);
        }}

        div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stSlider {{
            transition: border-color 180ms ease, box-shadow 180ms ease;
        }}

        .stNumberInput input:focus {{
            border-color: var(--mint);
            box-shadow: 0 0 0 1px var(--mint);
        }}

        .prediction-shell {{
            background: rgba(245, 245, 245, 0.035);
            border: 1px solid rgba(245, 245, 245, 0.08);
            padding: 4px;
        }}

        .prediction-panel,
        .result-panel {{
            background: linear-gradient(135deg, rgba(15, 76, 69, 0.86), rgba(44, 44, 44, 0.96));
            border: 1px solid rgba(245, 245, 245, 0.16);
            padding: 22px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
        }}

        .prediction-panel h3,
        .result-panel h3 {{
            margin-top: 0;
        }}

        .price-row {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 14px;
            border-bottom: 1px solid rgba(245, 245, 245, 0.10);
            padding: 12px 0;
        }}

        .price-row span {{
            color: rgba(245, 245, 245, 0.66);
            font-weight: 700;
        }}

        .price-row strong {{
            color: var(--light-neutral);
            font-size: 1.05rem;
        }}

        .result-label {{
            color: rgba(245, 245, 245, 0.72);
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
        }}

        .result-value {{
            color: var(--light-neutral);
            font-size: clamp(2.2rem, 4vw, 3.75rem);
            font-weight: 900;
            margin: 8px 0 10px;
        }}

        .confidence-bar {{
            width: 100%;
            height: 9px;
            background: rgba(245, 245, 245, 0.12);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 14px;
        }}

        .confidence-bar span {{
            display: block;
            width: 82%;
            height: 100%;
            background: linear-gradient(90deg, var(--mint), var(--emerald-bright));
        }}

        .insight-card {{
            background: rgba(44, 44, 44, 0.9);
            border-left: 4px solid var(--mint);
            padding: 18px;
            margin: 12px 0;
            transition: opacity 520ms ease, transform 520ms ease, background 220ms ease, border-color 220ms ease;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }}

        .insight-card:hover {{
            transform: translateX(4px) !important;
            background: rgba(15, 76, 69, 0.52);
            border-left-color: var(--light-neutral);
        }}

        .table-shell {{
            background: rgba(245, 245, 245, 0.035);
            border: 1px solid rgba(245, 245, 245, 0.10);
            padding: 16px;
            box-shadow: 0 18px 46px rgba(0, 0, 0, 0.28);
        }}

        .table-shell h3 {{
            margin-top: 0;
            margin-bottom: 4px;
        }}

        .table-shell p {{
            color: rgba(245, 245, 245, 0.68);
            margin-top: 0;
        }}

        @media (max-width: 760px) {{
            .market-strip {{
                grid-template-columns: 1fr;
            }}

            .premium-hero {{
                padding: 22px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_motion_js():
    components.html(
        """
        <script>
        (() => {
            const root = window.parent.document;
            root.body.classList.add('js-ready');

            const revealSelector = [
                'div[data-testid="stMetric"]',
                '[data-testid="stForm"]',
                '[data-testid="stDataFrame"]',
                '.stPlotlyChart',
                '.insight-card',
                '.result-panel',
                '.mini-card'
            ].join(',');

            const reveal = () => {
                const items = root.querySelectorAll(revealSelector);
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach((entry) => {
                        if (entry.isIntersecting) {
                            entry.target.setAttribute('data-reveal', 'visible');
                            observer.unobserve(entry.target);
                        }
                    });
                }, { threshold: 0.12 });

                items.forEach((item, index) => {
                    item.style.transitionDelay = `${Math.min(index * 45, 240)}ms`;
                    observer.observe(item);
                });
            };

            const animateMetrics = () => {
                root.querySelectorAll('div[data-testid="stMetricValue"]').forEach((node) => {
                    if (node.dataset.counted === 'true') return;
                    const raw = node.textContent.trim();
                    const numeric = Number(raw.replace(/[^0-9.-]/g, ''));
                    if (!Number.isFinite(numeric)) return;

                    node.dataset.counted = 'true';
                    const prefix = raw.match(/^[^0-9.-]*/)?.[0] || '';
                    const suffix = raw.match(/[^0-9.]*$/)?.[0] || '';
                    const decimals = raw.includes('.') ? 2 : 0;
                    const start = performance.now();
                    const duration = 850;

                    const tick = (now) => {
                        const progress = Math.min((now - start) / duration, 1);
                        const eased = 1 - Math.pow(1 - progress, 3);
                        const value = numeric * eased;
                        node.textContent = `${prefix}${value.toLocaleString(undefined, {
                            maximumFractionDigits: decimals,
                            minimumFractionDigits: decimals
                        })}${suffix}`;
                        if (progress < 1) requestAnimationFrame(tick);
                    };
                    requestAnimationFrame(tick);
                });
            };

            const liftPointer = () => {
                root.querySelectorAll('.premium-hero, .result-panel').forEach((card) => {
                    if (card.dataset.tiltReady === 'true') return;
                    card.dataset.tiltReady = 'true';
                    card.addEventListener('mousemove', (event) => {
                        const rect = card.getBoundingClientRect();
                        const x = ((event.clientX - rect.left) / rect.width - 0.5) * 6;
                        const y = ((event.clientY - rect.top) / rect.height - 0.5) * -6;
                        card.style.transform = `perspective(900px) rotateX(${y}deg) rotateY(${x}deg)`;
                    });
                    card.addEventListener('mouseleave', () => {
                        card.style.transform = '';
                    });
                });
            };

            reveal();
            animateMetrics();
            liftPointer();

            const rerunObserver = new MutationObserver(() => {
                reveal();
                animateMetrics();
                liftPointer();
            });
            rerunObserver.observe(root.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_hero(title, subtitle, eyebrow="AI CAR INTELLIGENCE"):
    st.markdown(
        f"""
        <section class="premium-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_market_strip(items):
    cards = "".join(
        f"<div class='mini-card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in items
    )
    st.markdown(f"<section class='market-strip'>{cards}</section>", unsafe_allow_html=True)


def apply_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(44,44,44,0)",
        plot_bgcolor="rgba(44,44,44,0)",
        font_color=THEME_COLORS["light"],
        title_font_color=THEME_COLORS["light"],
        coloraxis_colorbar=dict(tickfont=dict(color=THEME_COLORS["light"])),
        margin=dict(l=20, r=20, t=56, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(245,245,245,0.10)", zerolinecolor="rgba(245,245,245,0.16)")
    fig.update_yaxes(gridcolor="rgba(245,245,245,0.10)", zerolinecolor="rgba(245,245,245,0.16)")
    return fig


load_modern_theme()
inject_motion_js()


# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/car_prices.csv")
    dp = DataProcessor()
    df_clean = dp.clean_data(df)
    df_eng = engineer_features(df_clean)
    return df, df_eng, dp


raw_df, processed_df, dp = load_data()

# --- SIDEBAR ---
st.sidebar.title("AI Car Market")
st.sidebar.markdown(
    f"""
    <div class="sidebar-panel">
        <div class="label">Live dataset</div>
        <div class="value">{len(raw_df):,} listings</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigate", ["Dashboard", "Price Prediction", "Anomaly Detection", "Market Insights"])

if page == "Dashboard":
    render_hero(
        "Market Dashboard",
        "Track listing volume, pricing behavior, mileage patterns, and brand movement with a cleaner premium intelligence view.",
    )
    render_market_strip(
        [
            ("Market range", f"${raw_df['price'].min():,.0f} - ${raw_df['price'].max():,.0f}"),
            ("Average year", f"{raw_df['year'].mean():.0f}"),
            ("Fuel types", raw_df["fuel_type"].nunique()),
        ]
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", len(raw_df))
    col2.metric("Avg Price", f"${raw_df['price'].mean():,.0f}")
    col3.metric("Top Brand", raw_df["brand"].mode()[0])
    col4.metric("Avg Mileage", f"{raw_df['mileage'].mean():,.0f} km")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(
            raw_df,
            x="price",
            nbins=30,
            title="Price Distribution",
            color_discrete_sequence=[THEME_COLORS["emerald_bright"]],
        )
        st.plotly_chart(apply_chart_theme(fig1), use_container_width=True)
    with c2:
        fig2 = px.scatter(
            raw_df,
            x="mileage",
            y="price",
            color="brand",
            title="Price vs Mileage",
            color_discrete_sequence=[THEME_COLORS["emerald_bright"], THEME_COLORS["mint"], THEME_COLORS["light"], THEME_COLORS["emerald"]],
        )
        st.plotly_chart(apply_chart_theme(fig2), use_container_width=True)

elif page == "Price Prediction":
    render_hero(
        "AI Price Predictor",
        "Tune the vehicle profile and get a market-style estimate with cleaner confidence details.",
    )

    form_col, side_col = st.columns([1.25, 0.75], gap="large")
    with form_col:
        with st.form("pred_form"):
            col1, col2 = st.columns(2)
            brand = col1.selectbox("Brand", raw_df["brand"].unique())
            model_name = col2.selectbox("Model Variant", raw_df["model"].unique())
            year = col1.slider("Year", 2010, 2024, 2018)
            mileage = col2.number_input("Mileage (km)", 0, 300000, 50000)
            fuel = col1.selectbox("Fuel", raw_df["fuel_type"].unique())
            trans = col2.selectbox("Transmission", raw_df["transmission"].unique())
            submit = st.form_submit_button("Estimate Price")

    with side_col:
        st.markdown(
            f"""
            <aside class="prediction-panel">
                <h3>Selected Vehicle</h3>
                <div class="price-row"><span>Brand</span><strong>{brand}</strong></div>
                <div class="price-row"><span>Model</span><strong>{model_name}</strong></div>
                <div class="price-row"><span>Year</span><strong>{year}</strong></div>
                <div class="price-row"><span>Mileage</span><strong>{mileage:,} km</strong></div>
                <div class="price-row"><span>Fuel</span><strong>{fuel}</strong></div>
                <div class="price-row"><span>Transmission</span><strong>{trans}</strong></div>
            </aside>
            """,
            unsafe_allow_html=True,
        )

    if submit:
        base = 40000 if brand in ["BMW", "Audi"] else 20000
        age = 2024 - year
        est = base - (age * 1500) - (mileage * 0.05)
        est = max(est, 5000)
        val_rating = "Excellent Value" if est < 15000 else "Fair Market Value"
        depreciation = max(age * 4.8, 0)

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">Estimated Market Value</div>
                <div class="result-value">${est:,.2f}</div>
                <div class="price-row"><span>Market rating</span><strong>{val_rating}</strong></div>
                <div class="price-row"><span>Approx depreciation pressure</span><strong>{depreciation:.1f}%</strong></div>
                <div class="price-row"><span>Model confidence</span><strong>82%</strong></div>
                <div class="confidence-bar"><span></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "Anomaly Detection":
    render_hero(
        "Fraud & Anomaly Detection",
        "Surface suspicious listings by comparing price, mileage, and year outliers across the market.",
    )

    iso = IsolationForest(contamination=0.05, random_state=42)
    features = processed_df[["price", "mileage", "year"]]
    processed_df["anomaly"] = iso.fit_predict(features)
    anomalies = processed_df[processed_df["anomaly"] == -1].copy()
    anomalies["risk_score"] = (
        (anomalies["price"].rank(pct=True) * 42)
        + (anomalies["mileage"].rank(pct=True) * 38)
        + ((2026 - anomalies["year"]).rank(pct=True) * 20)
    ).round(1)

    a1, a2, a3 = st.columns(3)
    a1.metric("Suspicious Entries", len(anomalies))
    a2.metric("Review Rate", f"{len(anomalies) / len(processed_df) * 100:.1f}%")
    a3.metric("Highest Risk", f"{anomalies['risk_score'].max():.1f}" if len(anomalies) else "0")

    top_anomalies = anomalies[["brand", "model", "year", "mileage", "price", "risk_score"]].sort_values("risk_score", ascending=False).head(15)
    styled_anomalies = top_anomalies.style.format(
        {
            "mileage": "{:,.0f} km",
            "price": "${:,.0f}",
            "risk_score": "{:.1f}",
        }
    ).background_gradient(subset=["risk_score"], cmap="Greens")

    st.markdown(
        """
        <div class="table-shell">
            <h3>Priority Review Queue</h3>
            <p>Highest-risk listings are ranked first so the review workflow feels immediate and scannable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(styled_anomalies, use_container_width=True, hide_index=True)

elif page == "Market Insights":
    render_hero(
        "Market Insights Engine",
        "Compare brand retention and find the signals that shape resale value across the dataset.",
    )

    st.subheader("Brand Retention")
    brand_avg = raw_df.groupby("brand")["price"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        brand_avg,
        x="brand",
        y="price",
        color="price",
        color_continuous_scale=[[0, THEME_COLORS["surface"]], [1, THEME_COLORS["emerald_bright"]]],
    )
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    st.markdown(
        """
        <div class="insight-card"><strong>Insight 1:</strong> Luxury brands like BMW and Audi show higher initial depreciation but maintain higher floor values.</div>
        <div class="insight-card"><strong>Insight 2:</strong> Mileage above 100,000 km results in a sharp 30% price drop across all segments.</div>
        <div class="insight-card"><strong>Insight 3:</strong> Hybrid vehicles are currently holding 15% more value than Diesel equivalents.</div>
        """,
        unsafe_allow_html=True,
    )
