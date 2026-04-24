import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import sqlite3, json, warnings, random
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WFM Forecasting Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── BRAND STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1A1C22; }
    [data-testid="stSidebar"] {
        background-color: #21242C;
        border-right: 1px solid #2E3245;
    }
    [data-testid="stSidebar"] * { color: #F4EEE6 !important; }
    h1, h2, h3 { color: #A884FF !important; }
    p, label, .stMarkdown { color: #F4EEE6; }
    .stButton > button {
        background-color: #7A4DDB !important;
        color: #F4EEE6 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        width: 100%;
    }
    .stButton > button:hover { background-color: #5A34A3 !important; }
    [data-testid="metric-container"] {
        background-color: #21242C;
        border: 1px solid #2E3245;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #A884FF !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F4EEE6 !important; }
    hr { border-color: #2E3245 !important; }
    .stDataFrame { background-color: #21242C; }
    .stSelectbox > div > div,
    .stSlider > div { background-color: transparent; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── BRAND COLORS ────────────────────────────────────────────────────────────
C = {
    'midnight':  '#1A1C22',
    'surface':   '#21242C',
    'surface2':  '#1E212A',
    'border':    '#2E3245',
    'violet':    '#7A4DDB',
    'purple':    '#5A34A3',
    'lavender':  '#A884FF',
    'lilac':     '#DCCBFF',
    'ivory':     '#F4EEE6',
    'teal':      '#3FA7A3',
    'gold':      '#D6B85A',
    'chartreuse':'#A6B85C',
    'terracotta':'#C65D43',
}

# ─── SYNTHETIC DATA ──────────────────────────────────────────────────────────
# ─── WFM TRIVIA ──────────────────────────────────────────────────────────────
WFM_FACTS = [
    (
        "The Man Behind the Math",
        "Agner Krarup Erlang, a Danish mathematician, developed the Erlang C formula in 1917 "
        "while working for the Copenhagen Telephone Company. He died in 1929 — decades before "
        "the modern contact center existed — never knowing his math would become the backbone "
        "of an entire global industry."
    ),
    (
        "The 80/20 Rule Has No Scientific Basis",
        "The ubiquitous '80% of calls answered in 20 seconds' service level target was never "
        "derived from research. It was reportedly chosen by a telecom manager in the 1980s "
        "because it simply felt right. It stuck, became the industry default, and has been "
        "questioned — and quietly accepted — ever since."
    ),
    (
        "IEX TotalView: The OG WFM Platform",
        "IEX Corporation was founded in 1988 and launched TotalView, one of the first software "
        "platforms dedicated entirely to contact center workforce management. It was acquired by "
        "NICE Systems in 2005 and lives on today as NICE IEX WFM — still running in thousands "
        "of centers worldwide."
    ),
    (
        "The First Call Center",
        "The first recognized call center was established by Birmingham Press and Mail in the "
        "United Kingdom in 1965. Agents handled inbound calls on a GEC PABX A private branch "
        "exchange. Workforce management, at that point, was a clipboard and a prayer."
    ),
    (
        "Shrinkage Came from Retail",
        "The term 'shrinkage' in WFM was borrowed directly from retail inventory management, "
        "where it described goods lost to theft or damage. Someone in the 1980s noticed that "
        "scheduled agent time 'shrinks' before it ever hits the phones — and the name stuck "
        "across the entire industry."
    ),
    (
        "The First ACD Went to an Airline",
        "The first Automatic Call Distributor (ACD) was installed at Continental Airlines in "
        "1973 to route flight reservation calls. Before ACDs, calls were manually transferred "
        "by switchboard operators — workforce management was literally just headcount and hope."
    ),
    (
        "Occupancy Has a Breaking Point",
        "Research consistently shows that agent burnout and error rates rise sharply when "
        "occupancy exceeds 85–88%. Yet many contact centers routinely target 90%+. The math "
        "says you're borrowing productivity from tomorrow — and tomorrow always collects."
    ),
    (
        "Erlang Never Saw a Contact Center",
        "When Erlang published his queuing theory, the concept of a 'contact center' didn't "
        "exist. His goal was simply to figure out how many telephone circuits Copenhagen needed. "
        "The Erlang C formula he wrote by hand is still used — largely unchanged — in every "
        "modern WFM platform today."
    ),
    (
        "Chat Changed Everything",
        "When live chat became mainstream in the late 1990s, WFM teams had to rethink "
        "everything. Erlang C assumes one interaction per agent at a time. Concurrent chat "
        "handling broke that assumption entirely, forcing vendors to build new concurrency "
        "models almost from scratch."
    ),
    (
        "Frederick Taylor's Ghost",
        "The scientific principles behind WFM trace back to Frederick Winslow Taylor's "
        "'Scientific Management' theory from 1911 — optimizing factory worker output through "
        "precise measurement and scheduling. Contact centers essentially applied his factory "
        "floor logic to phone lines, a century later."
    ),
    (
        "Genesys and the World Wide Web",
        "Genesys was founded in 1990 — the same year Tim Berners-Lee invented the World Wide "
        "Web. The two technologies grew up together, and eventually collided: today's contact "
        "centers run almost entirely on the web infrastructure that didn't exist when modern "
        "WFM was born."
    ),
    (
        "Half-Hour vs. 15-Minute Intervals",
        "Early WFM systems used 30-minute scheduling intervals because that's what the hardware "
        "could handle. As processing power improved through the 1990s, 15-minute intervals "
        "became the standard — cutting scheduling error in half overnight and making every "
        "WFM analyst's job simultaneously more precise and more complicated."
    ),
]

@st.cache_data(show_spinner=False)
def generate_data():
    np.random.seed(42)
    queues = {
        'General Inbound':    {'channel': 'Voice',  'base': 2800, 'aht': 340, 'trend': 0.0020},
        'Billing & Payments': {'channel': 'Voice',  'base': 1900, 'aht': 420, 'trend': 0.0010},
        'Technical Support':  {'channel': 'Voice',  'base': 1400, 'aht': 480, 'trend': 0.0030},
        'General Chat':       {'channel': 'Chat',   'base': 1200, 'aht': 720, 'trend': 0.0040},
        'Sales Chat':         {'channel': 'Chat',   'base':  900, 'aht': 660, 'trend': 0.0050},
        'Technical Chat':     {'channel': 'Chat',   'base':  500, 'aht': 840, 'trend': 0.0020},
        'General Email':      {'channel': 'Email',  'base':  650, 'aht': 420, 'trend': 0.0010},
        'Billing Email':      {'channel': 'Email',  'base':  450, 'aht': 380, 'trend': 0.0005},
        'Technical Email':    {'channel': 'Email',  'base':  320, 'aht': 500, 'trend': 0.0020},
    }
    dow_f = {0:1.15, 1:1.08, 2:1.02, 3:0.98, 4:0.87, 5:0.55, 6:0.30}
    dates = pd.date_range('2024-04-22', '2025-04-21', freq='D')
    rows = []
    for q, p in queues.items():
        wnoise = {}
        for i, d in enumerate(dates):
            wk = d.isocalendar()[1]
            if wk not in wnoise:
                wnoise[wk] = np.clip(np.random.normal(1.0, 0.05), 0.88, 1.12)
            dow = d.dayofweek
            if p['channel'] in ('Voice','Chat') and dow == 6:
                continue
            tf = 1 + p['trend'] * i
            vol = int(np.random.poisson(max(1, p['base'] * dow_f[dow] * wnoise[wk] * tf)))
            aht = max(60, np.random.normal(p['aht'], p['aht'] * 0.05))
            rows.append({
                'date':              pd.Timestamp(d),
                'queue':             q,
                'channel':           p['channel'],
                'offered':           vol,
                'handled':           int(vol * np.random.uniform(0.92, 0.98)),
                'abandoned':         int(vol * np.random.uniform(0.02, 0.08)),
                'aht_seconds':       round(aht, 1),
                'service_level_pct': round(np.clip(np.random.normal(82, 5), 50, 99), 1),
                'occupancy_pct':     round(np.clip(np.random.normal(78, 5), 50, 95), 1),
            })
    return pd.DataFrame(rows)

# ─── DATABASE ────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('wfm_forecasts.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at        TEXT,
            queue         TEXT,
            model         TEXT,
            units         TEXT,
            horizon       INTEGER,
            wmape         REAL,
            parameters    TEXT,
            forecast_json TEXT
        )
    """)
    conn.commit()
    return conn

def save_forecast(queue, model, units, horizon, wmape, params, fdf):
    conn = init_db()
    conn.execute(
        "INSERT INTO forecast_runs (run_at,queue,model,units,horizon,wmape,parameters,forecast_json) VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), queue, model, units, horizon,
         round(float(wmape), 6) if wmape is not None else None,
         json.dumps(params), fdf.to_json())
    )
    conn.commit()
    conn.close()

def load_saved():
    conn = init_db()
    try:
        df = pd.read_sql("SELECT * FROM forecast_runs ORDER BY run_at DESC LIMIT 25", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _drop_partial(s_raw, s_resampled, units):
    """Remove the last resampled bucket if it only contains partial-period data."""
    last_day = s_raw.index[-1]
    if units == 'Weekly':
        # Weekly buckets end on Sunday (dayofweek == 6)
        if last_day.dayofweek != 6:
            return s_resampled.iloc[:-1]
    elif units == 'Monthly':
        # Monthly buckets end on the last calendar day of the month
        month_end = last_day + pd.offsets.MonthEnd(0)
        if last_day != month_end:
            return s_resampled.iloc[:-1]
    return s_resampled

def aggregate(df, queue, units):
    s = df[df['queue'] == queue].set_index('date')['offered'].sort_index()
    if units == 'Weekly':
        return _drop_partial(s, s.resample('W').sum(), units)
    if units == 'Monthly':
        return _drop_partial(s, s.resample('MS').sum(), units)
    return s  # Daily

def aggregate_channel(df, channel, units):
    """Sum all queues within a channel — each channel runs independently."""
    s = df[df['channel'] == channel].groupby('date')['offered'].sum().sort_index()
    if units == 'Weekly':
        return _drop_partial(s, s.resample('W').sum(), units)
    if units == 'Monthly':
        return _drop_partial(s, s.resample('MS').sum(), units)
    return s

CHANNEL_COLORS = {
    'Voice': {'line': C['violet'],    'fill': 'rgba(122, 77, 219, 0.15)'},
    'Chat':  {'line': C['teal'],      'fill': 'rgba( 63,167, 163, 0.15)'},
    'Email': {'line': C['gold'],      'fill': 'rgba(214,184,  90, 0.15)'},
}

def future_index(last, units, n):
    if units == 'Daily':
        return pd.date_range(last + timedelta(days=1), periods=n, freq='D')
    if units == 'Weekly':
        return pd.date_range(last + timedelta(weeks=1), periods=n, freq='W')
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=n, freq='MS')

def wmape(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    mask = a > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.sum(np.abs(a[mask] - p[mask])) / np.sum(a[mask]))

# ─── SEASONAL HELPERS ────────────────────────────────────────────────────────
def get_seasonal_factors(series, units):
    """Derive day-of-week (daily) or month-of-year (monthly) scaling factors."""
    try:
        overall = series.mean()
        if overall <= 0:
            return None, {}
        if units == 'Daily' and len(series) >= 14:
            avg_dow = series.groupby(series.index.dayofweek).mean()
            return 'dow', (avg_dow / overall).to_dict()
        if units == 'Weekly' and len(series) >= 12:
            avg_woy = series.groupby(series.index.isocalendar().week.astype(int)).mean()
            return 'woy', (avg_woy / overall).to_dict()
        if units == 'Monthly' and len(series) >= 6:
            avg_moy = series.groupby(series.index.month).mean()
            return 'moy', (avg_moy / overall).to_dict()
    except Exception:
        pass
    return None, {}

def apply_seasonal(idx, base_vals, s_type, factors):
    """Scale an array of base values by seasonal factor for each date."""
    base = np.asarray(base_vals, dtype=float)
    result = []
    for i, dt in enumerate(idx):
        bv = base[i] if i < len(base) else base[-1]
        if s_type == 'dow':
            f = factors.get(dt.dayofweek, 1.0)
        elif s_type == 'woy':
            f = factors.get(int(dt.isocalendar()[1]), 1.0)
        elif s_type == 'moy':
            f = factors.get(dt.month, 1.0)
        else:
            f = 1.0
        result.append(max(0.0, bv * f))
    return np.array(result)

# ─── FORECAST MODELS ─────────────────────────────────────────────────────────
def m_hist_avg(series, n, units, lookback=10):
    base  = float(series.iloc[-lookback:].mean())
    idx   = future_index(series.index[-1], units, n)
    stype, factors = get_seasonal_factors(series, units)
    vals  = apply_seasonal(idx, np.full(n, base), stype, factors)
    return pd.Series(vals, index=idx), {'lookback_periods': lookback}

def m_weighted_avg(series, n, units, decay=0.82):
    vals  = series.values[-14:]
    w     = np.array([decay**i for i in range(len(vals)-1, -1, -1)])
    w    /= w.sum()
    base  = float(np.dot(w, vals))
    idx   = future_index(series.index[-1], units, n)
    stype, factors = get_seasonal_factors(series, units)
    result = apply_seasonal(idx, np.full(n, base), stype, factors)
    return pd.Series(result, index=idx), {'decay': decay}

def m_linear_reg(series, n, units):
    x  = np.arange(len(series), dtype=float)
    y  = series.values.astype(float)
    c  = np.polyfit(x, y, 1)
    fx = np.arange(len(series), len(series) + n, dtype=float)
    trend = np.maximum(0, np.polyval(c, fx))
    idx   = future_index(series.index[-1], units, n)
    stype, factors = get_seasonal_factors(series, units)
    # Scale each trend value relative to the trend mean so shape is preserved
    tmean = trend.mean() if trend.mean() > 0 else 1.0
    scaled = []
    for i, dt in enumerate(idx):
        if stype == 'dow':
            f = factors.get(dt.dayofweek, 1.0)
        elif stype == 'woy':
            f = factors.get(int(dt.isocalendar()[1]), 1.0)
        elif stype == 'moy':
            f = factors.get(dt.month, 1.0)
        else:
            f = 1.0
        scaled.append(max(0.0, trend[i] + (f - 1.0) * tmean))
    return pd.Series(scaled, index=idx), \
           {'slope': round(c[0], 3), 'intercept': round(c[1], 3)}

def m_exp_smooth(series, n, units):
    sp = {'Daily': 7, 'Weekly': 4, 'Monthly': 12}[units]
    use_seasonal = len(series) >= sp * 3
    idx = future_index(series.index[-1], units, n)
    for init_method in ('heuristic', 'estimated', 'legacy-heuristic'):
        try:
            fit = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add' if use_seasonal else None,
                seasonal_periods=sp if use_seasonal else None,
                initialization_method=init_method
            ).fit(optimized=True, disp=False)
            fc = fit.forecast(n).clip(lower=0)
            fc.index = idx
            return fc, {'seasonal': use_seasonal, 'periods': sp, 'init': init_method}
        except Exception:
            continue
    # Final fallback: weighted avg with seasonality
    return m_weighted_avg(series, n, units)

def m_arima(series, n, units):
    idx = future_index(series.index[-1], units, n)
    # Try SARIMA with weekly seasonality for daily data, else plain ARIMA
    orders = [(2, 1, 2), (1, 1, 1), (2, 1, 1), (1, 1, 0)]
    for order in orders:
        try:
            fit = ARIMA(series, order=order).fit()
            fc  = fit.forecast(n).clip(lower=0)
            fc.index = idx
            # Apply seasonal correction on top of ARIMA output
            stype, factors = get_seasonal_factors(series, units)
            if stype:
                fc_adj = apply_seasonal(idx, fc.values, stype, factors)
                # Blend: 60% ARIMA, 40% seasonal-adjusted to preserve ARIMA dynamics
                fc = pd.Series(0.6 * fc.values + 0.4 * fc_adj, index=idx)
            return fc, {'order': str(order)}
        except Exception:
            continue
    return m_exp_smooth(series, n, units)

MODEL_MAP = {
    'Historical Average':   m_hist_avg,
    'Weighted Average':     m_weighted_avg,
    'Linear Regression':    m_linear_reg,
    'Exponential Smoothing':m_exp_smooth,
    'ARIMA':                m_arima,
}

def m_auto_select(series, n, units):
    holdout = min(max(4, int(len(series) * 0.15)), 10)
    if len(series) < holdout + 8:
        fc, params = m_weighted_avg(series, n, units)
        return fc, params, 'Weighted Average', {}
    train  = series.iloc[:-holdout]
    actual = series.iloc[-holdout:].values
    scores = {}
    for name, fn in MODEL_MAP.items():
        try:
            pred, _ = fn(train, holdout, units)
            scores[name] = wmape(actual, pred.values[:len(actual)])
        except Exception:
            scores[name] = 9.99
    winner = min(scores, key=scores.get)
    fc, params = MODEL_MAP[winner](series, n, units)
    params['competition_scores'] = {k: round(v*100, 2) for k, v in scores.items()}
    return fc, params, winner, scores

# ─── CHART ───────────────────────────────────────────────────────────────────
def make_chart(history, forecast, queue, model_label, units):
    ci = forecast * 0.10

    # Bridge point — last historical value prepended to forecast so lines connect
    bridge_x = [history.index[-1]]  + list(forecast.index)
    bridge_y = [history.values[-1]] + list(forecast.values)
    ci_upper  = [history.values[-1]] + list((forecast + ci).values)
    ci_lower  = [history.values[-1]] + list((forecast - ci).values)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        name='Historical', mode='lines',
        line=dict(color=C['purple'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=bridge_x + bridge_x[::-1],
        y=ci_upper  + ci_lower[::-1],
        fill='toself', fillcolor='rgba(122,77,219,0.18)',
        line=dict(color='rgba(0,0,0,0)'), name='±10% band'
    ))
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        name=f'Forecast ({model_label})', mode='lines+markers',
        line=dict(color=C['violet'], width=2.5, dash='dash'),
        marker=dict(size=5, color=C['lavender'])
    ))
    fig.add_vline(
        x=history.index[-1].timestamp() * 1000,
        line_dash='dot', line_color=C['teal'], line_width=1.5,
        annotation_text='  Forecast start',
        annotation_font_color=C['teal'],
        annotation_font_size=11,
        annotation_position='top right'
    )
    fig.update_layout(
        title=dict(
            text=f'<b>{queue}</b> — {units} Offered Volume Forecast',
            font=dict(color=C['lavender'], size=16)
        ),
        paper_bgcolor=C['midnight'],
        plot_bgcolor=C['surface'],
        font=dict(color=C['ivory'], family='system-ui'),
        xaxis=dict(gridcolor=C['border'], linecolor=C['border'], title=units),
        yaxis=dict(gridcolor=C['border'], linecolor=C['border'], title='Offered Contacts'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=C['ivory']),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
        ),
        height=430,
        margin=dict(l=10, r=10, t=60, b=20),
        hovermode='x unified'
    )
    return fig

# ─── MULTI-CHANNEL CHART ─────────────────────────────────────────────────────
def _hov(v):
    """Format a single numeric value for chart hover tooltips."""
    v = float(v)
    return f"{v/1000:.1f}K" if v >= 10000 else f"{v:,.0f}"

def make_multichannel_chart(channel_results, display_window, units, model_label):
    fig = go.Figure()
    last_hist_date = None

    for channel, res in channel_results.items():
        if res is None:
            continue
        col     = CHANNEL_COLORS[channel]
        display = res['series'].iloc[-display_window:]
        fc      = res['fc']
        ci      = fc * 0.10

        bridge_x = [display.index[-1]]  + list(fc.index)
        bridge_y = [display.values[-1]] + list(fc.values)
        ci_upper = [display.values[-1]] + list((fc + ci).values)
        ci_lower = [display.values[-1]] + list((fc - ci).values)

        if last_hist_date is None:
            last_hist_date = display.index[-1]

        fig.add_trace(go.Scatter(
            x=display.index, y=display.values,
            name=f'{channel}', legendgroup=channel,
            mode='lines', line=dict(color=col['line'], width=2),
            customdata=[_hov(v) for v in display.values],
            hovertemplate='%{customdata}<extra>' + channel + '</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=bridge_x + bridge_x[::-1],
            y=ci_upper  + ci_lower[::-1],
            fill='toself', fillcolor=col['fill'],
            line=dict(color='rgba(0,0,0,0)'),
            legendgroup=channel, showlegend=False,
            hoverinfo='skip',
            name=f'{channel} ±10%'
        ))
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            name=f'{channel} forecast',
            legendgroup=channel, showlegend=True,
            mode='lines+markers',
            line=dict(color=col['line'], width=2.5, dash='dash'),
            marker=dict(size=5, color=col['line']),
            customdata=[_hov(v) for v in bridge_y],
            hovertemplate='%{customdata}<extra>' + channel + ' forecast</extra>'
        ))

    if last_hist_date:
        fig.add_vline(
            x=last_hist_date.timestamp() * 1000,
            line_dash='dot', line_color=C['teal'], line_width=1.5,
            annotation_text='  Forecast start',
            annotation_font_color=C['teal'],
            annotation_font_size=11,
            annotation_position='top right'
        )

    fig.update_layout(
        title=dict(
            text=f'<b>Channel Forecasts</b> — {units} Offered Volume · {model_label}',
            font=dict(color=C['lavender'], size=16)
        ),
        paper_bgcolor=C['midnight'],
        plot_bgcolor=C['surface'],
        font=dict(color=C['ivory'], family='system-ui'),
        xaxis=dict(gridcolor=C['border'], linecolor=C['border'], title=units),
        yaxis=dict(gridcolor=C['border'], linecolor=C['border'], title='Offered Contacts'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=C['ivory']),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
        ),
        height=450,
        margin=dict(l=10, r=10, t=60, b=20),
        hovermode='x unified'
    )
    return fig

# ─── APP ─────────────────────────────────────────────────────────────────────
def main():
    df   = generate_data()
    init_db()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Forecast Settings")
        st.divider()

        # Queue selector — for KPI row context only, not used in forecasting
        queue = st.selectbox('Queue (KPI reference)', sorted(df['queue'].unique()))

        model = st.selectbox('Forecast Model', [
            'Auto-Select Model',
            '─────────────────',
            'Historical Average',
            'Weighted Average',
            'Linear Regression',
            'Exponential Smoothing',
            'ARIMA',
        ])

        units = st.selectbox('Forecast Units', ['Daily', 'Weekly', 'Monthly'])

        hlimits  = {'Daily': 365, 'Weekly': 52, 'Monthly': 24}
        hdefault = {'Daily': 14,  'Weekly': 8,  'Monthly': 3}
        horizon  = st.number_input(
            f'Horizon ({units.lower()})',
            min_value=1,
            max_value=hlimits[units],
            value=hdefault[units],
            step=1
        )

        st.divider()
        run_btn = st.button('▶  Run Forecast', use_container_width=True)
        st.caption('Runs Voice, Chat, and Email independently.')
        st.divider()

        q_data = df[df['queue'] == queue]
        st.markdown('**Data range**')
        st.caption(f"{df['date'].min().strftime('%b %d, %Y')} → {df['date'].max().strftime('%b %d, %Y')}")
        st.caption(f"**{len(df):,}** total records · **{df['queue'].nunique()}** queues")

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown("# 📊 WFM Forecasting Suite")
    st.caption(f"Model: **{model}** · Units: **{units}** · Horizon: **{horizon} {units.lower()}**")

    # ── KPI ROW ──────────────────────────────────────────────────────────────
    recent_cutoff = q_data['date'].max() - timedelta(days=30)
    recent = q_data[q_data['date'] >= recent_cutoff]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Avg Daily Volume (30d)', f"{recent['offered'].mean():,.0f}")
    c2.metric('Avg Service Level',      f"{recent['service_level_pct'].mean():.1f}%")
    c3.metric('Avg AHT',                f"{recent['aht_seconds'].mean():.0f}s")
    c4.metric('Avg Occupancy',          f"{recent['occupancy_pct'].mean():.1f}%")

    st.divider()

    # ── FORECAST EXECUTION ───────────────────────────────────────────────────
    if model == '─────────────────':
        st.info('Please select a forecast model from the sidebar.')
        return

    CHANNELS = ['Voice', 'Chat', 'Email']

    def run_channel(series, model, horizon, units):
        """Run selected model for one channel's aggregated series."""
        all_fc = {}
        if model == 'Auto-Select Model':
            fc, params, winner, scores = m_auto_select(series, horizon, units)
            model_label = f'Auto-Select → {winner}'
            holdout = min(max(4, int(len(series) * 0.15)), 10)
            train, actual = series.iloc[:-holdout], series.iloc[-holdout:].values
            pred_h, _ = MODEL_MAP[winner](train, holdout, units)
            err = wmape(actual, pred_h.values[:len(actual)])
            # Run every model so we can show all forecasts in the results table
            for name, fn in MODEL_MAP.items():
                try:
                    f, _ = fn(series, horizon, units)
                    all_fc[name] = f.values.round(0).astype(int)
                except Exception:
                    pass
        else:
            fn = MODEL_MAP[model]
            fc, params = fn(series, horizon, units)
            model_label = winner = model
            scores = None
            holdout = min(8, max(1, len(series) // 5))
            train, actual = series.iloc[:-holdout], series.iloc[-holdout:].values
            pred_h, _ = fn(train, holdout, units)
            err = wmape(actual, pred_h.values[:len(actual)])
        fdf = pd.DataFrame({'date': fc.index,
                            'forecasted_offered': fc.values.round(0).astype(int)})
        return dict(series=series, fc=fc, params=params, err=err,
                    winner=winner, scores=scores, model_label=model_label,
                    fdf=fdf, all_fc=all_fc)

    if run_btn:
        fact_title, fact_text = random.choice(WFM_FACTS)
        _overlay = st.empty()
        _overlay.markdown(f"""
<style>
@keyframes wfm-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.55; transform: scale(0.92); }}
}}
@keyframes wfm-bar {{
    0%   {{ width: 0%; }}
    100% {{ width: 92%; }}
}}
</style>
<div style="
    position:fixed; inset:0; z-index:99999;
    background:rgba(16,17,22,0.93);
    display:flex; align-items:center; justify-content:center;
    backdrop-filter:blur(4px);
">
  <div style="
      background:#1E2028;
      border:1px solid #3A3D4E;
      border-radius:20px;
      padding:48px 52px;
      max-width:560px;
      width:90%;
      text-align:center;
      box-shadow:0 16px 64px rgba(0,0,0,0.7);
  ">
    <div style="animation:wfm-pulse 1.5s ease-in-out infinite; font-size:36px; margin-bottom:18px;">📡</div>
    <div style="color:#A884FF; font-size:21px; font-weight:700; font-family:system-ui; letter-spacing:-0.01em; margin-bottom:6px;">
      Forecast Generating
    </div>
    <div style="color:#5A4080; font-size:13px; margin-bottom:20px; font-family:system-ui;">
      Running Voice · Chat · Email independently
    </div>
    <div style="background:#2A2D3A; border-radius:4px; height:3px; width:100%; margin-bottom:32px; overflow:hidden;">
      <div style="background:linear-gradient(90deg,#7A4DDB,#A884FF); height:100%; border-radius:4px;
                  animation:wfm-bar 3s ease-out forwards;"></div>
    </div>
    <div style="
        border-top:1px solid #2E3245;
        padding-top:24px;
    ">
      <div style="color:#D6B85A; font-size:11px; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.12em; margin-bottom:10px;">
        💡 Did you know?
      </div>
      <div style="color:#DCCBFF; font-size:15px; font-weight:600; margin-bottom:10px; font-family:system-ui;">
        {fact_title}
      </div>
      <div style="color:#8A8EA8; font-size:13.5px; line-height:1.75; font-family:system-ui;">
        {fact_text}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        channel_results = {}
        for ch in CHANNELS:
            series = aggregate_channel(df, ch, units)
            channel_results[ch] = run_channel(series, model, horizon, units)
            save_forecast(ch, channel_results[ch]['model_label'],
                          units, horizon, channel_results[ch]['err'],
                          channel_results[ch]['params'],
                          channel_results[ch]['fdf'])

        _overlay.empty()
        st.session_state['ch_results']  = channel_results
        st.session_state['run_units']   = units
        st.session_state['run_model']   = model
        st.session_state['run_horizon'] = horizon

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if 'ch_results' in st.session_state:
        cr    = st.session_state['ch_results']
        r_units = st.session_state['run_units']
        r_model = st.session_state['run_model']

        # Channel checkboxes + historical window — above the chart
        ctrl_cols = st.columns([1, 1, 1, 2])
        show = {
            'Voice': ctrl_cols[0].checkbox('Voice', value=True),
            'Chat':  ctrl_cols[1].checkbox('Chat',  value=True),
            'Email': ctrl_cols[2].checkbox('Email', value=True),
        }
        display_limits = {'Daily': (7, 365), 'Weekly': (2, 52), 'Monthly': (2, 12)}
        d_min, d_max = display_limits[r_units]
        r_horizon = st.session_state.get('run_horizon', d_min)
        d_def = min(d_max, max(d_min, r_horizon))
        with ctrl_cols[3]:
            display_window = st.number_input(
                f'Historical window ({r_units.lower()})',
                min_value=d_min, max_value=d_max, value=d_def, step=1,
                help='Controls how much history is shown. Does not affect forecast calculations.'
            )

        # Filter to selected channels only
        visible = {ch: cr[ch] for ch in CHANNELS if show.get(ch) and cr.get(ch)}

        if visible:
            fig = make_multichannel_chart(visible, display_window, r_units,
                                          cr[list(visible.keys())[0]]['model_label'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Select at least one channel above to display the chart.')

        # Per-channel results tabs
        st.markdown('#### Forecast Values & Performance')
        tabs = st.tabs([ch for ch in CHANNELS if cr.get(ch)])
        for tab, ch in zip(tabs, [ch for ch in CHANNELS if cr.get(ch)]):
            r = cr[ch]
            with tab:
                left, right = st.columns([2, 1])
                with left:
                    def fmt_k(x):
                        return f"{x/1000:.1f}K" if x >= 1000 else str(int(x))

                    date_strs = pd.to_datetime(r['fdf']['date']).dt.strftime('%b %d, %Y').tolist()

                    if r['all_fc']:
                        # Auto-Select: winner first, then remaining models alphabetically
                        others = sorted(k for k in r['all_fc'] if k != r['winner'])
                        ordered = [r['winner']] + others
                        tbl = {'Date': date_strs}
                        for name in ordered:
                            vals = r['all_fc'][name]
                            label = f"★ {name}" if name == r['winner'] else name
                            tbl[label] = [fmt_k(v) for v in vals]
                        out = pd.DataFrame(tbl)
                    else:
                        out = pd.DataFrame({
                            'Date': date_strs,
                            'Forecasted Volume': [fmt_k(v) for v in r['fdf']['forecasted_offered']]
                        })
                    st.dataframe(out, use_container_width=True, hide_index=True, height=280)
                with right:
                    wmape_pct = r['err'] * 100
                    delta_str = 'Excellent' if wmape_pct < 8 else \
                                'Good'      if wmape_pct < 15 else \
                                'Moderate'  if wmape_pct < 25 else 'Review'
                    st.metric('WMAPE (holdout)', f"{wmape_pct:.1f}%", delta=delta_str)
                    if r['scores']:
                        st.markdown('**Competition Results**')
                        sc_df = pd.DataFrame([
                            {'Model': k, 'WMAPE': f"{v*100:.1f}%",
                             '': '✓' if k == r['winner'] else ''}
                            for k, v in sorted(r['scores'].items(), key=lambda x: x[1])
                        ])
                        st.dataframe(sc_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#5A34A3;font-size:18px;">'
            '⬅ Configure settings and click <strong style="color:#7A4DDB">Run Forecast</strong> to begin'
            '</div>',
            unsafe_allow_html=True
        )

    # ── SAVED RUNS ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('#### Recent Forecast Runs')
    saved = load_saved()
    if not saved.empty:
        disp = saved[['run_at','queue','model','units','horizon','wmape']].copy()
        disp['run_at'] = pd.to_datetime(disp['run_at']).dt.strftime('%b %d  %H:%M')
        disp['wmape']  = disp['wmape'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '—')
        disp.columns   = ['Run At', 'Queue', 'Model', 'Units', 'Horizon', 'WMAPE']
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.caption('No forecasts saved yet — run your first forecast above.')

if __name__ == '__main__':
    main()
