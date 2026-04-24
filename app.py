import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import sqlite3, json, warnings
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
def aggregate(df, queue, units):
    s = df[df['queue'] == queue].set_index('date')['offered'].sort_index()
    if units == 'Weekly':
        return s.resample('W').sum()
    if units == 'Monthly':
        return s.resample('MS').sum()
    return s  # Daily

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
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        name='Historical', mode='lines',
        line=dict(color=C['purple'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast.index) + list(forecast.index[::-1]),
        y=list((forecast + ci).values) + list((forecast - ci).values[::-1]),
        fill='toself', fillcolor='rgba(122,77,219,0.18)',
        line=dict(color='rgba(0,0,0,0)'), name='±10% band'
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
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

# ─── APP ─────────────────────────────────────────────────────────────────────
def main():
    df   = generate_data()
    init_db()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Forecast Settings")
        st.divider()

        channel_opts = ['All Channels'] + sorted(df['channel'].unique())
        ch = st.selectbox('Channel', channel_opts)

        q_list = sorted(df['queue'].unique()) if ch == 'All Channels' \
                 else sorted(df[df['channel'] == ch]['queue'].unique())
        queue = st.selectbox('Queue', q_list)

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

        hlimits  = {'Daily': 90, 'Weekly': 26, 'Monthly': 12}
        hdefault = {'Daily': 14, 'Weekly': 8,  'Monthly': 3}
        horizon  = st.slider(
            f'Horizon ({units.lower()})',
            1, hlimits[units], hdefault[units]
        )

        st.divider()
        run_btn = st.button('▶  Run Forecast', use_container_width=True)
        st.divider()

        q_data = df[df['queue'] == queue]
        st.markdown(f"**Data range**")
        st.caption(f"{df['date'].min().strftime('%b %d, %Y')} → {df['date'].max().strftime('%b %d, %Y')}")
        st.caption(f"**{len(q_data):,}** records for this queue")

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown("# 📊 WFM Forecasting Suite")
    st.caption(f"Queue: **{queue}** · Model: **{model}** · Horizon: **{horizon} {units.lower()}**")

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

    if run_btn:
        series  = aggregate(df, queue, units)
        display = series.iloc[-90:] if units == 'Daily' else \
                  series.iloc[-52:] if units == 'Weekly' else series

        with st.spinner('Running forecast models...'):
            if model == 'Auto-Select Model':
                fc, params, winner, scores = m_auto_select(series, horizon, units)
                model_label = f'Auto-Select → {winner}'

                holdout = min(max(4, int(len(series) * 0.15)), 10)
                train   = series.iloc[:-holdout]
                actual  = series.iloc[-holdout:].values
                pred_h, _ = MODEL_MAP[winner](train, holdout, units)
                err = wmape(actual, pred_h.values[:len(actual)])

            else:
                fn = MODEL_MAP[model]
                fc, params = fn(series, horizon, units)
                model_label = model
                scores = None
                winner = model

                holdout = min(8, len(series) // 5)
                train   = series.iloc[:-holdout]
                actual  = series.iloc[-holdout:].values
                pred_h, _ = fn(train, holdout, units)
                err = wmape(actual, pred_h.values[:len(actual)])

            fdf = pd.DataFrame({
                'date':              fc.index,
                'forecasted_offered': fc.values.round(0).astype(int)
            })
            save_forecast(queue, model_label, units, horizon, err, params, fdf)

            st.session_state['result'] = dict(
                display=display, fc=fc, queue=queue,
                model_label=model_label, winner=winner,
                units=units, err=err, params=params,
                scores=scores, fdf=fdf
            )

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if 'result' in st.session_state:
        r = st.session_state['result']

        fig = make_chart(r['display'], r['fc'], r['queue'], r['model_label'], r['units'])
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([2, 1])

        with left:
            st.markdown('#### Forecast Values')
            out = r['fdf'].copy()
            out['date'] = pd.to_datetime(out['date']).dt.strftime('%b %d, %Y')
            out.columns = ['Date', 'Forecasted Volume']
            st.dataframe(out, use_container_width=True, hide_index=True, height=320)

        with right:
            st.markdown('#### Model Performance')
            wmape_pct = r['err'] * 100
            delta_str = 'Excellent' if wmape_pct < 8 else \
                        'Good'      if wmape_pct < 15 else \
                        'Moderate'  if wmape_pct < 25 else 'Review'
            st.metric('WMAPE (holdout)', f"{wmape_pct:.1f}%", delta=delta_str)

            if r['scores']:
                st.markdown('**Competition Results**')
                sc_df = pd.DataFrame([
                    {'Model': k,
                     'WMAPE': f"{v*100:.1f}%",
                     '': '✓ Selected' if k == r['winner'] else ''}
                    for k, v in sorted(r['scores'].items(), key=lambda x: x[1])
                ])
                st.dataframe(sc_df, use_container_width=True, hide_index=True)

            if r['params']:
                filtered = {k: v for k, v in r['params'].items()
                            if k != 'competition_scores'}
                if filtered:
                    st.markdown('**Parameters**')
                    for k, v in filtered.items():
                        st.caption(f"`{k}`: {v}")
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
