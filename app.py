"""
CrowdAI — Dark Control Room Dashboard  (v4 — stable)
======================================================
Architecture: NO negative-margin CSS hacks.
  - Navbar is a pure HTML/CSS fixed bar (visual only — logo + stats + live pill)
  - Controls (scenario radio, speed, pause, AI, reset) are in their own
    st.columns row BELOW the navbar, styled to look integrated
  - ☰ button in the left panel opens Streamlit native sidebar (zone selector)
  - Left mini-panel: risk counts + zone cards + Map/Charts toggle
  - Main area: heatmap view OR charts view for SELECTED zone only
"""

import math, time, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# ── src/ imports (identical to original) ──────────────────────────────────────
from src.simulate_data import (
    generate_normal_day,
    generate_post_event_rush,
    generate_emergency_evacuation,
)
from src.features import get_realtime_features, get_feature_columns, ROLLING_WINDOW
from src.predictor import (
    predict_zone, PredictionResult,
    YELLOW_THRESHOLD, RED_THRESHOLD,
    set_use_bedrock, get_use_bedrock,
)
from src.model import load_model
from src.aws_bedrock import (
    is_bedrock_available, generate_incident_summary, generate_crowd_recommendation,
)
from src.aws_storage import get_aws_status, store_incident

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdAI — Control Room",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_KEYS = ["🏢 Normal Day", "🎉 Post-Event Rush", "🚨 Emergency Evacuation"]
SCENARIOS = {
    "🏢 Normal Day":           generate_normal_day,
    "🎉 Post-Event Rush":      generate_post_event_rush,
    "🚨 Emergency Evacuation": generate_emergency_evacuation,
}
ZONE_COLORS = {"Zone_A": "#22D3EE", "Zone_B": "#818CF8", "Zone_C": "#34D399"}
ZONE_NAMES  = {"Zone_A": "North Entrance", "Zone_B": "Main Concourse", "Zone_C": "South Exit"}
ZONE_CAPS   = {"Zone_A": 200, "Zone_B": 500, "Zone_C": 180}

def _rc(lv):  return {"green":"#22C55E","yellow":"#F59E0B","red":"#EF4444"}.get(lv,"#22C55E")
def _rbg(lv): return {"green":"#052e16","yellow":"#1c1505","red":"#2d0808"}.get(lv,"#052e16")
def _rl(lv):  return {"green":"NOMINAL","yellow":"ELEVATED","red":"CRITICAL"}.get(lv,"NOMINAL")

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — clean, no layout hacks
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;800&display=swap');

/* ── Base ── */
html,body,[class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stApp"],.main,section.main {
    background:#060d18 !important;
    font-family:'Inter',system-ui,sans-serif !important;
    color:#c8d8f0 !important;
}
/* Make the app take full width and remove the floating paper effect */
.main .block-container, 
div[data-testid="stAppViewBlockContainer"],
.stAppViewBlockContainer, .block-container {
    padding-top: 85px !important;  
    padding-bottom: 0px !important;
    padding-left: 0px !important;
    padding-right: 0px !important;
    max-width: 100% !important;
    width: 100% !important;
}

#MainMenu, footer { display: none !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    z-index: 99999999 !important;
    pointer-events: none !important;
    left: 0 !important;
}
header[data-testid="stHeader"] * {
    pointer-events: auto !important;
}


/* Hide the 'deploy' / toolbar stuff on the right but keep the sidebar toggle */
header[data-testid="stHeader"] .stAppDeployButton,
[data-testid="stAppDeployButton"],
[data-testid="stLogo"] {
    display: none !important;
}

/* Hide native Streamlit sidebar toggle buttons completely without breaking React click listeners */
[data-testid="collapsedControl"], 
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    opacity: 0 !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#060d18}
::-webkit-scrollbar-thumb{background:#0f1e30;border-radius:2px}

/* ── Streamlit buttons ── */
.stButton > button {
    background:#07101c !important; color:#607898 !important;
    border:1px solid #0d1826 !important; border-radius:6px !important;
    font-size:14px !important; font-weight:600 !important;
    font-family:'JetBrains Mono',monospace !important;
    padding:6px 14px !important; transition:all .15s !important;
    white-space: nowrap !important;
}
.stButton > button:hover { border-color:#22D3EE !important; color:#22D3EE !important; }

/* ── Progress bar ── */
div[data-testid="stProgress"] > div {
    background:#07101c !important; border-radius:2px !important; height:3px !important;
}
div[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,#22D3EE,#818CF8) !important;
}
div[data-testid="stProgress"] p {
    color:#1e2d45 !important; font-size:10px !important;
    font-family:'JetBrains Mono',monospace !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] label { color:#3d5070 !important; font-size:13px !important; }
div[data-testid="stSlider"] { padding:0 !important; }

/* ── Toggle ── */
div[data-testid="stToggle"] label { color:#3d5070 !important; font-size:13px !important; }

/* ── Radio (scenario pills) ── */
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] > div {
    display:flex !important; flex-direction:row !important; gap:6px !important;
    background:#07101c !important; border:1px solid #1a3050 !important;
    border-radius:10px !important; padding:6px !important;
}
div[data-testid="stRadio"] label {
    background:#0b1828 !important; border-radius:6px !important;
    padding:8px 18px !important; font-size:14px !important; font-weight: 500 !important;
    color:#ffffff !important; cursor:pointer !important;
    transition:all .2s ease !important; font-family:'Inter',sans-serif !important;
    border: 1px solid #1e3a5f !important;
}
div[data-testid="stRadio"] label p {
    color:#ffffff !important;
}
div[data-testid="stRadio"] label:hover {
    background:#13253f !important;
    color:#ffffff !important;
    border-color: #2d528b !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(135deg, rgba(34,211,238,0.15), rgba(34,211,238,0.05)) !important;
    color:#22D3EE !important;
    font-weight:700 !important;
    border: 1px solid #22D3EE !important;
    box-shadow: 0 0 12px rgba(34,211,238,0.2) !important;
}
div[data-testid="stRadio"] label p {
    margin: 0 !important; padding: 0 !important;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background:#07101c !important;
    border-right:1px solid #0a1622 !important;
    min-width: 260px !important;
    max-width: 300px !important;
    top: 80px !important;
    height: calc(100vh - 80px) !important;
    z-index: 999998 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 12px 10px !important;
}
section[data-testid="stSidebar"] h3 {
    color:#ffffff !important; font-size:9px !important;
    text-transform:uppercase; letter-spacing:1px;
    font-family:'JetBrains Mono',monospace !important;
    margin-bottom:8px !important;
}

.ml-footer {
    font-size: 10px; color: #ffffff !important; font-family: 'JetBrains Mono', monospace; line-height: 1.4;
}

/* ── Animations ── */
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.dot-live{display:inline-block;width:7px;height:7px;border-radius:50%;
          background:#22C55E;animation:blink 1.8s infinite;}
.dot-pause{display:inline-block;width:7px;height:7px;border-radius:50%;
           background:#3d5070;}

/* ── Fixed navbar ── */
.cr-nav {
    position: fixed; top: 0; left: 0; right: 0; height: 80px;
    background: rgba(7,16,28,0.85); backdrop-filter: blur(12px);
    border-bottom: 1px solid #1a3050; z-index: 9999999 !important;
    padding: 0 20px 0 80px; display: flex; align-items: center; justify-content: flex-start;
}
.cr-logo {
    display: flex; align-items: center; gap: 12px;
    font-size: 24px; font-weight: 800; letter-spacing: 1px;
    margin-right: 30px;
}
.cr-logo-icon {
    width:32px; height:32px; border-radius:8px;
    background:linear-gradient(135deg,#6366F1,#22D3EE);
    display:flex; align-items:center; justify-content:center; font-size:16px;
}
.nav-stat{font-size:10px;color:#1a2a40;font-family:'JetBrains Mono',monospace;}
.nav-stat b{font-weight:800;}
.live-pill{
    display:flex; align-items:center; gap:6px;
    background:#050b14; border:1px solid #0a1622;
    border-radius:24px; padding:6px 14px;
    font-size:13px; font-weight:700; color:#3d5070;
    font-family:'JetBrains Mono',monospace;
}

/* ── Controls bar (below navbar) ── */
.cr-controls {
    background:#060d18; border-bottom:1px solid #0a1420;
    padding:6px 16px;
    display:flex; align-items:center; gap:8px;
}

/* ── Zone cards ── */
.zcard { border-radius:8px; padding:9px 11px; margin-bottom:5px; transition:all .15s; }
.zcard.sel { background:#0e1829; border:1px solid #1a3050; }
.zcard.nor { background:#06101a; border:1px solid #0a1420; }

/* ── Heatmap card ── */
.hmap-card{background:#06101a;border:1px solid #0a1420;border-radius:10px;overflow:hidden;}
.hmap-hdr{padding:8px 12px;border-bottom:1px solid #0a1420;
          display:flex;align-items:center;justify-content:space-between;}

/* ── Chart card ── */
.chart-card{background:#07101c;border:1px solid #0a1420;border-radius:10px;
            padding:10px 10px 4px 10px;margin-bottom:0;}
.chart-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;}
.chart-lbl{font-size:9px;font-weight:700;color:#3d5070;text-transform:uppercase;
           letter-spacing:.8px;font-family:'JetBrains Mono',monospace;}
.chart-val{font-size:13px;font-weight:800;font-family:'JetBrains Mono',monospace;
           font-variant-numeric:tabular-nums;}
.chart-unit{font-size:7.5px;color:#1e2d45;margin-left:2px;}
.chart-n{font-size:7px;color:#182535;text-align:right;margin-top:1px;
         font-family:'JetBrains Mono',monospace;}

/* ── Incident card ── */
.incident-card{background:#0d0a1e;border:1px solid #1e1560;
               border-left:4px solid #6366F1;border-radius:10px;
               padding:10px 14px;margin-top:8px;}
.incident-lbl{font-size:8px;font-weight:700;color:#6366F1;letter-spacing:.8px;
              margin-bottom:4px;font-family:'JetBrains Mono',monospace;}
.incident-body{font-size:11px;color:#818CF8;line-height:1.6;}

/* ── Signage card ── */
.sig-card{border-radius:8px;padding:9px 11px;border-left:3px solid;margin-bottom:6px;}
.sig-lbl{font-size:8px;font-weight:700;letter-spacing:.8px;color:#3d5070;
         margin-bottom:4px;font-family:'JetBrains Mono',monospace;}
.sig-txt{font-size:11px;color:#c8d8f0;line-height:1.5;}

/* ── ML footer ── */
.ml-footer{background:#06101a;border:1px solid #0a1420;border-radius:7px;
           padding:9px 10px;margin-top:6px;
           font-size:7.5px;color:#182535;
           font-family:'JetBrains Mono',monospace;line-height:1.8;}
.ml-footer b{color:#253848;}

/* ── Row label ── */
.row-lbl{font-size:8px;color:#1a2a40;text-transform:uppercase;letter-spacing:1px;
         margin:6px 0 4px 0;font-family:'JetBrains Mono',monospace;}

/* ── Scroll hint ── */
.scroll-hint{text-align:center;font-size:9px;color:#1a2a40;padding:6px 0;
             font-family:'JetBrains Mono',monospace;animation:blink 2.5s infinite;}

/* ── Status bar ── */
.cr-statusbar{height:18px;background:#04080f;border-top:1px solid #0a1420;
              display:flex;align-items:center;padding:0 12px;gap:14px;
              font-size:7px;color:#182535;font-family:'JetBrains Mono',monospace;}

/* ── Zone tab buttons — active style ── */
.ztab-active > button {
    border-color:#22D3EE !important; color:#22D3EE !important;
    background:#07101c !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VENUE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
_ROOMS = [
    (12,12,148,72,"Registration"),(12,96,68,88,"VIP Lounge"),
    (92,96,68,88,"Atrium"),(12,196,148,76,"Workshop A"),
    (12,284,148,80,"Poster Hall"),(198,12,134,108,"Main Stage"),
    (198,132,60,64,"Booth A"),(270,132,62,64,"Booth B"),
    (198,208,134,76,"Activities"),(198,296,134,68,"STEM Lab"),
    (370,12,100,72,"F&B Court"),(370,96,100,72,"Networking"),
    (370,180,100,80,"Conf Rm 1"),(370,272,100,56,"Conf Rm 2"),
    (370,340,100,56,"Breakout"),(488,12,92,104,"SOW Theatre"),
    (488,128,92,68,"Meeting Rms"),(488,208,92,68,"Sponsor Hub"),
    (488,288,92,56,"Staff Only"),(592,12,96,130,"Tech Pavilion"),
    (592,154,96,110,"Demo Zone"),(592,276,96,120,"Startup Alley"),
]
_CORRIDORS = [
    (160,0,28,420),(332,0,28,420),(480,0,28,420),
    (0,84,700,24),(0,196,700,24),(0,284,700,24),
]
_ZONE_BOUNDS = [
    ("Zone_A",0,0,700,108,"#22D3EE","ZONE A · NORTH ENTRANCE"),
    ("Zone_B",0,108,700,112,"#818CF8","ZONE B · MAIN CONCOURSE"),
    ("Zone_C",0,220,700,200,"#34D399","ZONE C · SOUTH EXIT"),
]
_HOTSPOTS = {
    "Zone_A":[(174,20,52,.9),(86,48,40,.55),(360,48,38,.45),(534,55,35,.4),(174,84,44,.65)],
    "Zone_B":[(174,150,60,.85),(265,165,55,.75),(396,140,48,.6),
              (510,170,42,.5),(640,200,40,.45),(174,218,50,.7)],
    "Zone_C":[(174,300,52,.8),(265,265,45,.6),(86,340,40,.55),
              (420,310,38,.45),(534,350,36,.4)],
}

def _heat_rgba(v):
    if v<0.02: return (0,0,0,0)
    t=max(0.,min(1.,v))
    if   t<.20: s=t/.20;       r=0;          g=int(s*180);        b=255
    elif t<.40: s=(t-.20)/.20; r=0;          g=180+int(s*75);     b=int(255*(1-s))
    elif t<.60: s=(t-.40)/.20; r=int(s*255); g=255;               b=0
    elif t<.80: s=(t-.60)/.20; r=255;        g=int(255*(1-s*.6)); b=0
    else:       s=(t-.80)/.20; r=255;        g=int(100*(1-s));    b=0
    return (r,g,b,min(.88,t*1.8))

def _build_heat_png(zd,VW=700,VH=420,CR=140,RR=84):
    buf=np.zeros(CR*RR,dtype=np.float32)
    for zid,spots in _HOTSPOTS.items():
        d=float(zd.get(zid,1.5)); w=float(np.clip(d/8.,.05,1.3))
        for cx,cy,sig,base in spots:
            cx_n=cx/VW*CR; cy_n=cy/VH*RR
            sig_n=(sig/VW*CR)*(0.7+w*0.6); amp=base*w
            for r in range(RR):
                for c in range(CR):
                    dx,dy=c-cx_n,r-cy_n
                    buf[r*CR+c]+=amp*math.exp(-(dx*dx+dy*dy)/(2*sig_n*sig_n))
    mx=buf.max()
    if mx>0: buf/=mx
    img=np.zeros((RR,CR,4),dtype=np.uint8)
    for i in range(RR*CR):
        ri,ci=divmod(i,CR)
        rv,gv,bv,av=_heat_rgba(float(buf[i]))
        img[ri,ci]=[rv,gv,bv,int(av*255)]
    try:
        from PIL import Image as PI, ImageFilter
        pil=PI.fromarray(img,mode="RGBA").filter(ImageFilter.GaussianBlur(radius=4))
    except Exception:
        from PIL import Image as PI
        pil=PI.fromarray(img,mode="RGBA")
    bio=BytesIO(); pil.save(bio,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(bio.getvalue()).decode()

def make_venue_fig(zd, sel, h=450):
    VW,VH=700,420
    fig=go.Figure()
    fig.add_layout_image(dict(source=_build_heat_png(zd,VW,VH),
        xref="x",yref="y",x=0,y=VH,sizex=VW,sizey=VH,
        sizing="stretch",opacity=0.80,layer="above"))
    shapes=[]
    shapes.append(dict(type="rect",x0=0,y0=0,x1=VW,y1=VH,
                       fillcolor="#07111e",line=dict(width=0),layer="below"))
    for cx,cy,cw,ch in _CORRIDORS:
        shapes.append(dict(type="rect",x0=cx,y0=VH-cy-ch,x1=cx+cw,y1=VH-cy,
                           fillcolor="#091525",line=dict(width=0),layer="below"))
    for rx,ry,rw,rh,_ in _ROOMS:
        shapes.append(dict(type="rect",x0=rx,y0=VH-ry-rh,x1=rx+rw,y1=VH-ry,
                           fillcolor="#0c1d30",line=dict(color="#152d4a",width=0.8),layer="below"))
    for lx in [160,332,480]:
        shapes.append(dict(type="line",x0=lx,y0=0,x1=lx,y1=VH,line=dict(color="#152d4a",width=1.2)))
    for ly in [84,196,284]:
        shapes.append(dict(type="line",x0=0,y0=VH-ly,x1=VW,y1=VH-ly,line=dict(color="#152d4a",width=1.2)))
    shapes.append(dict(type="rect",x0=0,y0=0,x1=VW,y1=VH,fillcolor="rgba(0,0,0,0)",
                       line=dict(color="#1a3558",width=2.5)))
    for zid,zx,zy,zw,zh,zcol,_ in _ZONE_BOUNDS:
        isel=(zid==sel)
        shapes.append(dict(type="rect",x0=zx+1,y0=VH-zy-zh+1,x1=zx+zw-1,y1=VH-zy-1,
                           fillcolor="rgba(0,0,0,0)",
                           line=dict(color=zcol,width=2.0 if isel else 0.8,
                                     dash="solid" if isel else "dot"),
                           opacity=0.85 if isel else 0.22))
    fig.update_layout(shapes=shapes)
    anns=[]
    for rx,ry,rw,rh,lbl in _ROOMS:
        fs=min(9.,rw/max(len(lbl),1)*1.6)
        anns.append(dict(x=rx+rw/2,y=VH-ry-rh/2,text=lbl,showarrow=False,
                         font=dict(size=max(6,fs),color="#162a42",family="JetBrains Mono"),
                         xanchor="center",yanchor="middle"))
    for zid,zx,zy,zw,zh,zcol,zlbl in _ZONE_BOUNDS:
        isel=(zid==sel)
        anns.append(dict(x=zx+6,y=VH-zy-8,text=zlbl,showarrow=False,
                         font=dict(size=7,color=zcol,family="JetBrains Mono"),
                         xanchor="left",yanchor="top",opacity=0.85 if isel else 0.28))
    for zid,zx,zy,zw,zh,zcol,zlbl in _ZONE_BOUNDS:
        d=float(zd.get(zid,0.))
        col="#EF4444" if d>4 else "#F97316" if d>2.5 else "#22C55E"
        anns.append(dict(x=zx+zw-8,y=VH-zy-8,
                         text=f"<b>{d:.1f}</b> <span style='font-size:7px;'>p/m²</span>",showarrow=False,
                         font=dict(size=10,color=col,family="JetBrains Mono"),
                         xanchor="right",yanchor="top",
                         bgcolor="rgba(7,17,30,0.85)",bordercolor=col,borderwidth=1,borderpad=4))
    fig.update_layout(annotations=anns)
    for ex,ey,sym,lbl,pos in [(174,VH,"triangle-up","N","top center"),
                               (174,0,"triangle-down","S","bottom center"),
                               (VW,VH/2,"triangle-right","E","middle right"),
                               (0,VH/2,"triangle-left","W","middle left")]:
        fig.add_trace(go.Scatter(x=[ex],y=[ey],mode="markers+text",
            marker=dict(symbol=sym,size=11,color="#07111e",line=dict(color="#22C55E",width=2)),
            text=[lbl],textfont=dict(size=7,color="rgba(34, 197, 94, 0.53)",family="JetBrains Mono"),
            textposition=pos,hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=[VW-22],y=[22],mode="markers+text",
        marker=dict(symbol="circle",size=18,color="#07111e",line=dict(color="#1a3558",width=1)),
        text=["N"],textfont=dict(size=8,color="#22D3EE",family="JetBrains Mono"),
        textposition="middle center",hoverinfo="skip",showlegend=False))
    fig.update_layout(
        xaxis=dict(range=[0,VW],visible=False,fixedrange=True),
        yaxis=dict(range=[0,VH],visible=False,fixedrange=True),
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#07111e",plot_bgcolor="#07111e",height=h,showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  GROWING CHART
# ══════════════════════════════════════════════════════════════════════════════
def growing_chart(df,x_col,y_col,label,y_label,color,y_range,y_ticks,
                  ref_val=None,ref_label="",h=185):
    fig=go.Figure(); n=len(df)
    if n>1:
        x=df[x_col]; y=df[y_col].fillna(0)
        rh,gh,bh=int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig.add_trace(go.Scatter(x=x,y=y,fill="tozeroy",
            fillcolor=f"rgba({rh},{gh},{bh},0.10)",line=dict(width=0),mode="none",
            hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=color,width=1.8),
            showlegend=False,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.3f}} {y_label}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[x.iloc[-1]],y=[y.iloc[-1]],mode="markers",
            marker=dict(size=6,color=color,line=dict(color="#060d18",width=2)),
            hoverinfo="skip",showlegend=False))
    if ref_val is not None:
        fig.add_hline(y=ref_val,line=dict(color="#EF4444",width=1,dash="dot"),opacity=.35,
                      annotation_text=ref_label,
                      annotation_font=dict(size=7,color="rgba(239,68,68,.5)",family="JetBrains Mono"),
                      annotation_position="top right")
    x_range=([df[x_col].iloc[0],df[x_col].iloc[-1]] if n>1 else None)
    fig.update_layout(height=h,margin=dict(l=40,r=8,t=2,b=24),
        paper_bgcolor="#07101c",plot_bgcolor="#060d18",
        font=dict(family="JetBrains Mono",color="#2d4060"),
        xaxis=dict(title=dict(text="",font=dict(size=7)),
                   tickfont=dict(size=7),gridcolor="#0c1a2c",linecolor="#0d1e30",
                   tickcolor="#0d1e30",range=x_range,showgrid=True),
        yaxis=dict(title=dict(text=y_label,font=dict(size=7),standoff=2),
                   tickfont=dict(size=7),tickvals=y_ticks,range=y_range,
                   gridcolor="#0c1a2c",linecolor="#0d1e30",tickcolor="#0d1e30",showgrid=True),
        hovermode="x unified")
    return fig

def overlay_chart(zone_dfs,y_col,y_label,y_range,y_ticks,ref_val=None,h=185):
    """Multi-zone overlay chart."""
    fig=go.Figure()
    for zone,df in zone_dfs.items():
        if len(df)<2 or y_col not in df.columns: continue
        zc=ZONE_COLORS.get(zone,"#22D3EE")
        rh,gh,bh=int(zc[1:3],16),int(zc[3:5],16),int(zc[5:7],16)
        fig.add_trace(go.Scatter(x=df["step_label"],y=df[y_col].fillna(0),
            fill="tozeroy",fillcolor=f"rgba({rh},{gh},{bh},0.07)",
            line=dict(width=0),mode="none",hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=df["step_label"],y=df[y_col].fillna(0),
            mode="lines",name=f"Z{zone[-1]}",line=dict(color=zc,width=1.8),
            hovertemplate=f"Z{zone[-1]}: %{{y:.2f}} {y_label}<extra></extra>"))
    if ref_val is not None:
        fig.add_hline(y=ref_val,line=dict(color="#EF4444",width=1,dash="dot"),opacity=.35,
                      annotation_text=str(ref_val),
                      annotation_font=dict(size=7,color="rgba(239,68,68,.5)",family="JetBrains Mono"),
                      annotation_position="top right")
    fig.update_layout(height=h,margin=dict(l=40,r=8,t=2,b=24),
        paper_bgcolor="#07101c",plot_bgcolor="#060d18",
        font=dict(family="JetBrains Mono",color="#2d4060"),
        legend=dict(orientation="h",yanchor="top",y=-0.20,xanchor="center",x=0.5,
                    font=dict(size=7),bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickfont=dict(size=7),gridcolor="#0c1a2c",linecolor="#0d1e30",showgrid=True),
        yaxis=dict(title=dict(text=y_label,font=dict(size=7),standoff=2),
                   tickfont=dict(size=7),tickvals=y_ticks,range=y_range,
                   gridcolor="#0c1a2c",linecolor="#0d1e30",showgrid=True),
        hovermode="x unified")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _load_model():
    return load_model()

def _dot(color, anim=False):
    a="animation:blink .9s infinite;" if anim else ""
    return f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{color};{a}"></span>'

def _badge(text,fg,bg,bdr):
    return (f'<span style="background:{bg};border:1px solid {bdr};color:{fg};'
            f'font-size:8px;font-weight:700;border-radius:3px;padding:2px 6px;'
            f'letter-spacing:.8px;font-family:JetBrains Mono,monospace;">{text}</span>')

def _wait(h=185):
    return (f'<div style="height:{h}px;display:flex;align-items:center;'
            f'justify-content:center;color:#1a2a40;font-size:10px;'
            f'font-family:JetBrains Mono,monospace;background:#07101c;'
            f'border:1px solid #0a1420;border-radius:10px;">Collecting data…</div>')

def chart_card(col, df, y_col, label, y_label, color, y_range, y_ticks,
               ref_val=None, ref_label="", key="", n_samples=0):
    with col:
        cur = "—"
        if len(df)>0 and y_col in df.columns:
            v = float(df[y_col].iloc[-1])
            cur = f"{v:.2f}"
        thr = (f'<span style="font-size:7.5px;color:rgba(239,68,68,.35);'
               f'font-family:JetBrains Mono,monospace;"> {ref_val}</span>'
               if ref_val is not None else "")
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-hdr">
            <div style="display:flex;align-items:center;gap:5px;">
              <div style="width:8px;height:2px;background:{color};border-radius:1px;
                          box-shadow:0 0 4px {color}88;"></div>
              <span class="chart-lbl">{label}</span>{thr}
            </div>
            <div><span class="chart-val" style="color:{color};">{cur}</span>
                 <span class="chart-unit">{y_label}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)
        if len(df)>1 and y_col in df.columns:
            fig = growing_chart(df,"step_label",y_col,label,y_label,color,
                                y_range,y_ticks,ref_val,ref_label)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar":False,"scrollZoom":True}, key=key)
        else:
            st.markdown(_wait(), unsafe_allow_html=True)
        st.markdown(f'<div class="chart-n">{n_samples} pts</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Defaults ──────────────────────────────────────────────────────────────
    for k,v in dict(scenario=SCENARIO_KEYS[0], selected_zone="Zone_A",
                    running=True, step=ROLLING_WINDOW,
                    update_speed=2.0, view="heatmap").items():
        if k not in st.session_state:
            st.session_state[k] = v

    scenario = st.session_state.scenario
    running  = st.session_state.running
    step     = st.session_state.step
    selected = st.session_state.selected_zone

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        model, scaler = _load_model()
    except Exception:
        st.error("⚠️ Model not found — run:  python -m src.model")
        st.stop()

    # ── Data ──────────────────────────────────────────────────────────────────
    np.random.seed(42)
    full_data = SCENARIOS[scenario]()
    zones = sorted(full_data["zone_id"].unique())
    zone_data = {z: full_data[full_data["zone_id"]==z].reset_index(drop=True) for z in zones}
    n_points = len(zone_data[zones[0]])
    if step >= n_points:
        step = st.session_state.step = ROLLING_WINDOW

    # ── Predictions ───────────────────────────────────────────────────────────
    predictions: dict[str, PredictionResult] = {}
    zone_histories: dict[str, pd.DataFrame] = {}
    for zone in zones:
        hist = zone_data[zone].iloc[max(0,step-50):step+1].copy()
        zone_histories[zone] = hist
        feats = get_realtime_features(hist)
        if feats:
            predictions[zone] = predict_zone(zone, feats, model, scaler)

    # ── Accumulate history ─────────────────────────────────────────────────────
    hk = f"hist_{scenario}"
    if hk not in st.session_state:
        st.session_state[hk] = {z: [] for z in zones}
    for zone in zones:
        hist = zone_histories.get(zone, pd.DataFrame())
        if len(hist) > 0:
            row = hist.iloc[-1].to_dict()
            row["step"] = step
            row["step_label"] = f"{step//60:02d}:{step%60:02d}"
            st.session_state[hk][zone].append(row)

    def _hdf(zone):
        rows = st.session_state[hk].get(zone, [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Aggregate stats ────────────────────────────────────────────────────────
    crit_cnt  = sum(1 for p in predictions.values() if p.risk_level=="red")
    any_crit  = crit_cnt > 0
    n_samples = len(st.session_state[hk].get(zones[0], []))
    aws       = get_aws_status()

    zd = {z: float(zone_histories[z].iloc[-1]["density"])
          if z in zone_histories and len(zone_histories[z])>0 else 1.5
          for z in zones}
    zv = {z: float(zone_histories[z].iloc[-1]["velocity"])
          if z in zone_histories and len(zone_histories[z])>0 else 1.0
          for z in zones}

    import streamlit.components.v1 as components
    components.html("""
        <script>
        const doc = window.parent.document;
        // Float the controls directly OVER the native header without detaching from React tree to preserve interactions
        setInterval(() => {
            // Find our custom target in the navbar
            const navTarget = doc.querySelector('.nav-controls-target');
            // Find the container holding the radio buttons
            const radios = doc.querySelectorAll('div[data-testid="stRadio"]');
            
            if (navTarget && radios.length > 0) {
                let ctrlRow = radios[0].closest('div[data-testid="stHorizontalBlock"]');
                if (ctrlRow) {
                    navTarget.style.width = '100%';
                    navTarget.style.flex = '1';
                    
                    const rect = navTarget.getBoundingClientRect();
                    const rowRect = ctrlRow.getBoundingClientRect();
                    
                    if (rect.width > 0) {
                        ctrlRow.style.setProperty('position', 'fixed', 'important');
                        
                        // Stretch explicitly over the top navigation area
                        ctrlRow.style.setProperty('left', '0', 'important');
                        ctrlRow.style.setProperty('right', '0', 'important');
                        ctrlRow.style.setProperty('top', '0', 'important');
                        // Remove bottom to prevent stretching vertically across entire screen
                        ctrlRow.style.setProperty('width', 'auto', 'important');
                        ctrlRow.style.setProperty('height', '80px', 'important'); 
                        ctrlRow.style.setProperty('z-index', '99999999', 'important');
                        ctrlRow.style.setProperty('margin', '0', 'important');
                        ctrlRow.style.setProperty('padding', '0', 'important');
                        ctrlRow.style.setProperty('display', 'flex', 'important');
                        ctrlRow.style.setProperty('align-items', 'center', 'important');
                        ctrlRow.style.setProperty('justify-content', 'center', 'important');
                        ctrlRow.style.setProperty('gap', '10px', 'important');
                        ctrlRow.style.setProperty('transform', 'none', 'important');
                        ctrlRow.style.setProperty('pointer-events', 'none', 'important');
                        
                        // Force explicit uncrushable sizing on the immediate column wrappers
                        const flexColumns = Array.from(ctrlRow.children);
                        
                        if (flexColumns.length >= 1) {
                            flexColumns[0].style.setProperty('width', 'max-content', 'important');
                            flexColumns[0].style.setProperty('flex', '0 0 max-content', 'important');
                            flexColumns[0].style.setProperty('pointer-events', 'auto', 'important');
                        }
                        if (flexColumns.length >= 2) {
                            // Container for the play/pause button
                            flexColumns[1].style.setProperty('width', '60px', 'important');
                            flexColumns[1].style.setProperty('flex', '0 0 60px', 'important');
                            flexColumns[1].style.setProperty('pointer-events', 'auto', 'important');
                            
                            // Hide any extraneous columns if they still exist
                            for(let i=2; i<flexColumns.length; i++) {
                                flexColumns[i].style.setProperty('display', 'none', 'important');
                            }
                        }
                        
                        flexColumns.forEach(c => {
                            c.style.padding = '0 5px';
                            c.style.display = 'flex';
                            c.style.alignItems = 'center';
                        });

                        // Force the radio buttons to be strictly horizontal
                        const radioGroup = ctrlRow.querySelector('div[role="radiogroup"]');
                        if (radioGroup) {
                            radioGroup.style.setProperty('display', 'flex', 'important');
                            radioGroup.style.setProperty('flex-direction', 'row', 'important');
                            radioGroup.style.setProperty('gap', '15px', 'important');
                            radioGroup.style.setProperty('flex-wrap', 'nowrap', 'important');
                            // Remove the internal block wrappers Streamlit puts securely around each option
                            Array.from(radioGroup.children).forEach(child => {
                                child.style.setProperty('display', 'inline-flex', 'important');
                                child.style.setProperty('align-items', 'center', 'important');
                                child.style.setProperty('margin-bottom', '0', 'important');
                                child.style.setProperty('white-space', 'nowrap', 'important');
                                child.style.setProperty('min-width', 'max-content', 'important');
                                const p = child.querySelector('p');
                                if (p) p.style.setProperty('white-space', 'nowrap', 'important');
                            });
                        }
                    }
                }
            }
            
            // Wire the custom sidebar toggle exactly once
            const customToggle = doc.getElementById('custom-sidebar-toggle');
            if (customToggle && !customToggle.dataset.hasListener) {
                customToggle.dataset.hasListener = 'true';
                customToggle.addEventListener('click', function() {
                    const targets = doc.querySelectorAll('[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]');
                    for (let el of targets) {
                        const btn = el.tagName.toLowerCase() === 'button' ? el : el.querySelector('button');
                        if (btn) {
                            btn.click();
                            break;
                        }
                    }
                });
            }
            
            // Ensure width is unbound globally so Plotly resizes
            doc.querySelectorAll('.block-container').forEach(b => {
                b.style.maxWidth = 'none';
            });
            
            // Sync the custom sidebar toggle arrows to point left or right depending on the drawer state!
            if (customToggle) {
                const s = doc.querySelector('[data-testid="stSidebar"]');
                const svg = customToggle.querySelector('svg');
                if (s && svg) {
                    if (s.getAttribute('aria-expanded') === 'true') {
                        svg.style.transform = 'rotate(180deg)';
                        svg.style.transition = 'transform 0.2s';
                    } else {
                        svg.style.transform = 'rotate(0deg)';
                    }
                }
            }
        }, 300);
        </script>
    """, height=0)

    # ╔══════════════════════════════════════════════════════════════════════════
    #  STICKY NAVBAR  (pure HTML — visual only)
    # ╚══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="cr-nav">
      <div id="custom-sidebar-toggle" style="width: 45px; height: 45px; background: #22D3EE; border-radius: 6px; display: flex; align-items: center; justify-content: center; cursor: pointer; margin-right: 20px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="13 17 18 12 13 7"></polyline>
          <polyline points="6 17 11 12 6 7"></polyline>
        </svg>
      </div>
      <div class="cr-logo">
        <div class="cr-logo-icon">🏟️</div>
        CROWD<span style="color:#22D3EE;">AI</span>
      </div>
      
      <!-- TARGET FOR JS INJECTION -->
      <div class="nav-controls-target" style="display:flex; align-items:center; justify-content:center; flex: 1; padding: 0 20px;"></div>
      
      <div class="nav-stat" style="margin-left:auto;">
        CRIT:&nbsp;<b style="color:{'#EF4444' if crit_cnt>0 else '#22C55E'};">{crit_cnt}</b>
        &nbsp;·&nbsp;SAMPLES:&nbsp;<b style="color:#818CF8;">{n_samples}</b>
        &nbsp;·&nbsp;STEP:&nbsp;<b style="color:#22D3EE;">{step}/{n_points}</b>
      </div>
      <div class="live-pill" style="margin-left:12px;">
        <span class="{"dot-live" if running else "dot-pause"}"></span>
        {"LIVE" if running else "PAUSED"}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ╔══════════════════════════════════════════════════════════════════════════
    #  CONTROLS ROW  — scenario pills · pause
    # ╚══════════════════════════════════════════════════════════════════════════
    # Enable bedrock silently since toggle is removed
    set_use_bedrock(is_bedrock_available())
    
    c1, c2 = st.columns([10.0, 1.0])

    with c1:
        new_sc = st.radio("sc", SCENARIO_KEYS,
                          index=SCENARIO_KEYS.index(scenario),
                          horizontal=True, label_visibility="collapsed",
                          key="sc_radio")
        if new_sc != scenario:
            st.session_state.update(scenario=new_sc, step=ROLLING_WINDOW,
                                    selected_zone="Zone_A", view="heatmap")
            st.session_state.pop(f"hist_{new_sc}", None)
            st.rerun()

    with c2:
        if st.button("⏸" if running else "▶", key="pb", use_container_width=True):
            st.session_state.running = not running
            st.rerun()

    # ╔══════════════════════════════════════════════════════════════════════════
    #  NATIVE SIDEBAR  — opened by ☰ button in left panel
    # ╚══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### Global Status")
        # Risk counts
        rb = st.columns(3)
        for i,(lv,lbl) in enumerate([("green","LOW"),("yellow","WARN"),("red","CRIT")]):
            cnt = sum(1 for p in predictions.values() if p.risk_level==lv)
            rc_ = _rc(lv)
            with rb[i]:
                st.markdown(f"""
                <div style="text-align:center;background:#06101a;border:1px solid #0a1420;
                            border-radius:6px;padding:5px 2px;margin-bottom:15px;">
                  <div style="font-size:17px;font-weight:800;color:{rc_};
                              font-family:'JetBrains Mono',monospace;line-height:1;">{cnt}</div>
                  <div style="font-size:7px;color:#ffffff;margin-top:1px;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("### Zone Selector")
        for zone in zones:
            pred = predictions.get(zone)
            if not pred: continue
            lv   = pred.risk_level
            rc_  = _rc(lv); rbg_ = _rbg(lv)
            zc   = ZONE_COLORS.get(zone,"#22D3EE")
            zn   = ZONE_NAMES.get(zone,zone)
            isel = (zone==selected)
            d    = zd.get(zone,0.); v = zv.get(zone,0.)
            pct  = min(100,pred.risk_probability*100)
            anim = "animation:blink .9s infinite;" if lv=="red" else ""

            st.markdown(f"""
            <div style="background:{'#0e1829' if isel else '#06101a'};
                        border:1px solid {'#1a3050' if isel else '#0a1420'};
                        border-left:3px solid {zc if isel else 'transparent'};
                        border-radius:8px;padding:9px 11px;margin-bottom:5px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                <div style="display:flex;align-items:center;gap:5px;">
                  <span style="background:{zc}18;border:1px solid {zc}38;color:{zc};
                    font-size:9px;font-weight:800;border-radius:4px;padding:1px 5px;
                    font-family:'JetBrains Mono',monospace;">Z{zone[-1]}</span>
                  <span style="font-size:11px;font-weight:600;color:#ffffff;">{zn.split()[0]}</span>
                </div>
                <span style="background:{rbg_};border:1px solid {rc_}44;color:{rc_};
                  font-size:7.5px;font-weight:700;border-radius:3px;padding:2px 6px;
                  font-family:'JetBrains Mono',monospace;{anim}">{_rl(lv)}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:5px;">
                <div><div style="font-size:7px;color:#ffffff;">DENSITY</div>
                  <span style="font-size:12px;color:#EF4444;font-weight:700;font-family:'JetBrains Mono',monospace;">{d:.1f}</span>
                  <span style="color:#ffffff;font-size:7px;"> p/m²</span></div>
                <div><div style="font-size:7px;color:#ffffff;">VELOCITY</div>
                  <span style="font-size:12px;color:#22D3EE;font-weight:700;font-family:'JetBrains Mono',monospace;">{v:.1f}</span>
                  <span style="color:#ffffff;font-size:7px;"> m/s</span></div>
              </div>
              <div style="height:2px;background:#0a1420;border-radius:2px;overflow:hidden;">
                <div style="height:100%;width:{pct:.0f}%;background:{rc_};box-shadow:0 0 4px {rc_}55;border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

            if st.button(f"📍 Select {zn}", key=f"sb_{zone}", use_container_width=True):
                st.session_state.selected_zone = zone; st.rerun()

        st.markdown("---")
        st.markdown("### View")
        sv1, sv2 = st.columns(2)
        with sv1:
            if st.button("🗺 Map", key="sb_map", use_container_width=True):
                st.session_state.view = "heatmap"; st.rerun()
        with sv2:
            if st.button("📊 Charts", key="sb_charts", use_container_width=True):
                st.session_state.view = "charts"; st.rerun()

        st.markdown("---")
        st.markdown(f"""
        <div class="ml-footer">
          <b>ML ENGINE</b><br>
          LogReg · 92% Acc<br>Horizon ~12.5 min<br>7 features · mmWave<br><br>
          <b>THRESHOLDS</b><br>
          🟢 &lt; {YELLOW_THRESHOLD:.0%} &nbsp; 🟡 –{RED_THRESHOLD:.0%} &nbsp; 🔴 &gt;{RED_THRESHOLD:.0%}
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("↺ Reset", key="sb_reset", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    # ╔══════════════════════════════════════════════════════════════════════════
    #  MAIN CONTENT
    # ╚══════════════════════════════════════════════════════════════════════════
    view = st.session_state.view

    # ══════════════════════════════════════════════════════════════════════
    #  HEATMAP VIEW
    # ══════════════════════════════════════════════════════════════════════
    if view == "heatmap":
        pred_sel = predictions.get(selected)
        lv_s  = pred_sel.risk_level if pred_sel else "green"
        rc_s  = _rc(lv_s); rbg_s = _rbg(lv_s)

        ttc = (f'<span style="font-size:9px;color:#EF4444;margin-left:8px;'
               f'font-family:JetBrains Mono,monospace;">'
               f'⏱ ~{pred_sel.time_to_congestion} min</span>'
               if pred_sel and pred_sel.time_to_congestion>0 else "")

        dpills = " &nbsp;·&nbsp; ".join(
            f'<span style="color:{ZONE_COLORS[z]};font-size:9px;font-weight:700;'
            f'font-family:JetBrains Mono,monospace;">'
            f'{z[-1]}: {zd.get(z,0):.1f}</span>'
            for z in zones if z in predictions)

        st.markdown(
            '<div class="hmap-card">'
            '<div class="hmap-hdr">'
            '<div style="display:flex;align-items:center;gap:8px;">'
            f'{_dot(rc_s, lv_s=="red")}'
            '<span style="font-size:13px;font-weight:700;color:#7090b0;">Live Crowd Density · Venue Floor Plan</span>'
            f'{_badge(_rl(lv_s),rc_s,rbg_s,rc_s+"44")}{ttc}'
            '</div>'
            '<div style="font-size:9px;color:#3d5070;font-family:\'JetBrains Mono\',monospace;">'
            f'p/m²: {dpills}'
            '</div>'
            '</div>', unsafe_allow_html=True)

        st.plotly_chart(make_venue_fig(zd, selected, h=450),
                        use_container_width=True,
                        config={"displayModeBar":False},
                        key=f"venue_{step}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="scroll-hint">↓ click 📊 Charts to view analytics</div>',
                    unsafe_allow_html=True)

        # AI incident brief
        if any_crit:
            zsum = {z: dict(risk_probability=p.risk_probability,
                            risk_level=p.risk_level,
                            density=zd.get(z,0.), velocity=zv.get(z,0.),
                            time_to_congestion=p.time_to_congestion)
                    for z,p in predictions.items()}
            bk = f"brief_{step}"
            if is_bedrock_available() and get_use_bedrock():
                if bk not in st.session_state:
                    brief = generate_incident_summary(zsum)
                    st.session_state[bk] = brief or \
                        "⚠️ Multiple zones at critical risk — deploy crowd control immediately."
                    store_incident(incident_id=f"INC-{step}-{int(time.time())}",
                                   zone_data=zsum, summary=st.session_state[bk],
                                   scenario=scenario)
                body = st.session_state[bk]
                lbl  = "🧠 AI INCIDENT BRIEF — Amazon Bedrock"
            else:
                rz = [z for z,p in predictions.items() if p.risk_level=="red"]
                body = (f"CRITICAL: {', '.join(rz)} at RED risk. "
                        "Deploy crowd control teams immediately. Activate alternate routing.")
                lbl  = "⚠️ SYSTEM ALERT"
            st.markdown(f"""
            <div class="incident-card">
              <div class="incident-lbl">{lbl}</div>
              <div class="incident-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  CHARTS VIEW  — selected zone only
    # ══════════════════════════════════════════════════════════════════════
    else:
        zc_s = ZONE_COLORS.get(selected,"#22D3EE")
        zn_s = ZONE_NAMES.get(selected, selected)
        cap  = ZONE_CAPS.get(selected, 300)
        df_s = _hdf(selected)

        # ── Zone tab strip ────────────────────────────────────────────────
        zt = st.columns(len(zones))
        for i, zone in enumerate(zones):
            zc2  = ZONE_COLORS.get(zone,"#22D3EE")
            isel2 = (zone == selected)
            with zt[i]:
                # Style active tab differently
                if isel2:
                    st.markdown(f"""
                    <div style="text-align:center;background:#07101c;
                                border:1px solid {zc2};border-radius:6px;
                                padding:5px 4px;margin-bottom:4px;cursor:default;">
                      <span style="font-size:11px;font-weight:700;color:{zc2};
                                   font-family:'JetBrains Mono',monospace;">
                        ▶ {ZONE_NAMES.get(zone,zone).split()[0]}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    if st.button(ZONE_NAMES.get(zone,zone).split()[0],
                                 key=f"zt_{zone}", use_container_width=True):
                        st.session_state.selected_zone = zone; st.rerun()

        # ── Section header ────────────────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding:6px 0 10px 0;border-bottom:1px solid #0a1420;
                    margin-bottom:10px;">
          <div style="width:3px;height:16px;background:{zc_s};border-radius:2px;"></div>
          <span style="font-size:13px;font-weight:700;color:#7090b0;">
            Analytics — {zn_s}</span>
          <span style="font-size:8.5px;color:#1a2a40;background:#06101a;
            border:1px solid #0a1420;border-radius:4px;padding:2px 7px;
            font-family:'JetBrains Mono',monospace;">
            {n_samples} pts · x-axis grows live</span>
        </div>""", unsafe_allow_html=True)

        # ── Derive columns ─────────────────────────────────────────────────
        if len(df_s) > 0:
            df_s = df_s.copy()
            df_s["headcount_est"] = (df_s["density"]/9.*cap).clip(0,cap+50).astype(int)
            df_s["risk_idx"] = (df_s["density"]/df_s["velocity"].clip(lower=0.1)).clip(0,20)/20

        # ── ROW 1 ─────────────────────────────────────────────────────────
        st.markdown('<div class="row-lbl">Live Metrics — Selected Zone</div>',
                    unsafe_allow_html=True)
        r1 = st.columns(4)
        chart_card(r1[0], df_s, "density", "Density", "p/m²", "#EF4444",
                   [0,10],[0,2,4,6,8,10], 4.0,"cong.",
                   f"r1_d_{selected}_{step}", n_samples)
        chart_card(r1[1], df_s, "velocity", "Velocity", "m/s", "#F97316",
                   [0,2],[0,.5,1.,1.5,2.], .5,"slow",
                   f"r1_v_{selected}_{step}", n_samples)
        chart_card(r1[2], df_s, "headcount_est", "Headcount", "ppl", "#22D3EE",
                   [0,cap+50],[0,cap//4,cap//2,cap*3//4,cap],
                   None,"", f"r1_h_{selected}_{step}", n_samples)
        chart_card(r1[3], df_s, "risk_idx", "Risk Index", "", "#818CF8",
                   [0,1],[0,.25,.5,.75,1.], .7,"hi",
                   f"r1_r_{selected}_{step}", n_samples)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── ROW 2  — all-zones comparison ──────────────────────────────────
        st.markdown('<div class="row-lbl">All Zones Comparison</div>',
                    unsafe_allow_html=True)
        r2 = st.columns(2)

        all_dfs = {z: _hdf(z) for z in zones}

        with r2[0]:
            st.markdown("""
            <div class="chart-card">
              <div class="chart-hdr">
                <div style="display:flex;align-items:center;gap:5px;">
                  <div style="width:8px;height:2px;background:#c8d8f0;border-radius:1px;"></div>
                  <span class="chart-lbl">All Zones · Density</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(
                overlay_chart(all_dfs,"density","p/m²",[0,10],[0,2,4,6,8,10],4.0),
                use_container_width=True,
                config={"displayModeBar":False}, key=f"r2_d_{step}")
            st.markdown(f'<div class="chart-n">{n_samples} pts</div>',
                         unsafe_allow_html=True)

        with r2[1]:
            st.markdown("""
            <div class="chart-card">
              <div class="chart-hdr">
                <div style="display:flex;align-items:center;gap:5px;">
                  <div style="width:8px;height:2px;background:#c8d8f0;border-radius:1px;"></div>
                  <span class="chart-lbl">All Zones · Velocity</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(
                overlay_chart(all_dfs,"velocity","m/s",[0,2],[0,.5,1.,1.5,2.],.5),
                use_container_width=True,
                config={"displayModeBar":False}, key=f"r2_v_{step}")
            st.markdown(f'<div class="chart-n">{n_samples} pts</div>',
                         unsafe_allow_html=True)

        # ── Signage for selected zone ──────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        pred_z = predictions.get(selected)
        if pred_z:
            lv_  = pred_z.risk_level
            rc_  = _rc(lv_); rbg_ = _rbg(lv_)
            st.markdown(f"""
            <div class="sig-card" style="background:{rbg_};border-color:{rc_};">
              <div class="sig-lbl">📺 {selected} — {zn_s}</div>
              <div class="sig-txt">{pred_z.signage_message}</div>
            </div>""", unsafe_allow_html=True)

    # ╔══════════════════════════════════════════════════════════════════════════
    #  STATUS BAR
    # ╚══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="cr-statusbar">
      <span>🧠 LOGISTIC REGRESSION · 92% ACC · 7 FEATURES · HORIZON ~12.5 MIN</span>
      <span>·</span><span>📡 mmWAVE RADAR · PRIVACY-FIRST · NO CAMERAS</span>
    </div>""", unsafe_allow_html=True)

    # ── Auto-advance ───────────────────────────────────────────────────────────
    if running and step < n_points - 1:
        time.sleep(float(st.session_state.update_speed))
        st.session_state.step = step + 1
        st.rerun()
    elif step >= n_points - 1:
        st.success("✅ Simulation complete — switch scenario or ↺ reset.")
        st.balloons()


if __name__ == "__main__":
    main()