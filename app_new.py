"""
TextInsightMiner — Dig Deeper. Classify Smarter. (v10.1)
================================================================================
Sidebar navigation. Hardened PII (8 patterns). Reporting suite.
Premium landing page with grid/orbs/glass mockup.
Vectorized Polars engine. Corporate teal/slate/gold palette.
DSL operators: AND | OR | NOT | NEAR (vectorized ≤5) | NOT LIKE.
================================================================================
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import re, hashlib, json, logging, io, os, duckdb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
SUPPORTED_FORMATS = ['csv','xlsx','xls','parquet','json']
DOMAIN_PACKS_DIR = "domain_packs"

SPEAKER_PATTERNS = {
    'agent': re.compile(r'^(?:agent|representative|rep|support|advisor|associate|consultant)\s*[:\-\|]', re.I|re.M),
    'customer': re.compile(r'^(?:customer|client|caller|member|subscriber|user|guest|visitor)\s*[:\-\|]', re.I|re.M),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SVG ICONS
# ═══════════════════════════════════════════════════════════════════════════════
class IC:
    _B='xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"'
    SEARCH=f'<svg {_B}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
    SETTINGS=f'<svg {_B}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
    UPLOAD=f'<svg {_B}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
    DOWNLOAD=f'<svg {_B}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
    CHECK=f'<svg {_B}><polyline points="20 6 9 17 4 12"/></svg>'
    ALERT=f'<svg {_B}><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    TREE=f'<svg {_B}><rect x="2" y="2" width="8" height="4" rx="1"/><rect x="14" y="2" width="8" height="4" rx="1"/><rect x="2" y="18" width="8" height="4" rx="1"/><rect x="14" y="18" width="8" height="4" rx="1"/><path d="M6 6v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V6"/><path d="M6 18v-4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v4"/></svg>'
    LAYERS=f'<svg {_B}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>'
    TABLE=f'<svg {_B}><rect width="18" height="18" x="3" y="3" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>'
    SHIELD=f'<svg {_B}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
    TOOL=f'<svg {_B}><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>'
    ZAPPER=f'<svg {_B}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
    BAR=f'<svg {_B}><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>'
    PIE=f'<svg {_B}><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>'
    GLOBE=f'<svg {_B}><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
    EYE=f'<svg {_B}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'
    ACTIVITY=f'<svg {_B}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
    ALERT_OCT=f'<svg {_B}><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    TRENDING=f'<svg {_B}><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
    SAVE=f'<svg {_B}><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>'
    PICKAXE=f'<svg {_B}><path d="M14.531 12.469 6.619 20.38a1 1 0 1 1-3-3l7.912-7.912"/><path d="M15.686 4.314A12.5 12.5 0 0 0 5.461 2.958 1 1 0 0 0 5.58 4.71a22 22 0 0 1 6.318 3.393"/><path d="M17.7 3.7a1 1 0 0 0-1.4 0l-4.6 4.6a1 1 0 0 0 0 1.4l2.6 2.6a1 1 0 0 0 1.4 0l4.6-4.6a1 1 0 0 0 0-1.4z"/></svg>'
    REPORT=f'<svg {_B}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
    INFO=f'<svg {_B}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'

    @staticmethod
    def icon(svg, color="#6B8A99", size=18):
        s=svg.replace('stroke="currentColor"',f'stroke="{color}"')
        s=s.replace('width="18"',f'width="{size}"').replace('height="18"',f'height="{size}"')
        return f'<span style="display:inline-flex;align-items:center;vertical-align:middle;margin-right:6px">{s}</span>'

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--teal:#2D5F6E;--teal-l:#3A7A8C;--slate:#6B8A99;--steel:#A8BCC8;--warm:#D1CFC4;--warm-l:#E8E6DD;
    --gold:#D4B94E;--gold-l:#E8D97A;--bg:#F5F4F0;--card:#FFFFFF;--border:#D1CFC4;
    --text:#1E2D33;--text2:#3D5A66;--muted:#6B8A99;--success:#3D7A5F;--warn:#B8963E;--err:#A04040}
.stApp{font-family:'DM Sans',sans-serif;background:var(--bg)}
.stApp h1,.stApp h2,.stApp h3{font-family:'DM Sans',sans-serif;font-weight:600;color:var(--text)}
code{font-family:'JetBrains Mono',monospace;font-size:13px}
section[data-testid="stSidebar"]{background:var(--warm-l)!important;border-right:1px solid var(--warm)!important}
section[data-testid="stSidebar"] *{color:var(--text)!important}
.mc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px 16px;text-align:center;
    border-top:3px solid var(--teal);box-shadow:0 1px 4px rgba(45,95,110,0.06);transition:all .2s}
.mc:hover{box-shadow:0 4px 16px rgba(45,95,110,0.1);transform:translateY(-1px)}
.mv{font-size:22px;font-weight:700;color:var(--text);margin:0;line-height:1.2}
.ml{font-size:10px;font-weight:600;color:var(--muted);margin:5px 0 0;text-transform:uppercase;letter-spacing:.7px}
.sh{display:flex;align-items:center;gap:8px;margin:28px 0 14px;font-size:15px;font-weight:600;color:var(--text);
    padding-bottom:8px;border-bottom:2px solid var(--warm)}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-right:3px}
.tag-b{background:#D6E8EE;color:var(--teal)}.tag-r{background:#F2D6D6;color:var(--err)}
.tag-g{background:#D4E8DC;color:var(--success)}.tag-a{background:#F0E6C8;color:#7A6620}.tag-p{background:#E0D6EE;color:#5A3D7A}
.cl{background:var(--card);padding:12px 16px;margin:5px 0;border-radius:8px;border-left:3px solid var(--teal);
    box-shadow:0 1px 4px rgba(45,95,110,0.05);font-size:13px;line-height:1.7}
.ckw{background:var(--gold-l);padding:2px 6px;border-radius:4px;font-weight:700;color:var(--text);border:1px solid var(--gold)}
.cmeta{font-size:11px;color:var(--muted);margin-bottom:5px;font-style:italic}
.badge{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:5px;font-size:12px;font-weight:600}
.b-ok{background:#D4E8DC;color:var(--success)}.b-warn{background:#F0E6C8;color:#7A6620}.b-info{background:#D6E8EE;color:var(--teal)}
.rc{background:var(--warm-l);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:5px;font-size:12px}
.conflict-row{background:#F9EDED;border:1px solid #E8C8C8;border-radius:8px;padding:10px 14px;margin:4px 0;font-size:12px}
.lvl-tbl{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
.lvl-tbl th{background:var(--teal);color:#fff;font-weight:600;padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.lvl-tbl td{padding:8px 14px;border-bottom:1px solid var(--warm-l);color:var(--text)}
.lvl-tbl tr:nth-child(even){background:var(--warm-l)}.lvl-tbl tr:hover{background:#D6E8EE}
.lvl-tbl .num{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px}
.lvl-tbl .bar{background:var(--warm);border-radius:4px;height:7px;overflow:hidden;min-width:80px}
.lvl-tbl .bfill{height:100%;border-radius:4px}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif;font-weight:500;color:var(--muted);font-size:14px}
.stTabs [aria-selected="true"]{color:var(--teal)!important;border-bottom-color:var(--teal)!important;font-weight:600}
div[data-testid="stForm"]{border:1px solid var(--border);border-radius:10px;padding:16px;background:var(--card)}
.stButton>button[kind="primary"]{background:var(--teal)!important;border-color:var(--teal)!important}
.stButton>button[kind="primary"]:hover{background:var(--teal-l)!important}
.stProgress>div>div>div{background:var(--teal)!important}
footer,.stDeployButton{display:none!important}
</style>
"""

def mcard(label,value,color="var(--teal)"):
    return f'<div class="mc" style="border-top-color:{color}"><p class="mv">{value}</p><p class="ml">{label}</p></div>'

def shdr(svg,text):
    st.markdown(f'<div class="sh">{IC.icon(svg,"#2D5F6E",20)}{text}</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DSLRule:
    rule_id: str = ""
    terms: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)
    proximity: Optional[int] = None
    position: Optional[str] = None
    within_words: int = 50
    sent_by: Optional[str] = None
    exclude_if: List[str] = field(default_factory=list)
    l1: str = "Uncategorized"; l2: str = "NA"; l3: str = "NA"; l4: str = "NA"; source: str = "manual"

    @property
    def term1(self): return self.terms[0] if self.terms else ""
    @property
    def term2(self): return self.terms[1] if len(self.terms)>1 else None
    @property
    def term3(self): return self.terms[2] if len(self.terms)>2 else None
    @property
    def operator1(self): return self.operators[0] if self.operators else None
    @property
    def operator2(self): return self.operators[1] if len(self.operators)>1 else None

    def to_dict(self):
        return {"rule_id":self.rule_id,"terms":self.terms,"operators":self.operators,
            "proximity":self.proximity,"position":self.position,"within_words":self.within_words,
            "sent_by":self.sent_by,"exclude_if":self.exclude_if,
            "l1":self.l1,"l2":self.l2,"l3":self.l3,"l4":self.l4,"source":self.source}

    @classmethod
    def from_dict(cls, d):
        if 'terms' in d:
            terms=[t for t in d['terms'] if t and str(t).strip()]
            operators=d.get('operators',[])
        else:
            terms=[str(d[k]).strip() for k in ('term1','term2','term3') if d.get(k) and str(d[k]).strip()]
            operators=[str(d[k]).strip() for k in ('operator1','operator2') if d.get(k) and str(d[k]).strip() and str(d[k])!="None"]
        return cls(rule_id=d.get("rule_id",""),terms=terms,operators=operators,
            proximity=d.get("proximity"),position=d.get("position"),
            within_words=d.get("within_words",50),sent_by=d.get("sent_by"),
            exclude_if=d.get("exclude_if",[]),l1=d.get("l1","Uncategorized"),
            l2=d.get("l2","NA"),l3=d.get("l3","NA"),l4=d.get("l4","NA"),source=d.get("source","manual"))

# ═══════════════════════════════════════════════════════════════════════════════
# PII REDACTION — HARDENED (8 patterns)
# ═══════════════════════════════════════════════════════════════════════════════
class PIIRedactor:
    PATS={
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'CARD':  r'\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b',
        'SSN':   r'\b\d{3}-\d{2}-\d{4}\b',
        'MRN':   r'\b(?:MRN|Medical\s*Record|Patient\s*ID)[:\s#]+[A-Z0-9]{5,12}\b',
        'DOB':   r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
        'IP':    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        'PHONE': r'(?:\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ADDR':  r'\b\d{1,5}\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Circle|Cir|Apt|Suite|Ste|Unit)\b',
    }
    @classmethod
    def redact_polars(cls, col: pl.Series, mode='hash') -> pl.Series:
        r = col
        for pt, pat in cls.PATS.items():
            tag = f"[{pt}]" if mode=='token' else f"[{pt}:REDACTED]"
            try:
                r = r.str.replace_all(pat, tag)
            except Exception:
                pass  # Skip patterns that fail on specific data
        return r

# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED DSL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class DSLEngine:
    def __init__(self, rules: List[DSLRule]):
        self.rules=rules; self._prep=self._prepare()
        logger.info(f"DSLEngine: {len(rules)} rules ready")

    def _mkpat(self, term):
        if not term or not term.strip(): return None
        esc=re.escape(term.strip().lower()).replace(r'\*',r'\w*')
        return rf'(?i)\b{esc}\b'

    def _near_pat(self, pat_a, pat_b, n):
        """Build bidirectional NEAR regex: A within N words of B (or B within N of A)."""
        # Strip per-term (?i) flags; apply single outer flag
        a = re.sub(r'^\(\?i\)', '', pat_a)
        b = re.sub(r'^\(\?i\)', '', pat_b)
        gap = r'\W+(?:\w+\W+){0,' + str(n) + r'}'
        return rf'(?i)(?:{a}{gap}{b}|{b}{gap}{a})'

    def _prepare(self):
        pr=[]
        for r in self.rules:
            pats=[self._mkpat(t) for t in r.terms]; pats=[p for p in pats if p]
            if not pats: continue
            ex=[self._mkpat(e) for e in r.exclude_if if e.strip()]; ex=[e for e in ex if e]
            sc=10
            for t in r.terms:
                if t and t.strip(): sc+=len(t.split())*10
            if r.operators: sc+=10
            if any(op=="NEAR" for op in r.operators): sc+=15
            if any(op=="NOT LIKE" for op in r.operators): sc+=5
            if r.position in ("START","END"): sc+=5
            pr.append({'rule':r,'pats':pats,'ops':r.operators,'ex':ex,'sc':sc})
        pr.sort(key=lambda x:x['sc'],reverse=True)
        return pr

    def classify_batch(self, df: pl.DataFrame, text_col: str) -> pl.DataFrame:
        n=len(df); ts=df[text_col].cast(pl.Utf8).fill_null(""); tlow=ts.str.to_lowercase()
        best_sc=np.zeros(n,dtype=np.float32); best_ri=np.full(n,-1,dtype=np.int32)
        for idx,p in enumerate(self._prep):
            r=p['rule']; pats=p['pats']; ops=p['ops']
            if not pats: continue
            mm=tlow.str.contains(pats[0]).to_numpy()
            for ti in range(1,len(pats)):
                if not mm.any(): break
                op=ops[ti-1].upper() if ti-1<len(ops) else "AND"
                ti_mask=tlow.str.contains(pats[ti]).to_numpy()
                if op=="AND": mm=mm&ti_mask
                elif op=="OR": mm=mm|ti_mask
                elif op=="NOT": mm=mm&~ti_mask
                elif op=="NOT LIKE":
                    # Vectorized: match term1 but exclude rows containing term2 pattern
                    mm=mm&~ti_mask
                elif op=="NEAR":
                    prox=r.proximity or 5
                    if prox<=5:
                        # Vectorized regex path — stays in Polars/Rust
                        prev_pat=pats[ti-1] if ti>1 else pats[0]
                        near_regex=self._near_pat(prev_pat, pats[ti], prox)
                        mm=mm&tlow.str.contains(near_regex).to_numpy()
                    else:
                        # Fallback: positional word-index check for larger windows
                        mm=mm&ti_mask
                        if mm.any():
                            ni=np.where(mm)[0]; txts=tlow.gather(ni.tolist()).to_list()
                            r1,r2=re.compile(pats[ti-1] if ti>1 else pats[0]),re.compile(pats[ti])
                            for li,txt in enumerate(txts):
                                ps1=[len(txt[:m.start()].split()) for m in r1.finditer(txt)]
                                ps2=[len(txt[:m.start()].split()) for m in r2.finditer(txt)]
                                if not any(abs(a-b)<=prox for a in ps1 for b in ps2): mm[ni[li]]=False
            for ex in p['ex']:
                if mm.any(): mm=mm&~tlow.str.contains(ex).to_numpy()
            if r.position in ("START","END") and mm.any():
                ni=np.where(mm)[0]; txts=ts.gather(ni.tolist()).to_list()
                for li,txt in enumerate(txts):
                    w=txt.split(); wn=min(r.within_words,len(w))
                    win=' '.join(w[:wn]) if r.position=="START" else ' '.join(w[-wn:])
                    if not re.search(pats[0],win,re.I): mm[ni[li]]=False
            if r.sent_by and r.sent_by.lower()!='any' and mm.any():
                ni=np.where(mm)[0]; txts=ts.gather(ni.tolist()).to_list()
                tp=SPEAKER_PATTERNS.get(r.sent_by.lower())
                if tp:
                    for li,txt in enumerate(txts):
                        if any(sp.search(txt) for sp in SPEAKER_PATTERNS.values()):
                            lines=re.split(r'\n|\r\n?',txt); stxt=[]; cap=False
                            for ln in lines:
                                if tp.match(ln.strip()): cap=True; stxt.append(re.sub(tp,'',ln.strip(),count=1).strip())
                                elif any(sp.match(ln.strip()) for sp in SPEAKER_PATTERNS.values()): cap=False
                                elif cap: stxt.append(ln.strip())
                            if not re.search(pats[0],' '.join(stxt),re.I): mm[ni[li]]=False
            better=mm&(p['sc']>best_sc); best_sc[better]=p['sc']; best_ri[better]=idx

        l1s,l2s,l3s,l4s,confs,mrules,mscs,mdets=[],[],[],[],[],[],[],[]
        for i in range(n):
            ri=best_ri[i]
            if ri<0:
                l1s.append("Uncategorized");l2s.append("NA");l3s.append("NA");l4s.append("NA")
                confs.append(0.0);mrules.append("");mscs.append(0.0);mdets.append("")
            else:
                rr=self._prep[ri]['rule']
                l1s.append(rr.l1);l2s.append(rr.l2);l3s.append(rr.l3);l4s.append(rr.l4)
                confs.append(min(best_sc[i]/100,1.0));mrules.append(rr.rule_id);mscs.append(float(best_sc[i]))
                parts=[]
                for ti,t in enumerate(rr.terms):
                    if ti>0 and ti-1<len(rr.operators): parts.append(rr.operators[ti-1])
                    parts.append(t)
                mdets.append(' '.join(parts))
        return pl.DataFrame({'l1':l1s,'l2':l2s,'l3':l3s,'l4':l4s,'confidence':confs,'matched_rule':mrules,'match_score':mscs,'match_detail':mdets})

    def classify_single(self, text):
        if not text or not isinstance(text,str) or not text.strip():
            return {'l1':'Uncategorized','l2':'NA','l3':'NA','l4':'NA','confidence':0.0,'matched_rule':'','match_score':0.0,'match_detail':''}
        return self.classify_batch(pl.DataFrame({'t':[text]}),'t').row(0,named=True)

    def test_rule(self, text):
        if not text: return []
        tl=text.lower(); hits=[]
        for p in self._prep:
            r=p['rule']; pats=p['pats']
            if not pats or not re.search(pats[0],tl): continue
            ok=True
            for ti in range(1,len(pats)):
                op=p['ops'][ti-1].upper() if ti-1<len(p['ops']) else "AND"
                has=bool(re.search(pats[ti],tl))
                if op=="AND" and not has: ok=False; break
                elif op=="NOT" and has: ok=False; break
                elif op=="NOT LIKE" and has: ok=False; break
                elif op=="NEAR":
                    if not has: ok=False; break
                    prox=r.proximity or 5
                    if prox<=5:
                        prev_pat=pats[ti-1] if ti>1 else pats[0]
                        near_regex=self._near_pat(prev_pat, pats[ti], prox)
                        if not re.search(near_regex, tl): ok=False; break
                    else:
                        r1,r2=re.compile(pats[ti-1] if ti>1 else pats[0]),re.compile(pats[ti])
                        ps1=[len(tl[:m.start()].split()) for m in r1.finditer(tl)]
                        ps2=[len(tl[:m.start()].split()) for m in r2.finditer(tl)]
                        if not any(abs(a-b)<=prox for a in ps1 for b in ps2): ok=False; break
                elif op=="OR" and not has: pass  # OR: at least one match is enough
            if not ok: continue
            for ex in p['ex']:
                if re.search(ex,tl): ok=False; break
            if ok:
                parts=[]
                for ti,t in enumerate(r.terms):
                    if ti>0 and ti-1<len(r.operators): parts.append(r.operators[ti-1])
                    parts.append(t)
                hits.append({'rule_id':r.rule_id,'score':p['sc'],'detail':' '.join(parts),'l1':r.l1,'l2':r.l2,'l3':r.l3,'l4':r.l4})
        hits.sort(key=lambda x:x['score'],reverse=True); return hits

    def detect_conflicts(self):
        tm={}
        for r in self.rules:
            k=r.term1.strip().lower() if r.term1 else ""
            if k: tm.setdefault(k,[]).append(r)
        return [{'term':t,'categories':', '.join(sorted(set(r.l1 for r in rs))),'rule_ids':', '.join(r.rule_id for r in rs)}
                for t,rs in tm.items() if len(set(r.l1 for r in rs))>1]

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN LOADER / FILE HANDLER / RUN HISTORY / PIPELINE / CONCORDANCE
# ═══════════════════════════════════════════════════════════════════════════════
class DomainLoader:
    def __init__(self): self.industries={}
    def auto_load(self):
        if not os.path.exists(DOMAIN_PACKS_DIR): return 0
        rr=os.path.join(DOMAIN_PACKS_DIR,"rules.json"); rk=os.path.join(DOMAIN_PACKS_DIR,"keywords.json")
        if os.path.exists(rr) and os.path.exists(rk): return self._load_merged(rr,rk)
        return self._load_subdirs()
    def _load_merged(self,rp,kp):
        try:
            with open(rp,encoding='utf-8') as f: ar=json.load(f)
            with open(kp,encoding='utf-8') as f: ak=json.load(f)
        except Exception as e: logger.error(f"Merged load fail: {e}"); return 0
        for e in ar: d=e.get('domain','Other'); self.industries.setdefault(d,{'r':[],'k':[]})['r'].append(e)
        for e in ak: d=e.get('domain','Other'); self.industries.setdefault(d,{'r':[],'k':[]})['k'].append(e)
        logger.info(f"Loaded {len(self.industries)} domains"); return len(self.industries)
    def _load_subdirs(self):
        n=0
        for item in os.listdir(DOMAIN_PACKS_DIR):
            p=os.path.join(DOMAIN_PACKS_DIR,item)
            if not os.path.isdir(p) or item.startswith('.'): continue
            rp,kp=os.path.join(p,"rules.json"),os.path.join(p,"keywords.json")
            if os.path.exists(rp) and os.path.exists(kp):
                try:
                    with open(rp,encoding='utf-8') as f: ru=json.load(f)
                    with open(kp,encoding='utf-8') as f: kw=json.load(f)
                    self.industries[item]={'r':ru,'k':kw}; n+=1
                except Exception as e: logger.error(f"Failed {item}: {e}")
        return n
    def get_rules(self,ind):
        d=self.industries.get(ind,{}); rules=[]; c=0
        for src in ('r','k'):
            for item in d.get(src,[]):
                cat=item.get('set',{})
                for cond in item.get('conditions',[]):
                    c+=1; rules.append(DSLRule(rule_id=f"{ind[:3].upper()}-{c:04d}",terms=[cond],
                        l1=cat.get('category','Uncategorized'),l2=cat.get('subcategory','NA'),
                        l3=cat.get('level_3','NA'),l4=cat.get('level_4','NA'),source='json'))
        return rules
    def get_industries(self): return list(self.industries.keys())

class FH:
    @staticmethod
    def read(uf):
        try:
            fsz=uf.size/(1024*1024); ext=Path(uf.name).suffix.lower()[1:]
            s=st.empty(); s.info(f"Reading {uf.name} ({fsz:.1f}MB)...")
            if ext=='csv': df=pl.read_csv(uf)
            elif ext in ('xlsx','xls'): df=pl.from_pandas(pd.read_excel(uf))
            elif ext=='parquet': df=pl.read_parquet(uf)
            elif ext=='json': df=pl.read_json(uf)
            else: st.error(f"Unsupported: {ext}"); return None
            s.success(f"Loaded {len(df):,} records")
            if len(df)>=1000 and ext!='parquet':
                o=st.empty(); o.info("Optimizing...")
                b=io.BytesIO(); df.write_parquet(b,compression='snappy')
                psz=len(b.getvalue())/(1024*1024); b.seek(0); df=pl.read_parquet(b)
                o.success(f"Optimized: {fsz:.1f}->{psz:.1f}MB ({fsz/psz:.1f}x)")
            return df
        except Exception as e: st.error(f"Error: {e}"); return None
    @staticmethod
    def save(df,fmt='csv'):
        b=io.BytesIO()
        if fmt=='csv': df.write_csv(b)
        elif fmt=='parquet': df.write_parquet(b)
        elif fmt=='xlsx': df.to_pandas().to_excel(b,index=False,engine='openpyxl')
        b.seek(0); return b.getvalue()

class RunHistory:
    def __init__(self):
        self.conn=duckdb.connect(':memory:')
        self.conn.execute("CREATE TABLE IF NOT EXISTS rh(rid INT,ts TIMESTAMP,tot INT,cat INT,unc INT,pct DOUBLE,ul1 INT,ul2 INT,secs DOUBLE,rps DOUBLE,rn INT,pii INT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS rh_hits(rid INT,rule_id VARCHAR,hits INT,l1 VARCHAR)")
        self._n=1
    def record(self,df,pt,rn,pii):
        tot=len(df);unc=int((df['Category']=='Uncategorized').sum());cat=tot-unc;rid=self._n;self._n+=1
        self.conn.execute("INSERT INTO rh VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",[rid,datetime.now(),tot,cat,unc,cat/tot*100 if tot>0 else 0,df['Category'].nunique(),df['Subcategory'].nunique(),pt,tot/pt if pt>0 else 0,rn,pii])
        if 'matched_rule' in df.columns:
            h=df[df['matched_rule']!=''].groupby(['matched_rule','Category']).size().reset_index(name='c')
            for _,r in h.iterrows(): self.conn.execute("INSERT INTO rh_hits VALUES(?,?,?,?)",[rid,r['matched_rule'],int(r['c']),r['Category']])
    def get_history(self): return self.conn.execute("SELECT * FROM rh ORDER BY ts DESC").fetchdf()
    def get_drift(self):
        df=self.conn.execute("SELECT * FROM rh ORDER BY ts DESC LIMIT 2").fetchdf()
        if len(df)<2: return None
        c,p=df.iloc[0],df.iloc[1]; return {'d_pct':c['pct']-p['pct'],'d_spd':c['rps']-p['rps']}
    def get_rule_perf(self): return self.conn.execute("SELECT rule_id,SUM(hits) h,l1 FROM rh_hits GROUP BY rule_id,l1 ORDER BY h DESC").fetchdf()
    def get_uncat(self,df,n=10):
        u=df[df['Category']=='Uncategorized']
        if len(u)==0: return pd.DataFrame()
        s=u.sample(min(n,len(u))); cols=[c for c in ['Conversation_ID','Original_Text'] if c in s.columns]
        return s[cols].reset_index(drop=True) if cols else s.head(n).reset_index(drop=True)

class Pipeline:
    def __init__(self,eng,pii=True): self.eng=eng;self.pii=pii;self.pii_n=0
    def process(self,df,tc,ic,mode='hash',cb=None):
        n=len(df)
        # Keep all original columns — only classify on text column
        if self.pii: PIIRedactor.redact_polars(df[tc].cast(pl.Utf8).fill_null(""),mode)
        if cb: cb(int(n*0.1),n)
        cls=self.eng.classify_batch(df,tc)
        if cb: cb(int(n*0.9),n)
        f=pl.concat([df,cls],how='horizontal')
        # Rename ID and text columns to standard names; keep all others intact
        rename_map={ic:'Conversation_ID',tc:'Original_Text','l1':'Category','l2':'Subcategory','l3':'L3','l4':'L4'}
        # Only rename if source != target (avoid error when column name already matches)
        rename_map={k:v for k,v in rename_map.items() if k in f.columns and k!=v}
        f=f.rename(rename_map)
        if cb: cb(n,n)
        return f

class Concordance:
    STOP=frozenset('the a an and or but in on at to for of with by from as is was are were be been being have has had do does did will would should could may might can i you he she it we they my your his her its our their this that these those'.split())
    def __init__(self,df): self.df=df
    def search(self,kw,ctx=10,cat=None):
        s=self.df
        if cat and cat!="All": s=s[s['Category']==cat]
        escaped=re.escape(kw)
        p=re.compile(escaped,re.I) if ' ' in kw.strip() else re.compile(rf'\b{escaped}\b',re.I)
        res=[]
        for _,row in s.iterrows():
            t=str(row.get('Original_Text',''))
            for m in p.finditer(t):
                l=t[:m.start()].split();r=t[m.end():].split()
                res.append({'Conversation_ID':row.get('Conversation_ID',''),'Left':' '.join(l[-ctx:]),'KW':m.group(),'Right':' '.join(r[:ctx]),'Category':row.get('Category',''),'Subcategory':row.get('Subcategory','')})
        return res
    def stats(self,r):
        if not r: return {'n':0,'uc':0,'cats':0,'avg':0}
        d=pd.DataFrame(r);uc=d['Conversation_ID'].nunique()
        return {'n':len(r),'uc':uc,'cats':d['Category'].nunique(),'avg':round(len(r)/uc,1) if uc>0 else 0}
    def colloc(self,r,n=8):
        if not r: return {'l':[],'r':[]}
        lw,rw=[],[]
        for x in r:
            lw.extend(w.strip('.,!?;:').lower() for w in x['Left'].split() if w.strip('.,!?;:').lower() not in self.STOP and len(w)>2)
            rw.extend(w.strip('.,!?;:').lower() for w in x['Right'].split() if w.strip('.,!?;:').lower() not in self.STOP and len(w)>2)
        return {'l':Counter(lw).most_common(n),'r':Counter(rw).most_common(n)}
    def export(self,r,fmt='csv'):
        d=pd.DataFrame(r);b=io.BytesIO()
        if fmt=='csv': d.to_csv(b,index=False)
        else: d.to_excel(b,index=False,engine='openpyxl')
        b.seek(0); return b.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def build_tree_data(df):
    if df.empty: return {"name":"No Data","value":0,"children":[]}
    total=len(df); root={"name":f"All ({total:,})","value":total,"children":[]}
    for l1,l1d in df.groupby('Category'):
        if l1 in ('Uncategorized','NA'): continue
        n1={"name":f"{l1} ({len(l1d):,})","value":len(l1d),"children":[]}
        for l2,l2d in l1d.groupby('Subcategory'):
            if l2=='NA': continue
            n2={"name":f"{l2} ({len(l2d):,})","value":len(l2d),"children":[]}
            for l3,l3d in l2d.groupby('L3'):
                if l3=='NA': continue
                l4c=[{"name":f"{v} ({c:,})","value":c} for v,c in l3d['L4'].value_counts().items() if v!='NA']
                if l4c: n2["children"].append({"name":f"{l3} ({len(l3d):,})","value":len(l3d),"children":l4c})
                else: n2["children"].append({"name":f"{l3} ({len(l3d):,})","value":len(l3d)})
            n1["children"].append(n2 if n2.get("children") else {"name":f"{l2} ({len(l2d):,})","value":len(l2d)})
        root["children"].append(n1)
    return root

def get_tree_option(data):
    return {"tooltip":{"trigger":"item","triggerOn":"mousemove","backgroundColor":"rgba(30,45,51,0.95)","borderWidth":0,
        "textStyle":{"color":"#E8E6DD","fontSize":13,"fontFamily":"DM Sans"},"formatter":"{b}",
        "extraCssText":"border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.2)"},
        "series":[{"type":"tree","data":[data],"top":"4%","left":"10%","bottom":"4%","right":"24%",
            "symbolSize":10,"orient":"LR","layout":"orthogonal","edgeShape":"curve","edgeForkPosition":"50%",
            "roam":True,"scaleLimit":{"min":0.3,"max":5},
            "label":{"show":True,"position":"right","distance":14,"fontSize":12,"fontFamily":"DM Sans",
                "color":"#1E2D33","fontWeight":500,"backgroundColor":"rgba(255,255,255,0.95)",
                "padding":[5,10],"borderRadius":6,"borderColor":"#D1CFC4","borderWidth":1,
                "shadowBlur":4,"shadowColor":"rgba(0,0,0,0.06)"},
            "leaves":{"label":{"fontSize":11,"color":"#3D5A66","fontWeight":400,
                "backgroundColor":"rgba(245,244,240,0.95)","padding":[4,8],"borderRadius":4,"borderColor":"#D1CFC4","borderWidth":1}},
            "expandAndCollapse":True,"animationDuration":500,"animationDurationUpdate":700,
            "animationEasingUpdate":"cubicInOut","initialTreeDepth":2,
            "lineStyle":{"color":"#A8BCC8","width":1.5,"curveness":0.4},
            "itemStyle":{"color":"#2D5F6E","borderColor":"#3A7A8C","borderWidth":2,
                "shadowBlur":6,"shadowColor":"rgba(45,95,110,0.25)"},
            "emphasis":{"focus":"descendant","itemStyle":{"color":"#D4B94E","borderColor":"#B8963E","borderWidth":3,
                "shadowBlur":12,"shadowColor":"rgba(212,185,78,0.4)"},
                "lineStyle":{"color":"#D4B94E","width":2.5},
                "label":{"fontWeight":700,"fontSize":13,"color":"#1E2D33","backgroundColor":"rgba(232,217,122,0.3)","borderColor":"#D4B94E"}}}]}

def build_sunburst(df):
    clean=df[~df['Category'].isin(['Uncategorized','NA'])].copy()
    if clean.empty: return None
    agg=clean.groupby(['Category','Subcategory','L3','L4']).size().reset_index(name='count')
    agg=agg[agg['count']>len(clean)*0.003]
    fig=px.sunburst(agg,path=['Category','Subcategory','L3','L4'],values='count',color='Category',
        color_discrete_sequence=['#2D5F6E','#6B8A99','#A8BCC8','#D4B94E','#3D7A5F','#8C6B4A','#5A7A6B','#4A6B8C'],height=650)
    fig.update_traces(textinfo="label+value+percent root",insidetextorientation='radial')
    fig.update_layout(margin=dict(t=20,l=0,r=0,b=0),font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)')
    return fig

def build_level_table(df,level_col,parent_col=None):
    total=len(df); rows=[]; clr='#2D5F6E'
    if parent_col:
        agg=df.groupby([parent_col,level_col]).size().reset_index(name='Count')
        agg=agg[agg[level_col]!='NA'].sort_values('Count',ascending=False)
        agg['%']=((agg['Count']/total)*100).round(1); cols_order=[parent_col,level_col,'Count','%']
    else:
        agg=df[level_col].value_counts().reset_index(); agg.columns=[level_col,'Count']
        agg=agg[agg[level_col]!='NA']; agg['%']=((agg['Count']/total)*100).round(1); cols_order=[level_col,'Count','%']
    for _,r in agg.iterrows():
        cells=''.join(f'<td class="num">{r[c]:,}</td>' if c=='Count' else f'<td class="num">{r[c]}%</td>' if c=='%' else f'<td>{r[c]}</td>' for c in cols_order)
        bw=min(r['%']*2,100)
        cells+=f'<td><div class="bar"><div class="bfill" style="width:{bw}%;background:{clr}"></div></div></td>'
        rows.append(f'<tr>{cells}</tr>')
    hdr=''.join(f'<th style="text-align:{"right" if c in ("Count","%") else "left"}">{c}</th>' for c in cols_order)
    hdr+='<th style="min-width:80px"></th>'
    return f'<table class="lvl-tbl"><thead><tr>{hdr}</tr></thead><tbody>{"".join(rows)}</tbody></table>'

# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def render_reports(out):
    shdr(IC.REPORT,"Reports & Analytics")
    rt1,rt2,rt3,rt4=st.tabs(["Top Drivers","Category Heatmap","Uncategorized Analysis","Summary"])

    with rt1:
        st.markdown("**Top Category Drivers by Volume**")
        top_n=st.slider("Show top N",5,30,15,key="rpt_topn")
        cat_counts=out['Category'].value_counts().head(top_n).reset_index()
        cat_counts.columns=['Category','Count']
        cat_counts=cat_counts[cat_counts['Category']!='Uncategorized']
        fig=px.bar(cat_counts,x='Count',y='Category',orientation='h',
            color='Count',color_continuous_scale=['#A8BCC8','#2D5F6E'],height=max(300,top_n*28))
        fig.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
            font_family="DM Sans",margin=dict(l=0,r=20,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig,use_container_width=True,key="rr_fig_cat")

        # L2 top drivers
        st.markdown("**Top Subcategory Drivers**")
        sub_counts=out[out['Subcategory']!='NA']['Subcategory'].value_counts().head(top_n).reset_index()
        sub_counts.columns=['Subcategory','Count']
        fig2=px.bar(sub_counts,x='Count',y='Subcategory',orientation='h',
            color='Count',color_continuous_scale=['#E8D97A','#D4B94E'],height=max(300,top_n*28))
        fig2.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
            font_family="DM Sans",margin=dict(l=0,r=20,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False)
        st.plotly_chart(fig2,use_container_width=True,key="rr_fig_sub")

    with rt2:
        st.markdown("**Category vs Subcategory Concentration**")
        cross=out[out['Category']!='Uncategorized'].groupby(['Category','Subcategory']).size().reset_index(name='Count')
        cross=cross[cross['Subcategory']!='NA']
        if not cross.empty:
            pivot=cross.pivot_table(index='Category',columns='Subcategory',values='Count',fill_value=0)
            # Keep top 15 subcategories by total volume
            top_subs=cross.groupby('Subcategory')['Count'].sum().nlargest(15).index.tolist()
            pivot=pivot[[c for c in top_subs if c in pivot.columns]]
            fig=px.imshow(pivot,color_continuous_scale=['#F5F4F0','#2D5F6E'],
                labels=dict(color="Count"),height=max(400,len(pivot)*35))
            fig.update_layout(font_family="DM Sans",margin=dict(l=0,r=0,t=30,b=0),paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True,key="rr_fig_heat")
        else:
            st.info("No categorized data for heatmap.")

    with rt3:
        st.markdown("**Uncategorized Record Analysis — Top N-grams**")
        unc=out[out['Category']=='Uncategorized']
        if len(unc)>0:
            st.markdown(f'<span class="badge b-warn">{len(unc):,} uncategorized records ({len(unc)/len(out)*100:.1f}%)</span>',unsafe_allow_html=True)
            # Extract top unigrams and bigrams
            stop=Concordance.STOP
            words=[]
            for t in unc['Original_Text'].dropna().tolist():
                ws=[w.strip('.,!?;:').lower() for w in str(t).split() if w.strip('.,!?;:').lower() not in stop and len(w)>2]
                words.extend(ws)
            if words:
                uni=Counter(words).most_common(20)
                uni_df=pd.DataFrame(uni,columns=['Word','Count'])
                fig=px.bar(uni_df,x='Count',y='Word',orientation='h',color='Count',
                    color_continuous_scale=['#D1CFC4','#A04040'],height=500)
                fig.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
                    font_family="DM Sans",margin=dict(l=0,r=20,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False,title="Top words in uncategorized records")
                st.plotly_chart(fig,use_container_width=True,key="rr_fig_unc")
                st.markdown("**Use these words to build new classification rules.**")
            # Bigrams
            bigrams=[]
            for t in unc['Original_Text'].dropna().tolist():
                ws=[w.strip('.,!?;:').lower() for w in str(t).split() if w.strip('.,!?;:').lower() not in stop and len(w)>2]
                bigrams.extend(zip(ws,ws[1:]))
            if bigrams:
                bi=Counter(bigrams).most_common(15)
                bi_df=pd.DataFrame([(' '.join(b),c) for b,c in bi],columns=['Bigram','Count'])
                st.dataframe(bi_df,hide_index=True,use_container_width=True)
        else:
            st.success("No uncategorized records — full coverage achieved.")

    with rt4:
        st.markdown("**Executive Summary**")
        total=len(out); unc_n=int((out['Category']=='Uncategorized').sum()); cat_n=total-unc_n
        sm1,sm2,sm3=st.columns(3)
        with sm1: st.markdown(mcard("Total Records",f"{total:,}"),unsafe_allow_html=True)
        with sm2: st.markdown(mcard("Categorized",f"{cat_n:,} ({cat_n/total*100:.1f}%)","var(--success)"),unsafe_allow_html=True)
        with sm3: st.markdown(mcard("Uncategorized",f"{unc_n:,} ({unc_n/total*100:.1f}%)","var(--err)" if unc_n/total>.1 else "var(--gold)"),unsafe_allow_html=True)
        st.markdown("---")
        # Top 5 categories
        st.markdown("**Top 5 Categories**")
        top5=out['Category'].value_counts().head(6).reset_index()
        top5.columns=['Category','Count']; top5=top5[top5['Category']!='Uncategorized'].head(5)
        top5['%']=(top5['Count']/total*100).round(1)
        st.dataframe(top5,hide_index=True,use_container_width=True)
        # Confidence distribution
        if 'confidence' in out.columns:
            st.markdown("**Confidence Distribution**")
            cd=pd.cut(out['confidence'],bins=[0,0.3,0.7,1.01],labels=['Low','Medium','High'],include_lowest=True).value_counts().reset_index()
            cd.columns=['Band','Count']
            st.dataframe(cd,hide_index=True,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RULE BUILDER TAB
# ═══════════════════════════════════════════════════════════════════════════════
def _get_hier(rules):
    t={}
    for r in rules:
        if r.l1 not in ('NA','Uncategorized'):
            t.setdefault(r.l1,{})
            if r.l2!='NA': t[r.l1].setdefault(r.l2,{})
            if r.l2!='NA' and r.l3!='NA': t[r.l1][r.l2].setdefault(r.l3,set())
            if r.l2!='NA' and r.l3!='NA' and r.l4!='NA': t[r.l1][r.l2][r.l3].add(r.l4)
    return t

def render_rule_builder(industry_rules):
    if 'custom_rules' not in st.session_state: st.session_state.custom_rules=[]
    all_r=industry_rules+st.session_state.custom_rules; hier=_get_hier(all_r)

    # ── Operator color map ──
    OP_COLORS={"AND":("#2D5F6E","#D6E8EE"),"OR":("#2563EB","#DBEAFE"),"NOT":("#A04040","#F2D6D6"),
               "NEAR":("#7A6620","#F0E6C8"),"NOT LIKE":("#5A3D7A","#E0D6EE")}

    def _op_badge(op):
        fg,bg=OP_COLORS.get(op,("#6B8A99","#E8E6DD"))
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:5px;font-size:12px;font-weight:700;background:{bg};color:{fg};margin:0 6px;letter-spacing:.5px">{op}</span>'

    def _term_pill(term):
        if not term or not term.strip(): return ''
        return f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;font-size:13px;font-weight:500;background:#FFFFFF;color:#1E2D33;border:1px solid #D1CFC4;margin:0 2px;font-family:\'JetBrains Mono\',monospace">{term}</span>'

    def _build_preview_html(terms, ops, prox=None):
        if not terms or not any(t.strip() for t in terms): return '<span style="color:#6B8A99;font-size:13px;font-style:italic">Add terms below to build your rule expression...</span>'
        parts=[]
        for i,t in enumerate(terms):
            if not t.strip(): continue
            if i>0 and i-1<len(ops) and ops[i-1]:
                op_label=ops[i-1]
                if op_label=="NEAR" and prox: op_label=f"NEAR({prox})"
                parts.append(_op_badge(ops[i-1].replace(f"({prox})","") if prox else ops[i-1]).replace(ops[i-1],op_label) if "NEAR" in (ops[i-1] or "") and prox else _op_badge(ops[i-1]))
            parts.append(_term_pill(t))
        return ''.join(parts)

    # ── Init dynamic term state ──
    if 'rb_term_count' not in st.session_state: st.session_state.rb_term_count=2
    if 'rb_terms_list' not in st.session_state: st.session_state.rb_terms_list=['']*2
    if 'rb_ops_list' not in st.session_state: st.session_state.rb_ops_list=['AND']

    sub_upload,sub_create,sub_edit,sub_test,sub_guide=st.tabs(["Upload Rules","Create Rule","Edit Rules","Test Rules","Guide"])

    with sub_upload:
        st.markdown('<p style="color:var(--muted);font-size:13px">Upload a JSON rules file.</p>',unsafe_allow_html=True)
        upl=st.file_uploader("Load Rules JSON",type=['json'],key='rb_upload_main')
        if upl and upl.name!=st.session_state.get('_last_uploaded_rules'):
            try:
                data=json.load(upl)
                if isinstance(data,list):
                    for d in data: st.session_state.custom_rules.append(DSLRule.from_dict(d))
                    st.session_state['_last_uploaded_rules']=upl.name
                    st.success(f"Loaded {len(data)} rules"); st.rerun()
            except Exception as e: st.error(f"Parse error: {e}")

    with sub_create:
        # ── Handle prefill from Reports/Concordance ──
        prefill_term=st.session_state.pop('_rb_prefill_term',None)
        if prefill_term:
            st.session_state['rb_t_0']=prefill_term
            st.markdown(f'<span class="badge b-info">Pre-filled from: "{prefill_term}"</span>',unsafe_allow_html=True)

        tc=st.session_state.rb_term_count

        # ── Category / Level selection ──
        c1,c2=st.columns([3,1])
        with c1: cat_name=st.text_input("Category Name *",placeholder="e.g. Subscription Cancellation",key="rb_cat_name")
        with c2: cat_level=st.selectbox("Level",["L1","L2","L3","L4"],key="rb_cat_level")
        parent_l1=parent_l2=parent_l3=""
        if cat_level in ("L2","L3","L4"): parent_l1=st.selectbox("Parent L1",["(select)"]+sorted(hier.keys()),key="rb_pl1")
        if cat_level in ("L3","L4") and parent_l1 in hier: parent_l2=st.selectbox("Parent L2",["(select)"]+sorted(hier.get(parent_l1,{}).keys()),key="rb_pl2")
        if cat_level=="L4" and parent_l1 in hier and parent_l2 in hier.get(parent_l1,{}): parent_l3=st.selectbox("Parent L3",["(select)"]+sorted(hier.get(parent_l1,{}).get(parent_l2,{}).keys()),key="rb_pl3")

        st.markdown("---")

        # ── Collect current terms and ops from widgets ──
        live_terms=[]
        live_ops=[]

        # Read current widget values (from previous render cycle)
        for i in range(tc):
            live_terms.append(st.session_state.get(f'rb_t_{i}',''))
        for i in range(tc-1):
            live_ops.append(st.session_state.get(f'rb_op_{i}','AND'))

        # ── Proximity (global for any NEAR in the chain) ──
        has_near=any(op=="NEAR" for op in live_ops)
        prox_val=st.session_state.get('rb_prox_dyn',3)

        # ── Live Preview (above inputs, always visible) ──
        preview_html=_build_preview_html(live_terms, live_ops, prox_val if has_near else None)
        st.markdown(f"""<div style="background:var(--warm-l);border:1px solid var(--border);border-radius:10px;padding:14px 18px;
            margin-bottom:16px;min-height:44px;display:flex;align-items:center;flex-wrap:wrap;gap:4px">
            <span style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-right:10px">Rule</span>
            {preview_html}</div>""",unsafe_allow_html=True)

        # ── Operator color legend ──
        legend=''.join(_op_badge(op) for op in OP_COLORS.keys())
        st.markdown(f'<div style="margin-bottom:12px;font-size:11px;color:var(--muted)">Available operators: {legend}</div>',unsafe_allow_html=True)

        # ── Dynamic term + operator rows ──
        for i in range(tc):
            if i>0:
                # Operator selector between terms
                op_cols=st.columns([1,2,1])
                with op_cols[1]:
                    op_val=st.selectbox(f"Operator {i}",["AND","OR","NOT","NEAR","NOT LIKE"],
                        key=f"rb_op_{i-1}",label_visibility="collapsed",
                        index=["AND","OR","NOT","NEAR","NOT LIKE"].index(live_ops[i-1]) if i-1<len(live_ops) and live_ops[i-1] in ["AND","OR","NOT","NEAR","NOT LIKE"] else 0)

            # Term input
            st.text_input(f"Term {i+1}",placeholder=f"Enter word or phrase (term {i+1})",key=f"rb_t_{i}",label_visibility="collapsed")

        # ── NEAR proximity (only visible when NEAR is used) ──
        # Re-read ops after widget render
        current_ops=[st.session_state.get(f'rb_op_{i}','AND') for i in range(tc-1)]
        if any(op=="NEAR" for op in current_ops):
            st.slider("NEAR Proximity (max word distance)",1,5,3,key="rb_prox_dyn",help="How many words apart the terms can be for a NEAR match")

        # ── Add / Remove term buttons ──
        btn_cols=st.columns([1,1,4])
        with btn_cols[0]:
            if st.button("+ Add Term",key="rb_add_term",type="secondary",use_container_width=True):
                st.session_state.rb_term_count+=1
                st.session_state.rb_terms_list.append('')
                st.session_state.rb_ops_list.append('AND')
                st.rerun()
        with btn_cols[1]:
            if tc>2 and st.button("- Remove",key="rb_rem_term",use_container_width=True):
                st.session_state.rb_term_count-=1
                st.session_state.rb_terms_list=st.session_state.rb_terms_list[:tc-1]
                st.session_state.rb_ops_list=st.session_state.rb_ops_list[:tc-2]
                # Clean up removed widget keys
                if f'rb_t_{tc-1}' in st.session_state: del st.session_state[f'rb_t_{tc-1}']
                if f'rb_op_{tc-2}' in st.session_state: del st.session_state[f'rb_op_{tc-2}']
                st.rerun()

        # ── Advanced options ──
        with st.expander("Advanced Options"):
            ac1,ac2,ac3=st.columns(3)
            with ac1: pos=st.selectbox("Position",["ANY","START","END"],key="rb_pos")
            with ac2: ww=st.slider("Within Words",10,200,50,key="rb_ww")
            with ac3: sb=st.selectbox("Sent By",["Any","Agent","Customer"],key="rb_sb")
            excl=st.text_input("Exclude If (comma-sep)",key="rb_excl")

        # ── Save Rule ──
        if st.button("Save Rule",type="primary",key="rb_save",use_container_width=True):
            final_terms=[st.session_state.get(f'rb_t_{i}','').strip() for i in range(tc)]
            final_terms=[t for t in final_terms if t]
            final_ops=[st.session_state.get(f'rb_op_{i}','AND') for i in range(tc-1)]
            final_ops=final_ops[:len(final_terms)-1] if len(final_terms)>1 else []

            if final_terms and cat_name.strip():
                l1=l2=l3=l4="NA"
                if cat_level=="L1": l1=cat_name.strip()
                elif cat_level=="L2": l1=parent_l1 if parent_l1!="(select)" else "NA"; l2=cat_name.strip()
                elif cat_level=="L3": l1=parent_l1 if parent_l1!="(select)" else "NA"; l2=parent_l2 if parent_l2!="(select)" else "NA"; l3=cat_name.strip()
                elif cat_level=="L4": l1=parent_l1 if parent_l1!="(select)" else "NA"; l2=parent_l2 if parent_l2!="(select)" else "NA"; l3=parent_l3 if parent_l3!="(select)" else "NA"; l4=cat_name.strip()
                ex_list=[x.strip() for x in excl.split(',') if x.strip()] if excl else []
                prox_final=st.session_state.get('rb_prox_dyn',3) if any(o=="NEAR" for o in final_ops) else None
                rid=f"USR-{len(st.session_state.custom_rules)+1:04d}"
                st.session_state.custom_rules.append(DSLRule(rule_id=rid,terms=final_terms,operators=final_ops,
                    proximity=prox_final,position=pos if pos!="ANY" else None,within_words=ww,
                    sent_by=sb if sb!="Any" else None,exclude_if=ex_list,l1=l1,l2=l2,l3=l3,l4=l4,source="manual"))
                # Reset builder
                st.session_state.rb_term_count=2
                for k in list(st.session_state.keys()):
                    if k.startswith('rb_t_') or k.startswith('rb_op_'): del st.session_state[k]
                st.success(f"Rule {rid} saved: {' '.join(final_terms)}"); st.rerun()
            else:
                st.warning("Enter at least one term and a category name.")

    with sub_edit:
        cr=st.session_state.custom_rules
        if cr:
            st.markdown(f'<span class="badge b-info">{len(cr)} custom rules</span>',unsafe_allow_html=True)

            # ── Visual preview of all rules ──
            for ri,r in enumerate(cr):
                preview=_build_preview_html(r.terms, r.operators, r.proximity if r.proximity else None)
                cat_path=' > '.join([x for x in [r.l1,r.l2,r.l3,r.l4] if x and x!='NA'])
                st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin:4px 0;
                    border-left:3px solid var(--teal);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
                        <span style="font-size:11px;font-weight:700;color:var(--teal);margin-right:8px">{r.rule_id}</span>{preview}
                    </div>
                    <span style="font-size:11px;color:var(--muted);font-style:italic">{cat_path}</span>
                </div>""",unsafe_allow_html=True)

            st.markdown("---")
            # ── Editable table (fallback for bulk ops) ──
            with st.expander("Bulk Edit (Table View)"):
                edit_data=[{'rule_id':r.rule_id,'terms':' | '.join(r.terms),'operators':' | '.join(r.operators) if r.operators else '',
                    'l1':r.l1,'l2':r.l2,'l3':r.l3,'l4':r.l4,'proximity':r.proximity or 0,
                    'position':r.position or 'ANY','sent_by':r.sent_by or 'Any','exclude_if':', '.join(r.exclude_if)} for r in cr]
                edited=st.data_editor(pd.DataFrame(edit_data),hide_index=True,use_container_width=True,num_rows="dynamic",
                    column_config={'terms':st.column_config.TextColumn("Terms (pipe-sep)",width="large"),
                        'operators':st.column_config.TextColumn("Operators (AND|OR|NOT|NEAR|NOT LIKE)",width="medium"),
                        'position':st.column_config.SelectboxColumn("Position",options=["ANY","START","END"]),
                        'sent_by':st.column_config.SelectboxColumn("Sent By",options=["Any","Agent","Customer"])})
                if st.button("Apply Changes",key="rb_apply",type="primary"):
                    new_rules=[]
                    for _,row in edited.iterrows():
                        terms=[t.strip() for t in str(row['terms']).split('|') if t.strip()]
                        ops=[o.strip().upper() for o in str(row['operators']).split('|') if o.strip()] if row['operators'] else []
                        excl=[e.strip() for e in str(row['exclude_if']).split(',') if e.strip()] if row['exclude_if'] else []
                        prx=int(row['proximity']) if row['proximity'] else None
                        new_rules.append(DSLRule(rule_id=row['rule_id'] or f"USR-{len(new_rules)+1:04d}",
                            terms=terms,operators=ops,proximity=prx if prx else None,
                            position=row['position'] if row['position']!='ANY' else None,within_words=50,
                            sent_by=row['sent_by'] if row['sent_by']!='Any' else None,exclude_if=excl,
                            l1=row['l1'] or 'NA',l2=row['l2'] or 'NA',l3=row['l3'] or 'NA',l4=row['l4'] or 'NA',source="manual"))
                    st.session_state.custom_rules=new_rules; st.success(f"Applied {len(new_rules)} rules"); st.rerun()

            ec1,ec2=st.columns(2)
            with ec1: st.download_button("Save to File",json.dumps([r.to_dict() for r in cr],indent=2,default=str),"rules.json","application/json",use_container_width=True,key="rb_dl_rules")
            with ec2:
                if st.button("Clear All",key="rb_clr"): st.session_state.custom_rules=[]; st.rerun()
        else: st.info("No custom rules. Upload or create rules first.")

    with sub_test:
        tt=st.text_area("Paste sample text",height=100,key="rb_tt")
        if st.button("Test All Rules",key="rb_tb") and tt:
            eng=DSLEngine(all_r); hits=eng.test_rule(tt)
            if hits:
                for h in hits:
                    parts=h['detail'].split()
                    # Rebuild with color badges
                    vis_parts=[]
                    for p in parts:
                        if p.upper() in OP_COLORS: vis_parts.append(_op_badge(p.upper()))
                        else: vis_parts.append(_term_pill(p))
                    vis=''.join(vis_parts)
                    cat_path=' > '.join([x for x in [h['l1'],h['l2'],h['l3'],h['l4']] if x and x!='NA'])
                    st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin:4px 0;
                        border-left:3px solid var(--success)">
                        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
                                <span style="font-size:11px;font-weight:700;color:var(--success);margin-right:8px">{h['rule_id']}</span>{vis}
                            </div>
                            <div style="text-align:right">
                                <span style="font-size:11px;color:var(--muted)">{cat_path}</span>
                                <span class="badge b-ok" style="margin-left:8px">Score: {h['score']}</span>
                            </div>
                        </div>
                    </div>""",unsafe_allow_html=True)
            else: st.info("No rules matched.")

    with sub_guide:
        # ── GUIDE: How to Use the Rule Builder ──
        st.markdown(f'<div style="background:linear-gradient(135deg,#2D5F6E 0%,#3A7A8C 100%);border-radius:12px;padding:24px 28px;margin-bottom:20px"><span style="font-size:20px;font-weight:700;color:#FFFFFF">{IC.icon(IC.INFO,"#FFFFFF",22)}Rule Builder Guide</span><br><span style="font-size:13px;color:#A8BCC8">Everything you need to classify text data using rules.</span></div>',unsafe_allow_html=True)

        g1,g2=st.tabs(["Quick Start","Operators & Examples"])

        with g1:
            st.markdown(f'<div class="sh">{IC.icon(IC.ZAPPER,"#2D5F6E",18)}Quick Start — Your First Rule in 60 Seconds</div>',unsafe_allow_html=True)

            _steps=[
                ("1","Go to Create Rule tab","Enter a <b>Category Name</b> and select the <b>Level</b> (L1 = top level, L2 = subcategory, etc.)."),
                ("2","Add your terms","Type a keyword or phrase into <b>Term 1</b>. Terms can be single words or multi-word phrases."),
                ("3","Pick an operator & add more terms","Select an operator from the dropdown, then type Term 2. Click <b>+ Add Term</b> to chain more."),
                ("4","Watch the live preview","The color-coded rule expression updates in real time above the inputs."),
                ("5","Save & run analysis","Click <b>Save Rule</b>. Go to <b>Upload &amp; Analyse</b>, upload data, and run."),
            ]
            for num,title,desc in _steps:
                st.markdown(f'<table style="border:none;margin:0 0 6px 0"><tr><td style="vertical-align:top;padding:0 12px 0 0;border:none"><span style="display:inline-block;width:28px;height:28px;background:#2D5F6E;color:#fff;border-radius:50%;text-align:center;line-height:28px;font-size:13px;font-weight:700">{num}</span></td><td style="border:none;padding:0"><b style="font-size:13px;color:var(--text)">{title}</b><br><span style="color:var(--muted);font-size:12px">{desc}</span></td></tr></table>',unsafe_allow_html=True)

            st.markdown(f'<div class="sh">{IC.icon(IC.TREE,"#2D5F6E",18)}Understanding the 4-Level Hierarchy</div>',unsafe_allow_html=True)
            st.markdown("""<table class="lvl-tbl"><thead><tr><th>Level</th><th>Purpose</th><th>Example</th></tr></thead><tbody>
            <tr><td><b>L1 — Category</b></td><td>Broadest grouping</td><td>Billing</td></tr>
            <tr><td><b>L2 — Subcategory</b></td><td>Specific issue</td><td>Overcharge</td></tr>
            <tr><td><b>L3 — Detail</b></td><td>Narrower breakdown</td><td>Double Charge</td></tr>
            <tr><td><b>L4 — Granular</b></td><td>Most specific</td><td>Duplicate Transaction</td></tr>
            </tbody></table>""",unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:var(--muted);margin-top:8px"><b>Tip:</b> Start with L1 categories first. Add L2/L3/L4 later for high-volume categories.</p>',unsafe_allow_html=True)

            st.markdown(f'<div class="sh">{IC.icon(IC.ACTIVITY,"#2D5F6E",18)}Recommended Workflow</div>',unsafe_allow_html=True)
            _wf=[("1. Upload Data","Upload your CSV/Excel with a text column"),
                 ("2. Run Initial Analysis","See how many records are uncategorized"),
                 ("3. Check Uncategorized","Reports tab shows top words in unclassified records"),
                 ("4. Build Rules","Create rules using those top words as terms"),
                 ("5. Re-run & Iterate","Re-analyse, check coverage, refine rules")]
            wf_html='<table style="width:100%;border-collapse:collapse">'
            for step,desc in _wf:
                wf_html+=f'<tr><td style="padding:8px 12px;border-bottom:1px solid var(--warm-l);font-weight:600;color:var(--teal);font-size:13px;white-space:nowrap">{step}</td><td style="padding:8px 12px;border-bottom:1px solid var(--warm-l);color:var(--muted);font-size:12px">{desc}</td></tr>'
            wf_html+='</table>'
            st.markdown(wf_html,unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:var(--muted);margin-top:10px"><b>Pro tip:</b> Use <b>Concordance</b> to search for a keyword in context before building a rule.</p>',unsafe_allow_html=True)

        with g2:
            st.markdown(f'<div class="sh">{IC.icon(IC.LAYERS,"#2D5F6E",18)}Operator Reference</div>',unsafe_allow_html=True)

            _ops=[
                ("AND","#2D5F6E","#D6E8EE","Both terms must appear in the text.","billing <b>AND</b> overcharge","I was overcharged on my billing","My billing looks fine"),
                ("OR","#2563EB","#DBEAFE","Either term (or both) must appear.","refund <b>OR</b> reimbursement","I need a refund","I love this product"),
                ("NOT","#A04040","#F2D6D6","First term must appear, second must NOT.","cancel <b>NOT</b> retention","I want to cancel my plan","cancel but retention offer was good"),
                ("NEAR","#7A6620","#F0E6C8","Both terms within N words of each other.","cancel <b>NEAR(3)</b> subscription","cancel my subscription","cancel the order and check subscription later"),
                ("NOT LIKE","#5A3D7A","#E0D6EE","First term must appear, exclude if second found.","charge <b>NOT LIKE</b> service charge","extra charge on my bill","The service charge is standard"),
            ]
            for op_name,fg,bg,desc,example,match,nomatch in _ops:
                st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin-bottom:8px;border-left:4px solid {fg}">
                <span style="display:inline-block;padding:3px 12px;border-radius:5px;font-size:13px;font-weight:700;background:{bg};color:{fg};margin-right:8px">{op_name}</span>
                <span style="font-size:13px;color:var(--text)">{desc}</span><br>
                <span style="font-family:monospace;font-size:12px;background:var(--warm-l);padding:3px 10px;border-radius:4px;display:inline-block;margin:8px 0">{example}</span><br>
                <span style="font-size:12px;color:var(--success)">&#10003; {match}</span> &nbsp; <span style="font-size:12px;color:var(--err)">&#10007; {nomatch}</span>
                </div>""",unsafe_allow_html=True)

            st.markdown(f'<div class="sh">{IC.icon(IC.TOOL,"#2D5F6E",18)}Chaining Operators</div>',unsafe_allow_html=True)
            st.markdown('<p style="font-size:13px;color:var(--text);margin-bottom:12px">Operators are evaluated <b>left to right</b> between consecutive terms. Chain unlimited terms.</p>',unsafe_allow_html=True)

            st.markdown("""<div style="background:var(--warm-l);border-radius:8px;padding:12px 16px;margin-bottom:8px">
            <span style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px">3-term rule</span><br>
            <span style="font-family:monospace;font-size:13px;margin-top:6px;display:inline-block">
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">billing</span>
            <span style="padding:3px 8px;background:#D6E8EE;color:#2D5F6E;border-radius:4px;font-weight:700;margin:0 4px">AND</span>
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">overcharge</span>
            <span style="padding:3px 8px;background:#F2D6D6;color:#A04040;border-radius:4px;font-weight:700;margin:0 4px">NOT</span>
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">resolved</span></span><br>
            <span style="font-size:12px;color:var(--muted)">Text must contain billing AND overcharge, but NOT resolved</span></div>""",unsafe_allow_html=True)

            st.markdown("""<div style="background:var(--warm-l);border-radius:8px;padding:12px 16px;margin-bottom:16px">
            <span style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px">OR + NOT LIKE</span><br>
            <span style="font-family:monospace;font-size:13px;margin-top:6px;display:inline-block">
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">refund</span>
            <span style="padding:3px 8px;background:#DBEAFE;color:#2563EB;border-radius:4px;font-weight:700;margin:0 4px">OR</span>
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">chargeback</span>
            <span style="padding:3px 8px;background:#E0D6EE;color:#5A3D7A;border-radius:4px;font-weight:700;margin:0 4px">NOT LIKE</span>
            <span style="padding:3px 8px;background:#fff;border:1px solid #D1CFC4;border-radius:4px">approved</span></span><br>
            <span style="font-size:12px;color:var(--muted)">Text must contain refund or chargeback, but NOT approved</span></div>""",unsafe_allow_html=True)

            st.markdown(f'<div class="sh">{IC.icon(IC.SETTINGS,"#2D5F6E",18)}Advanced Features</div>',unsafe_allow_html=True)
            st.markdown("""<table class="lvl-tbl"><thead><tr><th>Feature</th><th>What It Does</th><th>When to Use</th></tr></thead><tbody>
            <tr><td><b>Position</b></td><td>Match only in first/last N words</td><td>Greetings, sign-offs</td></tr>
            <tr><td><b>Within Words</b></td><td>Word window for position match</td><td>Short/long texts</td></tr>
            <tr><td><b>Sent By</b></td><td>Match agent or customer turns only</td><td>Speaker-specific phrases</td></tr>
            <tr><td><b>Exclude If</b></td><td>Comma-separated exclusions</td><td>Quick multi-exclusion</td></tr>
            <tr><td><b>Wildcards (*)</b></td><td>cancel* = cancel, cancelled, cancellation</td><td>Catch all word forms</td></tr>
            </tbody></table>""",unsafe_allow_html=True)

            st.markdown(f'<div class="sh">{IC.icon(IC.ALERT,"#2D5F6E",18)}Tips for Better Classification</div>',unsafe_allow_html=True)
            _tips=[
                "**Start broad, then narrow.** Create L1 rules first. Add L2/L3 later for high-volume categories.",
                "**Use Reports tab.** Uncategorized Analysis shows top words — best candidates for new rules.",
                "**Use Concordance first.** Search a keyword in context before building a rule.",
                "**Test before running.** Paste sample text in Test Rules to verify matches.",
                "**NEAR for flexible phrases.** Catches variant word orders.",
                "**NOT LIKE for false positives.** Filter out known non-issues.",
                "**Check Rule Performance.** Rules that never fire may need broader terms.",
                "**Export & reuse.** Save rules to JSON. Upload in future sessions or share with team.",
            ]
            for tip in _tips:
                st.markdown(f'<span style="color:var(--success);font-weight:700">&#10003;</span> {tip}',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
LANDING_HTML="""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
header[data-testid="stHeader"],footer,.stDeployButton,section[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
@keyframes fadeUp{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
@keyframes gradSweep{0%{background-position:0% center}100%{background-position:200% center}}
@keyframes pulse1{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:.3}50%{transform:translate(-45%,-55%) scale(1.2);opacity:.5}}
@keyframes pulse2{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:.2}50%{transform:translate(-55%,-45%) scale(1.25);opacity:.4}}
@keyframes float3d{0%,100%{transform:perspective(1000px) rotateX(2deg) rotateY(-1deg) translateY(0)}50%{transform:perspective(1000px) rotateX(-1deg) rotateY(1deg) translateY(-14px)}}
@keyframes barG1{0%{width:0}100%{width:72%}}@keyframes barG2{0%{width:0}100%{width:54%}}
@keyframes barG3{0%{width:0}100%{width:88%}}@keyframes barG4{0%{width:0}100%{width:41%}}
@keyframes barG5{0%{width:0}100%{width:65%}}@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes gridP{0%,100%{opacity:.04}50%{opacity:.08}}
.lp *{margin:0;padding:0;box-sizing:border-box;font-family:'DM Sans',sans-serif}
.lp-hero{position:relative;min-height:100vh;background:#0C1418;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 24px}
.lp-grd{position:absolute;inset:0;background-image:linear-gradient(rgba(168,188,200,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(168,188,200,.05) 1px,transparent 1px);background-size:52px 52px;animation:gridP 8s ease-in-out infinite;z-index:1;pointer-events:none}
.lp-o1{position:absolute;width:800px;height:800px;border-radius:50%;background:radial-gradient(circle,rgba(45,95,110,.45) 0%,transparent 70%);top:15%;left:20%;transform:translate(-50%,-50%);filter:blur(100px);animation:pulse1 10s ease-in-out infinite;z-index:0;pointer-events:none}
.lp-o2{position:absolute;width:600px;height:600px;border-radius:50%;background:radial-gradient(circle,rgba(212,185,78,.3) 0%,transparent 70%);top:65%;left:75%;transform:translate(-50%,-50%);filter:blur(80px);animation:pulse2 12s ease-in-out infinite;z-index:0;pointer-events:none}
.lp-bdg{position:relative;z-index:2;display:inline-flex;align-items:center;gap:6px;background:rgba(212,185,78,.08);color:#D4B94E;padding:8px 22px;border-radius:24px;font-size:11px;font-weight:600;letter-spacing:2px;border:1px solid rgba(212,185,78,.2);margin-bottom:32px;animation:fadeUp .7s ease-out both;backdrop-filter:blur(4px)}
.lp-bdg::before{content:'';width:6px;height:6px;border-radius:50%;background:#D4B94E;box-shadow:0 0 8px rgba(212,185,78,.6)}
.lp-ttl{position:relative;z-index:2;font-size:clamp(44px,7.5vw,78px);font-weight:700;line-height:1.05;text-align:center;margin-bottom:14px;letter-spacing:-1px;background:linear-gradient(90deg,#6B8A99,#E8E6DD 20%,#D4B94E 40%,#E8E6DD 60%,#A8BCC8 80%,#6B8A99);background-size:200% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:fadeUp .7s ease-out .15s both,gradSweep 5s linear infinite}
.lp-sub{position:relative;z-index:2;font-size:19px;color:#A8BCC8;text-align:center;font-style:italic;margin-bottom:18px;animation:fadeUp .7s ease-out .3s both}
.lp-dsc{position:relative;z-index:2;font-size:15px;color:#4A6B78;text-align:center;max-width:560px;line-height:1.8;margin:0 auto 50px;animation:fadeUp .7s ease-out .45s both}
.lp-mk{position:relative;z-index:2;width:min(700px,92vw);margin:0 auto;animation:fadeUp .8s ease-out .6s both,float3d 7s ease-in-out 2s infinite}
.lp-wn{background:rgba(22,36,42,.5);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border:1px solid rgba(168,188,200,.12);border-radius:16px;overflow:hidden;box-shadow:0 40px 100px rgba(0,0,0,.5)}
.lp-wh{display:flex;align-items:center;gap:8px;padding:16px 20px;background:rgba(12,20,24,.7);border-bottom:1px solid rgba(168,188,200,.08)}
.lp-dt{width:12px;height:12px;border-radius:50%}.lp-dr{background:#A04040}.lp-dy{background:#D4B94E}.lp-dg{background:#3D7A5F}
.lp-wt{font-size:12px;color:#4A6B78;margin-left:10px;font-family:'JetBrains Mono',monospace}
.lp-wb{padding:26px 24px;font-family:'JetBrains Mono',monospace;font-size:13px;line-height:2;color:#6B8A99}
.ck{color:#D4B94E}.cf{color:#A8BCC8}.cs{color:#3D7A5F}.cm{color:#3D5A66;font-style:italic}
.lp-cur{display:inline-block;width:2px;height:16px;background:#D4B94E;animation:blink 1s step-end infinite;vertical-align:text-bottom;margin-left:2px}
.lp-bars{margin-top:22px;padding-top:18px;border-top:1px solid rgba(168,188,200,.08);display:flex;flex-direction:column;gap:10px}
.lp-br{display:flex;align-items:center;gap:12px}.lp-bl{width:110px;text-align:right;font-size:11px;color:#4A6B78}
.lp-bt{flex:1;height:8px;background:rgba(168,188,200,.08);border-radius:4px;overflow:hidden}
.lp-bf{height:100%;border-radius:4px}
.lb1{background:linear-gradient(90deg,#2D5F6E,#3A7A8C);animation:barG1 1.8s cubic-bezier(.4,0,.2,1) 1.4s both}
.lb2{background:linear-gradient(90deg,#3D7A5F,#5A9A7B);animation:barG2 1.8s cubic-bezier(.4,0,.2,1) 1.6s both}
.lb3{background:linear-gradient(90deg,#2D5F6E,#3A7A8C);animation:barG3 1.8s cubic-bezier(.4,0,.2,1) 1.8s both}
.lb4{background:linear-gradient(90deg,#D4B94E,#E8D97A);animation:barG4 1.8s cubic-bezier(.4,0,.2,1) 2.0s both}
.lb5{background:linear-gradient(90deg,#6B8A99,#A8BCC8);animation:barG5 1.8s cubic-bezier(.4,0,.2,1) 2.2s both}
.lp-bp{width:44px;font-size:11px;color:#A8BCC8;font-family:'JetBrains Mono',monospace;text-align:right}
.lp-sts{position:relative;z-index:2;display:flex;justify-content:center;gap:48px;margin-top:56px;flex-wrap:wrap;animation:fadeUp .7s ease-out .8s both}
.lp-st{text-align:center}.lp-sn{font-size:32px;font-weight:700;color:#E8E6DD;font-family:'JetBrains Mono',monospace}
.lp-sn span{color:#D4B94E}.lp-sl{font-size:11px;color:#4A6B78;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px}
.lp-ft{background:#F5F4F0;padding:80px 40px;text-align:center}
.lp-fh{font-size:32px;font-weight:700;color:#1E2D33;margin-bottom:48px}
.lp-fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;max-width:1100px;margin:0 auto}
.lp-fc{background:#fff;border:1px solid #D1CFC4;border-radius:14px;padding:32px 24px;transition:all .35s;position:relative;overflow:hidden}
.lp-fc::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#2D5F6E,#D4B94E);transform:scaleX(0);transform-origin:left;transition:transform .35s}
.lp-fc:hover{transform:translateY(-6px);box-shadow:0 16px 48px rgba(45,95,110,.1)}.lp-fc:hover::after{transform:scaleX(1)}
.lp-fi{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;margin-bottom:18px;background:linear-gradient(135deg,#2D5F6E,#3A7A8C);box-shadow:0 4px 12px rgba(45,95,110,.2)}
.lp-fc h3{font-size:15px;font-weight:600;color:#1E2D33;margin-bottom:8px}.lp-fc p{font-size:13px;color:#6B8A99;line-height:1.65}
.lp-hw{background:#0C1418;padding:80px 40px;position:relative;overflow:hidden}
.lp-hw::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(168,188,200,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(168,188,200,.03) 1px,transparent 1px);background-size:52px 52px;pointer-events:none}
.lp-hwt{text-align:center;font-size:32px;font-weight:700;color:#E8E6DD;margin-bottom:52px;position:relative;z-index:1}
.lp-hws{display:flex;justify-content:center;gap:24px;max-width:960px;margin:0 auto;flex-wrap:wrap;position:relative;z-index:1}
.lp-stp{text-align:center;flex:1;min-width:240px;padding:32px 20px;background:rgba(22,36,42,.4);backdrop-filter:blur(12px);border:1px solid rgba(168,188,200,.08);border-radius:16px;transition:all .3s}
.lp-stp:hover{border-color:rgba(212,185,78,.2)}.lp-snm{width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#D4B94E,#E8D97A);color:#1E2D33;font-size:22px;font-weight:700;display:flex;align-items:center;justify-content:center;margin:0 auto 18px;box-shadow:0 4px 24px rgba(212,185,78,.25)}
.lp-stp h4{font-size:16px;font-weight:600;color:#E8E6DD;margin-bottom:8px}.lp-stp p{font-size:13px;color:#6B8A99;line-height:1.6}
.lp-tc{background:#F5F4F0;padding:44px;text-align:center}.lp-tl{font-size:11px;color:#6B8A99;text-transform:uppercase;letter-spacing:2px;font-weight:600}
.lp-tr{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:16px}
.lp-tp{background:#fff;border:1px solid #D1CFC4;border-radius:8px;padding:9px 20px;font-size:13px;font-weight:500;color:#3D5A66;transition:all .2s}.lp-tp:hover{border-color:#2D5F6E;color:#2D5F6E}
.lp-fo{background:#0C1418;padding:28px;text-align:center;font-size:12px;color:#3D5A66;border-top:1px solid rgba(168,188,200,.06)}
</style>
<div class="lp">
<div class="lp-hero"><div class="lp-grd"></div><div class="lp-o1"></div><div class="lp-o2"></div>
<div class="lp-bdg">ENTERPRISE TEXT ANALYTICS ENGINE</div>
<h1 class="lp-ttl">TextInsightMiner</h1>
<p class="lp-sub">Dig Deeper. Classify Smarter.</p>
<p class="lp-dsc">Transform unstructured conversations into actionable intelligence. Vectorized classification at 20K+ records/sec with Boolean DSL rules, proximity matching, and full audit trails.</p>
<div class="lp-mk"><div class="lp-wn"><div class="lp-wh"><div class="lp-dt lp-dr"></div><div class="lp-dt lp-dy"></div><div class="lp-dt lp-dg"></div><span class="lp-wt">engine.py — Vectorized DSL Engine</span></div>
<div class="lp-wb"><span class="cm"># Polars column-ops — Rust internals, zero Python loops</span><br><span class="ck">import</span> polars <span class="ck">as</span> pl<br><br><span class="ck">def</span> <span class="cf">classify_batch</span>(df, rules):<br>&nbsp;&nbsp;mask = text.<span class="cf">str.contains</span>(<span class="cs">"cancel subscription"</span>)<br>&nbsp;&nbsp;mask = mask & ~text.<span class="cf">str.contains</span>(<span class="cs">"policy"</span>)<br>&nbsp;&nbsp;<span class="ck">return</span> <span class="cf">best_match</span>(scores, <span class="ck">263</span> rules)<span class="lp-cur"></span>
<div class="lp-bars"><div class="lp-br"><span class="lp-bl">Cancellation</span><div class="lp-bt"><div class="lp-bf lb1"></div></div><span class="lp-bp">31.2%</span></div>
<div class="lp-br"><span class="lp-bl">Billing</span><div class="lp-bt"><div class="lp-bf lb2"></div></div><span class="lp-bp">25.6%</span></div>
<div class="lp-br"><span class="lp-bl">Technology</span><div class="lp-bt"><div class="lp-bf lb3"></div></div><span class="lp-bp">19.4%</span></div>
<div class="lp-br"><span class="lp-bl">Account</span><div class="lp-bt"><div class="lp-bf lb4"></div></div><span class="lp-bp">12.8%</span></div>
<div class="lp-br"><span class="lp-bl">Products</span><div class="lp-bt"><div class="lp-bf lb5"></div></div><span class="lp-bp">11.0%</span></div></div></div></div></div>
<div class="lp-sts"><div class="lp-st"><div class="lp-sn">20<span>K+</span></div><div class="lp-sl">Records / Second</div></div>
<div class="lp-st"><div class="lp-sn">10</div><div class="lp-sl">Industry Domains</div></div>
<div class="lp-st"><div class="lp-sn">4</div><div class="lp-sl">Hierarchy Levels</div></div>
<div class="lp-st"><div class="lp-sn">0</div><div class="lp-sl">Python Loops</div></div></div></div>
<div class="lp-ft"><h2 class="lp-fh">Built for Enterprise Scale</h2>
<div class="lp-fg">
<div class="lp-fc"><div class="lp-fi"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div><h3>Vectorized Engine</h3><p>20K+ records/sec via Polars. Rust internals, zero Python loops.</p></div>
<div class="lp-fc"><div class="lp-fi"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg></div><h3>Boolean DSL Rules</h3><p>AND, OR, NOT, NEAR with proximity. Unlimited terms. Phrase support.</p></div>
<div class="lp-fc"><div class="lp-fi"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div><h3>PII Redaction</h3><p>8 pattern types. Emails, phones, SSNs, cards, IPs, addresses.</p></div>
<div class="lp-fc"><div class="lp-fi"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></div><h3>4-Level Hierarchy</h3><p>Decomposition trees, sunburst charts, independent level tables.</p></div>
</div></div>
<div class="lp-hw"><div class="lp-hwt">Three Steps to Intelligence</div><div class="lp-hws">
<div class="lp-stp"><div class="lp-snm">1</div><h4>Upload</h4><p>CSV, Excel, Parquet, JSON. Auto-optimized.</p></div>
<div class="lp-stp"><div class="lp-snm">2</div><h4>Configure</h4><p>Select domain or build custom DSL rules.</p></div>
<div class="lp-stp"><div class="lp-snm">3</div><h4>Insights</h4><p>Classification in seconds. Reports & analytics.</p></div></div></div>
<div class="lp-tc"><p class="lp-tl">Powered By</p><div class="lp-tr"><div class="lp-tp">Polars</div><div class="lp-tp">DuckDB</div><div class="lp-tp">Streamlit</div><div class="lp-tp">Plotly</div><div class="lp-tp">ECharts</div><div class="lp-tp">NumPy</div></div></div>
<div class="lp-fo">TextInsightMiner v9.0 — Dig Deeper. Classify Smarter.</div>
</div>
"""

def render_landing():
    st.markdown(LANDING_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _,cc,_ = st.columns([1,2,1])
    with cc:
        if st.button("Launch Application", type="primary", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP — Collapsible sections, sidebar config, cached computations
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_results(out):
    """Pre-compute and cache all expensive operations once per dataset."""
    sig = id(out)
    if st.session_state.get('_cache_sig') == sig:
        return
    st.session_state._cache_sig = sig
    st.session_state.tree_data = build_tree_data(out)
    st.session_state.sunburst_fig = build_sunburst(out)
    st.session_state.level_tables = {
        'l1': build_level_table(out, 'Category'),
        'l2': build_level_table(out, 'Subcategory', 'Category'),
        'l3': build_level_table(out, 'L3', 'Subcategory'),
        'l4': build_level_table(out, 'L4', 'L3'),
    }
    total = len(out)
    cc = out['Category'].value_counts().reset_index()
    cc.columns = ['Category','Count']; cc = cc[cc['Category'] != 'Uncategorized']
    st.session_state.rpt_cat = cc
    sc = out[out['Subcategory'] != 'NA']['Subcategory'].value_counts().reset_index()
    sc.columns = ['Subcategory','Count']
    st.session_state.rpt_sub = sc
    cross = out[out['Category'] != 'Uncategorized'].groupby(['Category','Subcategory']).size().reset_index(name='Count')
    st.session_state.rpt_heat = cross[cross['Subcategory'] != 'NA']

def main_app():
    st.markdown(CSS, unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:10px 0 12px">
            {IC.icon(IC.PICKAXE,'#2D5F6E',26)}
            <div><span style="font-size:17px;font-weight:700;color:#1E2D33">TextInsightMiner</span><br>
            <span style="font-size:11px;color:#6B8A99;font-style:italic">Dig Deeper. Classify Smarter.</span></div>
            <span class="badge b-info" style="margin-left:auto;font-size:10px">v10</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        with st.expander("Configuration", expanded=True):
            if 'dl' not in st.session_state:
                st.session_state.dl = DomainLoader(); st.session_state.dl.auto_load()
            inds = st.session_state.dl.get_industries()
            st.markdown(f'{IC.icon(IC.GLOBE,"#2D5F6E",15)} **Industry Domain**', unsafe_allow_html=True)
            sel = st.selectbox("Industry", ["None"]+sorted(inds), label_visibility="collapsed", key="sb_ind")
            st.session_state.sel_ind = sel if sel != "None" else None
            if st.session_state.sel_ind:
                st.markdown(f'<span class="badge b-ok">{IC.icon(IC.CHECK,"#065f46",13)}{sel}</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f'{IC.icon(IC.SHIELD,"#2D5F6E",15)} **PII Redaction**', unsafe_allow_html=True)
            en_pii = st.checkbox("Enable PII Redaction", value=True, key="sb_pii")
            rd_mode = st.selectbox("Mode", ['hash','mask','token','remove'], key="sb_mode")
            st.markdown("---")
            st.markdown(f'{IC.icon(IC.DOWNLOAD,"#2D5F6E",15)} **Output**', unsafe_allow_html=True)
            out_fmt = st.selectbox("Format", ['csv','xlsx','parquet'], label_visibility="collapsed", key="sb_fmt")

        with st.expander("Quick Rule Upload"):
            rul = st.file_uploader("JSON Rules", type=['json'], key='sb_rul')
            if rul and rul.name != st.session_state.get('_sb_rul_loaded'):
                try:
                    data = json.load(rul)
                    if isinstance(data, list):
                        if 'custom_rules' not in st.session_state: st.session_state.custom_rules = []
                        for d in data: st.session_state.custom_rules.append(DSLRule.from_dict(d))
                        st.session_state['_sb_rul_loaded'] = rul.name
                        st.success(f"Loaded {len(data)} rules"); st.rerun()
                except Exception as e: st.error(str(e))

        st.markdown("---")
        # Navigation with icons rendered as HTML above the radio
        st.markdown("""<style>
        .nav-item{display:flex;align-items:center;gap:8px;padding:6px 10px;margin:2px 0;border-radius:6px;font-size:13px;color:#3D5A66;cursor:default}
        div[data-testid="stRadio"] label{font-weight:500!important;font-size:13px!important}
        div[data-testid="stRadio"] > div{gap:2px!important}
        </style>""", unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:#6B8A99;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Navigation</div>', unsafe_allow_html=True)
        _nav_pages = [
            "Upload & Analyse",
            "Reports & Insights",
            "Narrative Intelligence",
            "Rule Builder",
            "Concordance",
            "Audit Trail",
            "Rule Performance",
            "Projects",
        ]
        _nav_target = st.session_state.pop('_nav_target', None)
        if _nav_target and _nav_target in _nav_pages:
            _default_idx = _nav_pages.index(_nav_target)
        elif 'nav_radio' in st.session_state and st.session_state.nav_radio in _nav_pages:
            _default_idx = _nav_pages.index(st.session_state.nav_radio)
        else:
            _default_idx = 0
        page = st.radio("nav", _nav_pages, label_visibility="collapsed", key="nav_radio", index=_default_idx)

        # Icon legend below radio
        icon_map = [
            (IC.UPLOAD,"Upload & Analyse"),(IC.REPORT,"Reports & Insights"),(IC.TRENDING,"Narrative Intelligence"),(IC.TOOL,"Rule Builder"),
            (IC.SEARCH,"Concordance"),(IC.EYE,"Audit Trail"),(IC.ACTIVITY,"Rule Performance"),(IC.GLOBE,"Projects")]
        selected_icon = next((svg for svg,name in icon_map if name==page), IC.UPLOAD)
        st.markdown(f'<div style="margin-top:8px;padding:8px 10px;background:#D6E8EE;border-radius:8px;font-size:12px;color:#2D5F6E;display:flex;align-items:center;gap:6px">{IC.icon(selected_icon,"#2D5F6E",16)}<strong>{page}</strong></div>', unsafe_allow_html=True)

    # ── INIT ──
    if 'rh' not in st.session_state: st.session_state.rh = RunHistory()
    if 'custom_rules' not in st.session_state: st.session_state.custom_rules = []
    ind_rules = st.session_state.dl.get_rules(st.session_state.sel_ind) if st.session_state.get('sel_ind') else []
    cust_rules = st.session_state.custom_rules
    all_rules = ind_rules + cust_rules

    # ── HEADER ──
    st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;margin-bottom:2px">
        {IC.icon(IC.PICKAXE,'#2D5F6E',32)}
        <div><h1 style="margin:0;font-size:26px;line-height:1.2;color:#1E2D33">TextInsightMiner</h1>
        <p style="margin:0;color:#6B8A99;font-size:12px">Dig Deeper. Classify Smarter.</p></div>
    </div>""", unsafe_allow_html=True)

    if all_rules:
        st.markdown(f'<span class="badge b-ok">{IC.icon(IC.CHECK,"#065f46",13)}{len(all_rules)} rules ({len(ind_rules)} domain + {len(cust_rules)} custom)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge b-warn">{IC.icon(IC.ALERT,"#7A6620",13)}No rules loaded</span>', unsafe_allow_html=True)

    if len(all_rules) > 1:
        _rules_sig=hashlib.md5(json.dumps([r.to_dict() for r in all_rules],default=str).encode()).hexdigest()
        if st.session_state.get('_eng_sig')!=_rules_sig:
            st.session_state._eng_cached=DSLEngine(all_rules)
            st.session_state._conf_cached=st.session_state._eng_cached.detect_conflicts()
            st.session_state._eng_sig=_rules_sig
        conf=st.session_state._conf_cached
        if conf:
            with st.expander(f"Rule Conflicts ({len(conf)})"):
                for c in conf:
                    st.markdown(f'<div class="conflict-row">{IC.icon(IC.ALERT_OCT,"var(--err)",16)}<strong>"{c["term"]}"</strong> -> <span class="tag tag-r">{c["categories"]}</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # S1: UPLOAD & ANALYSE
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Upload & Analyse":
        shdr(IC.UPLOAD, "Upload & Analyse")

        # Show existing results status if available
        if 'out' in st.session_state and st.session_state.get('_uploaded_name'):
            _uname=st.session_state._uploaded_name
            _urows=len(st.session_state.out)
            _upt=st.session_state.get('pt',0)
            st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin-bottom:12px;border-left:3px solid var(--success)">'
                f'{IC.icon(IC.CHECK,"var(--success)",16)}<b style="color:var(--text)">{_uname}</b> — '
                f'<span style="color:var(--muted);font-size:13px">{_urows:,} records classified in {_upt:.1f}s. Results available in Reports.</span></div>',unsafe_allow_html=True)

        df_file = st.file_uploader("Upload data file", type=SUPPORTED_FORMATS, label_visibility="collapsed")

        # Cache the read data to avoid re-parsing on every rerun
        if df_file:
            _file_key=f"{df_file.name}_{df_file.size}"
            if st.session_state.get('_file_cache_key')!=_file_key:
                st.session_state._file_cache_data=FH.read(df_file)
                st.session_state._file_cache_key=_file_key
                st.session_state._uploaded_name=df_file.name
            data_df=st.session_state._file_cache_data
        else:
            data_df=None

        if data_df is not None and all_rules:
            cols = data_df.columns
            c1, c2 = st.columns(2)
            lid = [c for c in cols if any(k in c.lower() for k in ['id','conversation','ticket'])]
            with c1: ic = st.selectbox("ID Column", cols, index=cols.index(lid[0]) if lid else 0)
            with c2: tc = st.selectbox("Text Column", [c for c in cols if c != ic])
            with st.expander(f"Preview ({len(data_df):,} records)", expanded=False):
                st.dataframe(data_df.select([ic,tc]).head(10).to_pandas(), hide_index=True, use_container_width=True)
            if len(data_df) > 10000:
                opt = st.radio("Scope", [f"First 10K", f"All {len(data_df):,}"], horizontal=True)
                if "First" in opt: data_df = data_df.head(10000)
            st.markdown("---")
            _, rc, _ = st.columns([1,2,1])
            with rc:
                run_btn = st.button(f"Run Analysis ({len(data_df):,} x {len(all_rules)} rules)", type="primary", use_container_width=True)
            if run_btn:
                # Use cached engine if available, else build fresh
                if st.session_state.get('_eng_cached'):
                    eng=st.session_state._eng_cached
                else:
                    eng = DSLEngine(all_rules)
                pipe = Pipeline(eng, en_pii)
                prog = st.progress(0, text="Starting vectorized engine...")
                t0 = datetime.now()
                def upd(d, t):
                    p = d/t; e = (datetime.now()-t0).total_seconds(); s = d/e if e>0 else 0
                    prog.progress(p, text=f"{d:,}/{t:,} ({p*100:.0f}%) | {s:.0f} rec/s")
                res = pipe.process(data_df, tc, ic, rd_mode, upd)
                pt = (datetime.now()-t0).total_seconds()
                prog.progress(1.0, text=f"Done: {len(res):,} in {pt:.1f}s ({len(res)/pt:.0f} rec/s)")
                out_df = res.to_pandas()
                st.session_state.out = out_df; st.session_state.pt = pt; st.session_state.of = out_fmt
                st.session_state._uploaded_name=df_file.name if df_file else "data"
                st.session_state.rh.record(out_df, pt, len(all_rules), pipe.pii_n)
                st.session_state.pop('_cache_sig', None)
                st.session_state.pop('_export_cache', None)  # Clear export cache
                st.toast(f"Complete: {len(out_df):,} records in {pt:.1f}s")
        elif df_file and not all_rules:
            st.warning("Select an industry domain or upload rules first.")

    # ═══════════════════════════════════════════════════════════════════════════
    # S2: REPORTS & INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Reports & Insights":
        shdr(IC.REPORT, "Reports & Insights")
        if 'out' not in st.session_state:
            st.info("Run analysis to see reports.")
        else:
            out = st.session_state.out; pt = st.session_state.get('pt', 0)
            _cache_results(out)
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            with m1: st.markdown(mcard("Records", f"{len(out):,}"), unsafe_allow_html=True)
            with m2: st.markdown(mcard("Time", f"{pt:.1f}s", "var(--slate)"), unsafe_allow_html=True)
            with m3: st.markdown(mcard("Speed", f"{len(out)/pt:.0f}/s" if pt>0 else "0", "var(--steel)"), unsafe_allow_html=True)
            with m4: st.markdown(mcard("L1", str(out['Category'].nunique()), "var(--success)"), unsafe_allow_html=True)
            with m5: st.markdown(mcard("L2", str(out['Subcategory'].nunique()), "var(--gold)"), unsafe_allow_html=True)
            with m6:
                cp = (1-(out['Category']=='Uncategorized').mean())*100
                st.markdown(mcard("Categorized", f"{cp:.1f}%", "var(--success)" if cp>90 else "var(--gold)"), unsafe_allow_html=True)
            drift = st.session_state.rh.get_drift()
            if drift:
                d = drift['d_pct']; a = "+" if d>=0 else ""; cl = "var(--success)" if d>=0 else "var(--err)"
                st.markdown(f'<div style="font-size:12px;color:{cl};margin:6px 0">{IC.icon(IC.TRENDING,cl,14)}Drift: <strong>{a}{d:.1f}%</strong></div>', unsafe_allow_html=True)
            st.markdown("---")
            rpt = st.tabs(["Data","Distribution","Top Drivers","Heatmap","Visualizations","Uncategorized","Summary","Custom Report","Metrics Explorer","Category Trends","Top Movers"])
            with rpt[0]:
                dc = [c for c in ['Conversation_ID','Original_Text','Category','Subcategory','L3','L4','confidence','matched_rule'] if c in out.columns]
                st.dataframe(out[dc], hide_index=True, use_container_width=True, height=500)
            with rpt[1]:
                dt = st.tabs(["L1","L2","L3","L4"])
                with dt[0]: st.markdown(st.session_state.level_tables['l1'], unsafe_allow_html=True)
                with dt[1]: st.markdown(st.session_state.level_tables['l2'], unsafe_allow_html=True)
                with dt[2]: st.markdown(st.session_state.level_tables['l3'], unsafe_allow_html=True)
                with dt[3]: st.markdown(st.session_state.level_tables['l4'], unsafe_allow_html=True)
            with rpt[2]:
                top_n = st.slider("Top N", 5, 30, 15, key="rpt_topn")
                cc = st.session_state.rpt_cat.head(top_n)
                if not cc.empty:
                    fig = px.bar(cc, x='Count', y='Category', orientation='h', color='Count', color_continuous_scale=['#A8BCC8','#2D5F6E'], height=max(300,top_n*28))
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, font_family="DM Sans", margin=dict(l=0,r=20,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True, key="main_rpt_cat")
                sc = st.session_state.rpt_sub.head(top_n)
                if not sc.empty:
                    fig2 = px.bar(sc, x='Count', y='Subcategory', orientation='h', color='Count', color_continuous_scale=['#E8D97A','#D4B94E'], height=max(300,top_n*28))
                    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, font_family="DM Sans", margin=dict(l=0,r=20,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True, key="main_rpt_sub")
            with rpt[3]:
                st.markdown("**Cross-Level Concentration Heatmap**")
                hm_row=st.selectbox("Rows",["Category","Subcategory","L3","L4"],index=0,key="hm_row")
                hm_col=st.selectbox("Columns",["Category","Subcategory","L3","L4"],index=1,key="hm_col")
                if hm_row==hm_col:
                    st.warning("Select different levels for rows and columns.")
                else:
                    hm_df=out[(out[hm_row]!='NA')&(out[hm_row]!='Uncategorized')&(out[hm_col]!='NA')&(out[hm_col]!='Uncategorized')]
                    if not hm_df.empty:
                        hm_cross=hm_df.groupby([hm_row,hm_col]).size().reset_index(name='Count')
                        hm_pivot=hm_cross.pivot_table(index=hm_row,columns=hm_col,values='Count',fill_value=0)
                        # Keep top 20 columns by volume
                        hm_top=hm_cross.groupby(hm_col)['Count'].sum().nlargest(20).index.tolist()
                        hm_pivot=hm_pivot[[c for c in hm_top if c in hm_pivot.columns]]
                        fig=px.imshow(hm_pivot,color_continuous_scale=['#F5F4F0','#2D5F6E'],
                            labels=dict(x=hm_col,y=hm_row,color="Count"),height=max(400,len(hm_pivot)*35))
                        fig.update_layout(font_family="DM Sans",margin=dict(l=0,r=0,t=30,b=0),paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig,use_container_width=True,key="main_rpt_heat")
                    else:
                        st.info("No data for selected level combination.")
            with rpt[4]:
                try:
                    from streamlit_echarts import st_echarts; has_ec = True
                except: has_ec = False
                vt = st.tabs(["Decomposition Tree","Sunburst"])
                with vt[0]:
                    if has_ec: st_echarts(get_tree_option(st.session_state.tree_data), height="700px", key="main_tree")
                    else: st.info("Install streamlit-echarts for tree chart.")
                with vt[1]:
                    fig = st.session_state.sunburst_fig
                    if fig: st.plotly_chart(fig, use_container_width=True, key="main_rpt_sun")
            with rpt[5]:
                unc = out[out['Category']=='Uncategorized']
                if len(unc) > 0:
                    st.markdown(f'<span class="badge b-warn">{len(unc):,} uncategorized ({len(unc)/len(out)*100:.1f}%)</span>', unsafe_allow_html=True)
                    stop = Concordance.STOP
                    words = [w.strip('.,!?;:').lower() for t in unc['Original_Text'].dropna().tolist() for w in str(t).split() if w.strip('.,!?;:').lower() not in stop and len(w)>2]

                    # ── Action bar ──
                    act1,act2=st.columns(2)
                    with act1:
                        if st.button("Explore in Concordance",key="unc_to_conc",type="primary",use_container_width=True):
                            st.session_state._conc_filter_uncat=True
                            st.session_state._nav_target="Concordance"
                            st.rerun()
                    with act2:
                        if st.button("Open Rule Builder",key="unc_to_rb",use_container_width=True):
                            st.session_state._nav_target="Rule Builder"
                            st.rerun()

                    if words:
                        uni=Counter(words).most_common(20)
                        uni_df = pd.DataFrame(uni, columns=['Word','Count'])
                        fig = px.bar(uni_df, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale=['#D1CFC4','#A04040'], height=500)
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, font_family="DM Sans", margin=dict(l=0,r=20,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, title="Top words — click below to build rules")
                        st.plotly_chart(fig, use_container_width=True, key="main_rpt_unc")

                        # ── Bigrams ──
                        bigrams=[]
                        for t in unc['Original_Text'].dropna().tolist():
                            ws=[w.strip('.,!?;:').lower() for w in str(t).split() if w.strip('.,!?;:').lower() not in stop and len(w)>2]
                            bigrams.extend(zip(ws,ws[1:]))
                        if bigrams:
                            bi=Counter(bigrams).most_common(10)
                            bi_df=pd.DataFrame([(' '.join(b),c) for b,c in bi],columns=['Bigram','Count'])
                            st.markdown("**Top Bigrams**")
                            st.dataframe(bi_df,hide_index=True,use_container_width=True)

                        # ── Clickable word pills → Concordance or Rule Builder ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.ZAPPER,"#2D5F6E",18)}Quick Actions — Click a Word</div>',unsafe_allow_html=True)
                        top_words=[w for w,c in uni[:15]]
                        wc1,wc2=st.columns(2)
                        with wc1:
                            sel_word=st.selectbox("Select a word to act on",["(pick a word)"]+top_words,key="unc_sel_word")
                        with wc2:
                            sel_action=st.selectbox("Action",["Search in Concordance","Create Rule"],key="unc_sel_action")
                        if sel_word!="(pick a word)":
                            if st.button(f"Go → {sel_action}",key="unc_action_go",type="primary"):
                                if sel_action=="Search in Concordance":
                                    st.session_state._conc_filter_uncat=True
                                    st.session_state._conc_prefill_kw=sel_word
                                    st.session_state._nav_target="Concordance"
                                    st.rerun()
                                else:
                                    st.session_state._rb_prefill_term=sel_word
                                    st.session_state._nav_target="Rule Builder"
                                    st.rerun()

                        # ── Inline mini rule builder ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.TOOL,"#2D5F6E",18)}Quick Rule Builder</div>',unsafe_allow_html=True)
                        st.markdown('<p style="font-size:12px;color:var(--muted)">Create a rule directly from uncategorized insights — no need to switch pages.</p>',unsafe_allow_html=True)
                        with st.form("unc_mini_rb",clear_on_submit=True):
                            mr1,mr2=st.columns([3,1])
                            with mr1: mini_cat=st.text_input("Category Name *",placeholder="e.g. Payment Issue",key="unc_mini_cat")
                            with mr2: mini_level=st.selectbox("Level",["L1","L2","L3","L4"],key="unc_mini_level")
                            mt1,mt2,mt3=st.columns([2,1,2])
                            with mt1: mini_t1=st.text_input("Term 1 *",placeholder="e.g. billing",key="unc_mini_t1")
                            with mt2: mini_op=st.selectbox("Operator",["AND","OR","NOT","NEAR","NOT LIKE"],key="unc_mini_op")
                            with mt3: mini_t2=st.text_input("Term 2 (optional)",placeholder="e.g. overcharge",key="unc_mini_t2")
                            if mini_op=="NEAR":
                                mini_prox=st.slider("NEAR Proximity",1,5,3,key="unc_mini_prox")
                            else:
                                mini_prox=3
                            if st.form_submit_button("Save Rule",type="primary") and mini_t1.strip() and mini_cat.strip():
                                terms=[mini_t1.strip()]
                                ops=[]
                                if mini_t2.strip():
                                    terms.append(mini_t2.strip())
                                    ops.append(mini_op)
                                l1=l2=l3=l4="NA"
                                if mini_level=="L1": l1=mini_cat.strip()
                                elif mini_level=="L2": l2=mini_cat.strip()
                                elif mini_level=="L3": l3=mini_cat.strip()
                                elif mini_level=="L4": l4=mini_cat.strip()
                                rid=f"USR-{len(st.session_state.custom_rules)+1:04d}"
                                st.session_state.custom_rules.append(DSLRule(rule_id=rid,terms=terms,operators=ops,
                                    proximity=mini_prox if mini_op=="NEAR" and mini_t2.strip() else None,
                                    l1=l1,l2=l2,l3=l3,l4=l4,source="manual"))
                                st.success(f"Rule {rid} created: {' '.join(terms)}"); st.rerun()

                else: st.success("Full coverage — no uncategorized records.")
            with rpt[6]:
                total = len(out); unc_n = int((out['Category']=='Uncategorized').sum()); cat_n = total-unc_n
                sm1,sm2,sm3 = st.columns(3)
                with sm1: st.markdown(mcard("Total", f"{total:,}"), unsafe_allow_html=True)
                with sm2: st.markdown(mcard("Categorized", f"{cat_n:,} ({cat_n/total*100:.1f}%)", "var(--success)"), unsafe_allow_html=True)
                with sm3: st.markdown(mcard("Uncategorized", f"{unc_n:,}", "var(--err)" if unc_n/total>.1 else "var(--gold)"), unsafe_allow_html=True)
                st.markdown("---")
                top5 = out['Category'].value_counts().head(6).reset_index(); top5.columns = ['Category','Count']
                top5 = top5[top5['Category'] != 'Uncategorized'].head(5); top5['%'] = (top5['Count']/total*100).round(1)
                st.markdown("**Top 5 Categories**")
                st.dataframe(top5, hide_index=True, use_container_width=True)
                if 'confidence' in out.columns:
                    cd = pd.cut(out['confidence'], bins=[0,0.3,0.7,1.01], labels=['Low','Medium','High'], include_lowest=True).value_counts().reset_index()
                    cd.columns = ['Band','Count']; st.markdown("**Confidence Distribution**"); st.dataframe(cd, hide_index=True, use_container_width=True)
            with rpt[7]:
                st.markdown(f'<div class="sh">{IC.icon(IC.REPORT,"#2D5F6E",18)}Custom Report Builder</div>',unsafe_allow_html=True)
                st.markdown('<p style="font-size:12px;color:var(--muted)">Select categories to generate a focused report. Export as a multi-sheet Excel workbook.</p>',unsafe_allow_html=True)

                # ── Category multi-select ──
                all_cats=sorted([c for c in out['Category'].unique().tolist() if c not in ('Uncategorized','NA')])
                sel_cats=st.multiselect("Select Categories",all_cats,default=all_cats[:5] if len(all_cats)>5 else all_cats,key="cr_cats")
                include_uncat=st.checkbox("Include Uncategorized",value=False,key="cr_inc_unc")

                if not sel_cats and not include_uncat:
                    st.info("Select at least one category to generate a report.")
                else:
                    # ── Filter dataset ──
                    mask=out['Category'].isin(sel_cats)
                    if include_uncat: mask=mask|(out['Category']=='Uncategorized')
                    fdf=out[mask].copy()
                    ftotal=len(fdf)

                    if ftotal==0:
                        st.warning("No records match the selected categories.")
                    else:
                        # ── Summary cards ──
                        st.markdown("---")
                        cr_m1,cr_m2,cr_m3,cr_m4=st.columns(4)
                        with cr_m1: st.markdown(mcard("Selected Records",f"{ftotal:,}"),unsafe_allow_html=True)
                        with cr_m2: st.markdown(mcard("% of Total",f"{ftotal/len(out)*100:.1f}%","var(--slate)"),unsafe_allow_html=True)
                        with cr_m3: st.markdown(mcard("Categories",str(fdf['Category'].nunique()),"var(--success)"),unsafe_allow_html=True)
                        with cr_m4:
                            fsub=fdf[fdf['Subcategory']!='NA']['Subcategory'].nunique()
                            st.markdown(mcard("Subcategories",str(fsub),"var(--gold)"),unsafe_allow_html=True)

                        # ── Distribution tables ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.TABLE,"#2D5F6E",18)}Distribution</div>',unsafe_allow_html=True)
                        cr_dt=st.tabs(["L1","L2","L3","L4"])
                        with cr_dt[0]: st.markdown(build_level_table(fdf,'Category'),unsafe_allow_html=True)
                        with cr_dt[1]: st.markdown(build_level_table(fdf,'Subcategory','Category'),unsafe_allow_html=True)
                        with cr_dt[2]: st.markdown(build_level_table(fdf,'L3','Subcategory'),unsafe_allow_html=True)
                        with cr_dt[3]: st.markdown(build_level_table(fdf,'L4','L3'),unsafe_allow_html=True)

                        # ── Top Drivers ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.BAR,"#2D5F6E",18)}Top Drivers</div>',unsafe_allow_html=True)
                        cr_topn=st.slider("Show top N",5,30,15,key="cr_topn")
                        cr_cat_counts=fdf['Category'].value_counts().head(cr_topn).reset_index()
                        cr_cat_counts.columns=['Category','Count']
                        cr_cat_counts=cr_cat_counts[cr_cat_counts['Category']!='Uncategorized']
                        if not cr_cat_counts.empty:
                            cr_fig1=px.bar(cr_cat_counts,x='Count',y='Category',orientation='h',color='Count',
                                color_continuous_scale=['#A8BCC8','#2D5F6E'],height=max(300,cr_topn*28))
                            cr_fig1.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
                                font_family="DM Sans",margin=dict(l=0,r=20,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False)
                            st.plotly_chart(cr_fig1,use_container_width=True,key="cr_fig_cat")

                        cr_sub_counts=fdf[fdf['Subcategory']!='NA']['Subcategory'].value_counts().head(cr_topn).reset_index()
                        cr_sub_counts.columns=['Subcategory','Count']
                        if not cr_sub_counts.empty:
                            cr_fig2=px.bar(cr_sub_counts,x='Count',y='Subcategory',orientation='h',color='Count',
                                color_continuous_scale=['#E8D97A','#D4B94E'],height=max(300,cr_topn*28))
                            cr_fig2.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
                                font_family="DM Sans",margin=dict(l=0,r=20,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False)
                            st.plotly_chart(cr_fig2,use_container_width=True,key="cr_fig_sub")

                        # ── Heatmap ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.LAYERS,"#2D5F6E",18)}Cross-Level Heatmap</div>',unsafe_allow_html=True)
                        cr_hm1,cr_hm2=st.columns(2)
                        with cr_hm1: cr_hm_row=st.selectbox("Rows",["Category","Subcategory","L3","L4"],index=0,key="cr_hm_row")
                        with cr_hm2: cr_hm_col=st.selectbox("Columns",["Category","Subcategory","L3","L4"],index=1,key="cr_hm_col")
                        if cr_hm_row==cr_hm_col:
                            st.warning("Select different levels for rows and columns.")
                        else:
                            cr_hm_df=fdf[(fdf[cr_hm_row]!='NA')&(fdf[cr_hm_row]!='Uncategorized')&(fdf[cr_hm_col]!='NA')&(fdf[cr_hm_col]!='Uncategorized')]
                            if not cr_hm_df.empty:
                                cr_hm_cross=cr_hm_df.groupby([cr_hm_row,cr_hm_col]).size().reset_index(name='Count')
                                cr_hm_pivot=cr_hm_cross.pivot_table(index=cr_hm_row,columns=cr_hm_col,values='Count',fill_value=0)
                                cr_hm_top=cr_hm_cross.groupby(cr_hm_col)['Count'].sum().nlargest(20).index.tolist()
                                cr_hm_pivot=cr_hm_pivot[[c for c in cr_hm_top if c in cr_hm_pivot.columns]]
                                cr_fig3=px.imshow(cr_hm_pivot,color_continuous_scale=['#F5F4F0','#2D5F6E'],
                                    labels=dict(x=cr_hm_col,y=cr_hm_row,color="Count"),height=max(400,len(cr_hm_pivot)*35))
                                cr_fig3.update_layout(font_family="DM Sans",margin=dict(l=0,r=0,t=30,b=0),paper_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(cr_fig3,use_container_width=True,key="cr_fig_heat")
                            else:
                                st.info("No data for selected level combination.")

                        # ── Sunburst ──
                        st.markdown(f'<div class="sh">{IC.icon(IC.PIE,"#2D5F6E",18)}Sunburst</div>',unsafe_allow_html=True)
                        cr_sun=build_sunburst(fdf)
                        if cr_sun: st.plotly_chart(cr_sun,use_container_width=True,key="cr_fig_sun")
                        else: st.info("Not enough data for sunburst.")

                        # ── Export as multi-sheet Excel ──
                        st.markdown("---")
                        st.markdown(f'<div class="sh">{IC.icon(IC.DOWNLOAD,"#2D5F6E",18)}Export Custom Report</div>',unsafe_allow_html=True)
                        if st.button("Generate Excel Report",key="cr_export",type="primary",use_container_width=True):
                            with st.spinner("Building Excel workbook..."):
                                xbuf=io.BytesIO()
                                with pd.ExcelWriter(xbuf,engine='openpyxl') as writer:
                                    # Sheet 1: Filtered raw data
                                    dc=[c for c in ['Conversation_ID','Original_Text','Category','Subcategory','L3','L4','confidence','matched_rule'] if c in fdf.columns]
                                    fdf[dc].to_excel(writer,sheet_name='Data',index=False)

                                    # Sheet 2: L1 distribution
                                    l1_agg=fdf['Category'].value_counts().reset_index()
                                    l1_agg.columns=['Category','Count']
                                    l1_agg['%']=((l1_agg['Count']/ftotal)*100).round(1)
                                    l1_agg.to_excel(writer,sheet_name='L1 Distribution',index=False)

                                    # Sheet 3: L2 distribution
                                    l2_agg=fdf.groupby(['Category','Subcategory']).size().reset_index(name='Count')
                                    l2_agg=l2_agg[l2_agg['Subcategory']!='NA'].sort_values('Count',ascending=False)
                                    l2_agg['%']=((l2_agg['Count']/ftotal)*100).round(1)
                                    l2_agg.to_excel(writer,sheet_name='L2 Distribution',index=False)

                                    # Sheet 4: L3 distribution
                                    l3_agg=fdf.groupby(['Category','Subcategory','L3']).size().reset_index(name='Count')
                                    l3_agg=l3_agg[l3_agg['L3']!='NA'].sort_values('Count',ascending=False)
                                    l3_agg['%']=((l3_agg['Count']/ftotal)*100).round(1)
                                    l3_agg.to_excel(writer,sheet_name='L3 Distribution',index=False)

                                    # Sheet 5: L4 distribution
                                    l4_agg=fdf.groupby(['Category','Subcategory','L3','L4']).size().reset_index(name='Count')
                                    l4_agg=l4_agg[l4_agg['L4']!='NA'].sort_values('Count',ascending=False)
                                    l4_agg['%']=((l4_agg['Count']/ftotal)*100).round(1)
                                    l4_agg.to_excel(writer,sheet_name='L4 Distribution',index=False)

                                    # Sheet 6: Cross-tab (heatmap data)
                                    xl_hm_df=fdf[(fdf[cr_hm_row]!='NA')&(fdf[cr_hm_row]!='Uncategorized')&(fdf[cr_hm_col]!='NA')&(fdf[cr_hm_col]!='Uncategorized')]
                                    if not xl_hm_df.empty:
                                        xl_hm_cross=xl_hm_df.groupby([cr_hm_row,cr_hm_col]).size().reset_index(name='Count')
                                        xl_hm_cross.to_excel(writer,sheet_name=f'{cr_hm_row} x {cr_hm_col}',index=False)

                                    # Sheet 7: Summary
                                    summary_data={
                                        'Metric':['Total Records (filtered)','Total Records (all)','% of Total',
                                                  'Categories Selected','Unique Subcategories','Unique L3','Unique L4',
                                                  'Avg Confidence','Report Generated'],
                                        'Value':[ftotal,len(out),f"{ftotal/len(out)*100:.1f}%",
                                                 fdf['Category'].nunique(),
                                                 fdf[fdf['Subcategory']!='NA']['Subcategory'].nunique(),
                                                 fdf[fdf['L3']!='NA']['L3'].nunique(),
                                                 fdf[fdf['L4']!='NA']['L4'].nunique(),
                                                 f"{fdf['confidence'].mean():.2f}" if 'confidence' in fdf.columns else 'N/A',
                                                 datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                                    }
                                    pd.DataFrame(summary_data).to_excel(writer,sheet_name='Summary',index=False)

                                xbuf.seek(0)
                                cats_label='_'.join(sel_cats[:3])
                                if len(sel_cats)>3: cats_label+=f"_+{len(sel_cats)-3}more"
                                fname=f"custom_report_{cats_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                                st.download_button("Download Excel Report",xbuf.getvalue(),fname,
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,key="cr_dl")
            with rpt[8]:
                st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#2D5F6E",18)}Metrics Explorer</div>',unsafe_allow_html=True)
                st.markdown('<p style="font-size:12px;color:var(--muted)">Analyse any numeric field in your data — AHT, Sentiment, CSAT, Duration, or any score — grouped by classification level.</p>',unsafe_allow_html=True)

                # ── Let user select numeric and date columns ──
                _system_cols={'Category','Subcategory','L3','L4','confidence','matched_rule','match_score','match_detail','l1','l2','l3','l4','Conversation_ID','Original_Text'}
                _all_cols=[c for c in out.columns if c not in _system_cols]

                # Pre-detect likely numeric columns (auto + object that can be coerced)
                _likely_num=[]
                for c in _all_cols:
                    if pd.api.types.is_numeric_dtype(out[c]):
                        _likely_num.append(c)
                    elif out[c].dtype=='object':
                        coerced=pd.to_numeric(out[c],errors='coerce')
                        if coerced.notna().sum()>len(out)*0.3:
                            _likely_num.append(c)

                # Pre-detect likely date columns
                _likely_date=[]
                for c in _all_cols:
                    if pd.api.types.is_datetime64_any_dtype(out[c]):
                        _likely_date.append(c)
                    elif out[c].dtype=='object' and c not in _likely_num:
                        sample=out[c].dropna().head(30)
                        if len(sample)>0:
                            try:
                                pd.to_datetime(sample,format='mixed',dayfirst=False)
                                _likely_date.append(c)
                            except: pass

                st.markdown(f'<p style="font-size:12px;color:var(--muted)">Select the columns you want to analyse. Pre-selected columns are auto-detected.</p>',unsafe_allow_html=True)
                me_sel1,me_sel2=st.columns(2)
                with me_sel1:
                    num_cols=st.multiselect("Numeric Columns (for metrics)",_all_cols,default=_likely_num,key="me_num_cols",
                        help="Pick columns containing numbers — AHT, Score, Duration, etc.")
                with me_sel2:
                    date_cols=st.multiselect("Date Columns (for trends)",_all_cols,default=_likely_date,key="me_date_cols",
                        help="Pick columns containing dates for trend analysis")

                if not num_cols:
                    st.info("Select at least one numeric column above to start analysing metrics.")
                else:
                    # ── Metric & grouping selectors ──
                    mx1,mx2,mx3=st.columns([2,2,1])
                    with mx1:
                        sel_metric=st.selectbox("Metric",num_cols,key="me_metric")
                    with mx2:
                        group_level=st.selectbox("Group By",["Category","Subcategory","L3","L4"],key="me_group")
                    with mx3:
                        agg_fn=st.selectbox("Aggregation",["Mean","Median","Sum","Min","Max","Count","Std Dev"],key="me_agg")

                    agg_map={"Mean":"mean","Median":"median","Sum":"sum","Min":"min","Max":"max","Count":"count","Std Dev":"std"}
                    agg_name=agg_map[agg_fn]

                    # ── Filter out NA/Uncategorized from grouping ──
                    me_df=out[(out[group_level]!='NA')&(out[group_level]!='Uncategorized')].copy()
                    me_df[sel_metric]=pd.to_numeric(me_df[sel_metric],errors='coerce')
                    me_valid=me_df.dropna(subset=[sel_metric])

                    if me_valid.empty:
                        st.warning(f"No valid numeric data in '{sel_metric}' for the selected grouping.")
                    else:
                        # ── Aggregated table ──
                        me_agg=me_valid.groupby(group_level)[sel_metric].agg(['mean','median','sum','min','max','count','std']).reset_index()
                        me_agg.columns=[group_level,'Mean','Median','Sum','Min','Max','Count','Std Dev']
                        me_agg=me_agg.sort_values(agg_fn,ascending=False)
                        for c in ['Mean','Median','Sum','Min','Max','Std Dev']:
                            me_agg[c]=me_agg[c].round(2)
                        me_agg['Count']=me_agg['Count'].astype(int)

                        # ── Modern theme helper ──
                        _PALETTE=['#2D5F6E','#D4B94E','#3D7A5F','#6B8A99','#A04040','#8C6B4A','#5A7A6B','#4A6B8C','#A8BCC8','#E8D97A']
                        def _modern_layout(fig,h=None):
                            fig.update_layout(
                                font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10,r=20,t=40,b=10),
                                xaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',zeroline=False),
                                yaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',zeroline=False),
                                hoverlabel=dict(bgcolor='#1E2D33',font_size=12,font_family='DM Sans',font_color='#E8E6DD'),
                            )
                            if h: fig.update_layout(height=h)
                            return fig
                        def _dynamic_textsize(n):
                            if n<=5: return 13
                            elif n<=10: return 11
                            elif n<=20: return 9
                            else: return 8

                        # ── Summary cards ──
                        overall_val=me_valid[sel_metric]
                        mc1,mc2,mc3,mc4,mc5=st.columns(5)
                        with mc1: st.markdown(mcard(f"Overall {agg_fn}",f"{getattr(overall_val,agg_name)():.2f}"),unsafe_allow_html=True)
                        with mc2: st.markdown(mcard("Records",f"{len(me_valid):,}","var(--slate)"),unsafe_allow_html=True)
                        with mc3: st.markdown(mcard("Groups",f"{me_valid[group_level].nunique()}","var(--success)"),unsafe_allow_html=True)
                        with mc4: st.markdown(mcard("Min",f"{overall_val.min():.2f}","var(--gold)"),unsafe_allow_html=True)
                        with mc5: st.markdown(mcard("Max",f"{overall_val.max():.2f}","var(--teal-l)"),unsafe_allow_html=True)

                        st.markdown("---")

                        # ═══ PRIMARY METRIC BAR CHART (modern, data labels) ═══
                        me_chart=me_agg[[group_level,agg_fn,'Count']].head(20)
                        n_bars=len(me_chart)
                        fig_me=go.Figure()
                        fig_me.add_trace(go.Bar(
                            y=me_chart[group_level],x=me_chart[agg_fn],orientation='h',
                            marker=dict(color=me_chart[agg_fn],colorscale=[[0,'#A8BCC8'],[0.5,'#3A7A8C'],[1,'#2D5F6E']],
                                line=dict(width=0),cornerradius=4),
                            text=[f"{v:,.1f}" for v in me_chart[agg_fn]],
                            textposition='outside',textfont=dict(size=_dynamic_textsize(n_bars),color='#2D5F6E',family='DM Sans'),
                            hovertemplate=f'<b>%{{y}}</b><br>{agg_fn} of {sel_metric}: %{{x:,.2f}}<br>Count: %{{customdata[0]:,}}<extra></extra>',
                            customdata=me_chart[['Count']].values,
                        ))
                        _modern_layout(fig_me,max(350,n_bars*32))
                        fig_me.update_layout(yaxis={'categoryorder':'total ascending'},showlegend=False,
                            title=dict(text=f"{agg_fn} of {sel_metric} by {group_level}",font=dict(size=15,color='#1E2D33'),x=0),
                            xaxis_title=None,yaxis_title=None)
                        st.plotly_chart(fig_me,use_container_width=True,key="me_fig_bar")

                        # ═══ BOX PLOT (modern) ═══
                        st.markdown(f'<div class="sh">{IC.icon(IC.ACTIVITY,"#2D5F6E",18)}{sel_metric} Distribution by {group_level}</div>',unsafe_allow_html=True)
                        top_groups=me_agg[group_level].head(12).tolist()
                        me_box=me_valid[me_valid[group_level].isin(top_groups)]
                        fig_box=go.Figure()
                        for gi,grp in enumerate(top_groups):
                            gdata=me_box[me_box[group_level]==grp][sel_metric]
                            fig_box.add_trace(go.Box(y=gdata,name=grp,marker_color=_PALETTE[gi%len(_PALETTE)],
                                boxmean='sd',line=dict(width=1.5),
                                hovertemplate=f'<b>{grp}</b><br>Value: %{{y:,.2f}}<extra></extra>'))
                        _modern_layout(fig_box,450)
                        fig_box.update_layout(showlegend=False,xaxis_tickangle=-35,
                            title=dict(text=f"{sel_metric} Distribution (Top {len(top_groups)} {group_level})",font=dict(size=15,color='#1E2D33'),x=0))
                        st.plotly_chart(fig_box,use_container_width=True,key="me_fig_box")

                        # ═══ COMBO CHART BUILDER ═══
                        st.markdown(f'<div class="sh">{IC.icon(IC.BAR,"#2D5F6E",18)}Combo Chart Builder</div>',unsafe_allow_html=True)
                        st.markdown('<p style="font-size:12px;color:var(--muted)">Overlay Category Count (bars) with one or two metrics (lines) on a dual-axis chart.</p>',unsafe_allow_html=True)

                        combo_metrics=st.multiselect("Metrics to overlay (line axis)",num_cols,default=[sel_metric] if sel_metric in num_cols else num_cols[:1],key="me_combo_metrics",max_selections=3)

                        if combo_metrics:
                            # Build combo data — count + selected metrics aggregated by group
                            combo_df=me_valid.copy()
                            for cm in combo_metrics:
                                combo_df[cm]=pd.to_numeric(combo_df[cm],errors='coerce')
                            combo_agg=combo_df.groupby(group_level).agg(
                                _count=(sel_metric,'count'),
                                **{f"_{cm}":pd.NamedAgg(column=cm,aggfunc=agg_name) for cm in combo_metrics}
                            ).reset_index().sort_values('_count',ascending=False).head(15)

                            n_combo=len(combo_agg)
                            fig_combo=go.Figure()

                            # Bars: category count
                            fig_combo.add_trace(go.Bar(
                                x=combo_agg[group_level],y=combo_agg['_count'],name='Record Count',
                                marker=dict(color='#A8BCC8',cornerradius=4,line=dict(width=0)),
                                text=[f"{int(v):,}" for v in combo_agg['_count']],
                                textposition='outside',textfont=dict(size=_dynamic_textsize(n_combo),color='#6B8A99',family='DM Sans'),
                                hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>',
                                yaxis='y',opacity=0.7,
                            ))

                            # Lines: each metric on secondary axis
                            line_colors=['#2D5F6E','#D4B94E','#A04040']
                            for mi,cm in enumerate(combo_metrics):
                                col_key=f"_{cm}"
                                vals=combo_agg[col_key]
                                fig_combo.add_trace(go.Scatter(
                                    x=combo_agg[group_level],y=vals,name=f"{agg_fn} of {cm}",
                                    mode='lines+markers+text',
                                    line=dict(color=line_colors[mi%3],width=2.5),
                                    marker=dict(size=8,color=line_colors[mi%3],line=dict(width=2,color='#FFFFFF')),
                                    text=[f"{v:,.1f}" for v in vals],
                                    textposition='top center',textfont=dict(size=_dynamic_textsize(n_combo),color=line_colors[mi%3],family='DM Sans'),
                                    hovertemplate=f'<b>%{{x}}</b><br>{agg_fn} of {cm}: %{{y:,.2f}}<extra></extra>',
                                    yaxis='y2',
                                ))

                            _modern_layout(fig_combo,480)
                            metric_label=', '.join(combo_metrics)
                            fig_combo.update_layout(
                                title=dict(text=f"Record Count vs {metric_label} by {group_level}",font=dict(size=15,color='#1E2D33'),x=0),
                                yaxis=dict(title='Record Count',showgrid=True,gridcolor='rgba(168,188,200,0.12)'),
                                yaxis2=dict(title=metric_label,overlaying='y',side='right',showgrid=False),
                                xaxis=dict(tickangle=-35),barmode='group',
                                legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,
                                    bgcolor='rgba(255,255,255,0.8)',bordercolor='#D1CFC4',borderwidth=1,
                                    font=dict(size=11)),
                            )
                            st.plotly_chart(fig_combo,use_container_width=True,key="me_fig_combo")

                        # ═══ TREND ANALYSIS ═══
                        if date_cols:
                            st.markdown("---")
                            st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#2D5F6E",18)}Trend Analysis</div>',unsafe_allow_html=True)
                            tr1,tr2=st.columns([2,1])
                            with tr1: sel_date=st.selectbox("Date Column",date_cols,key="me_date")
                            with tr2: sel_period=st.selectbox("Period",["Day","Week","Month"],key="me_period")

                            me_trend=me_valid.copy()
                            try:
                                me_trend[sel_date]=pd.to_datetime(me_trend[sel_date],errors='coerce')
                                me_trend=me_trend.dropna(subset=[sel_date])
                            except: pass

                            if not me_trend.empty and pd.api.types.is_datetime64_any_dtype(me_trend[sel_date]):
                                freq_map={"Day":"D","Week":"W","Month":"M"}
                                me_trend['_period']=me_trend[sel_date].dt.to_period(freq_map[sel_period]).astype(str)

                                # Overall trend (modern)
                                trend_overall=me_trend.groupby('_period')[sel_metric].agg(agg_name).reset_index()
                                trend_overall.columns=['Period',sel_metric]
                                fig_trend=go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=trend_overall['Period'],y=trend_overall[sel_metric],
                                    mode='lines+markers+text',
                                    line=dict(color='#2D5F6E',width=2.5,shape='spline'),
                                    marker=dict(size=7,color='#D4B94E',line=dict(width=2,color='#2D5F6E')),
                                    text=[f"{v:,.1f}" for v in trend_overall[sel_metric]],
                                    textposition='top center',textfont=dict(size=_dynamic_textsize(len(trend_overall)),color='#2D5F6E'),
                                    fill='tozeroy',fillcolor='rgba(45,95,110,0.06)',
                                    hovertemplate='<b>%{x}</b><br>'+f'{agg_fn}: %{{y:,.2f}}<extra></extra>',
                                ))
                                _modern_layout(fig_trend,370)
                                fig_trend.update_layout(title=dict(text=f"{agg_fn} of {sel_metric} Over Time",font=dict(size=15,color='#1E2D33'),x=0),
                                    xaxis_title=None,yaxis_title=sel_metric)
                                st.plotly_chart(fig_trend,use_container_width=True,key="me_fig_trend")

                                # Trend by top groups (modern)
                                top5_groups=me_agg[group_level].head(5).tolist()
                                trend_grp=me_trend[me_trend[group_level].isin(top5_groups)]
                                if not trend_grp.empty:
                                    st.markdown(f"**{sel_metric} Trend — Top 5 {group_level}**")
                                    trend_by=trend_grp.groupby(['_period',group_level])[sel_metric].agg(agg_name).reset_index()
                                    trend_by.columns=['Period',group_level,sel_metric]
                                    fig_trend2=go.Figure()
                                    for gi,grp in enumerate(top5_groups):
                                        gd=trend_by[trend_by[group_level]==grp]
                                        fig_trend2.add_trace(go.Scatter(
                                            x=gd['Period'],y=gd[sel_metric],name=grp,
                                            mode='lines+markers',
                                            line=dict(color=_PALETTE[gi%len(_PALETTE)],width=2.5,shape='spline'),
                                            marker=dict(size=6,color=_PALETTE[gi%len(_PALETTE)],line=dict(width=1.5,color='#FFFFFF')),
                                            hovertemplate=f'<b>{grp}</b><br>%{{x}}<br>{agg_fn}: %{{y:,.2f}}<extra></extra>',
                                        ))
                                    _modern_layout(fig_trend2,420)
                                    fig_trend2.update_layout(
                                        title=dict(text=f"{sel_metric} by {group_level} Over Time",font=dict(size=15,color='#1E2D33'),x=0),
                                        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,
                                            bgcolor='rgba(255,255,255,0.8)',bordercolor='#D1CFC4',borderwidth=1,font=dict(size=11)),
                                        xaxis_title=None,yaxis_title=sel_metric)
                                    st.plotly_chart(fig_trend2,use_container_width=True,key="me_fig_trend_grp")
                            else:
                                st.info("Could not parse date column for trend analysis.")

                        # ── Full stats table ──
                        st.markdown("---")
                        st.markdown(f"**Full Statistics: {sel_metric} by {group_level}**")
                        st.dataframe(me_agg,hide_index=True,use_container_width=True)

                        # ── Export ──
                        me_buf=io.BytesIO()
                        with pd.ExcelWriter(me_buf,engine='openpyxl') as writer:
                            me_agg.to_excel(writer,sheet_name='Aggregated',index=False)
                            me_valid[[group_level,sel_metric]].to_excel(writer,sheet_name='Raw Data',index=False)
                        me_buf.seek(0)
                        st.download_button(f"Export {sel_metric} Analysis",me_buf.getvalue(),
                            f"metrics_{sel_metric}_{group_level}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,key="me_dl")
            with rpt[9]:
                # ═══ CATEGORY TRENDS ═══
                st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#2D5F6E",18)}Category Volume Over Time</div>',unsafe_allow_html=True)

                # Detect date columns
                _sys={'Category','Subcategory','L3','L4','confidence','matched_rule','match_score','match_detail','Conversation_ID','Original_Text'}
                _dcols=[]
                for c in out.columns:
                    if c in _sys: continue
                    if pd.api.types.is_datetime64_any_dtype(out[c]): _dcols.append(c)
                    elif out[c].dtype=='object':
                        try:
                            pd.to_datetime(out[c].dropna().head(20),format='mixed'); _dcols.append(c)
                        except: pass

                if not _dcols:
                    st.info("No date columns detected. Upload data with a date field to see category trends over time.")
                else:
                    ct1,ct2,ct3=st.columns([2,1,1])
                    with ct1: ct_date=st.selectbox("Date Column",_dcols,key="ct_date")
                    with ct2: ct_period=st.selectbox("Period",["Day","Week","Month"],key="ct_period")
                    with ct3: ct_level=st.selectbox("Level",["Category","Subcategory","L3","L4"],key="ct_level")

                    ct_df=out.copy()
                    ct_df[ct_date]=pd.to_datetime(ct_df[ct_date],errors='coerce')
                    ct_df=ct_df.dropna(subset=[ct_date])
                    ct_df=ct_df[(ct_df[ct_level]!='NA')&(ct_df[ct_level]!='Uncategorized')]

                    if ct_df.empty:
                        st.warning("No valid data for trend analysis.")
                    else:
                        freq_map={"Day":"D","Week":"W","Month":"M"}
                        ct_df['_period']=ct_df[ct_date].dt.to_period(freq_map[ct_period]).astype(str)

                        # Top N categories by volume
                        ct_topn=st.slider("Top N categories",3,15,7,key="ct_topn")
                        top_cats=ct_df[ct_level].value_counts().head(ct_topn).index.tolist()
                        ct_filt=ct_df[ct_df[ct_level].isin(top_cats)]
                        ct_agg=ct_filt.groupby(['_period',ct_level]).size().reset_index(name='Count')

                        # Stacked area chart
                        _PAL=['#2D5F6E','#D4B94E','#3D7A5F','#6B8A99','#A04040','#8C6B4A','#5A7A6B','#4A6B8C','#A8BCC8','#E8D97A','#7A6620','#5A3D7A','#2563EB','#D1CFC4','#3A7A8C']
                        fig_ct=go.Figure()
                        for gi,cat in enumerate(top_cats):
                            gd=ct_agg[ct_agg[ct_level]==cat].sort_values('_period')
                            fig_ct.add_trace(go.Scatter(
                                x=gd['_period'],y=gd['Count'],name=cat,
                                mode='lines',stackgroup='one',
                                line=dict(width=0.5,color=_PAL[gi%len(_PAL)]),
                                fillcolor=_PAL[gi%len(_PAL)],
                                hovertemplate=f'<b>{cat}</b><br>%{{x}}<br>Count: %{{y:,}}<extra></extra>',
                            ))
                        fig_ct.update_layout(
                            font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10,r=20,t=40,b=10),height=480,
                            title=dict(text=f"{ct_level} Volume Over Time ({ct_period}ly)",font=dict(size=15,color='#1E2D33'),x=0),
                            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',title='Record Count'),
                            legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),
                            hoverlabel=dict(bgcolor='#1E2D33',font_size=12,font_family='DM Sans',font_color='#E8E6DD'),
                        )
                        st.plotly_chart(fig_ct,use_container_width=True,key="ct_fig_area")

                        # Individual line chart (non-stacked)
                        st.markdown("**Individual Trend Lines**")
                        fig_ct2=go.Figure()
                        for gi,cat in enumerate(top_cats):
                            gd=ct_agg[ct_agg[ct_level]==cat].sort_values('_period')
                            fig_ct2.add_trace(go.Scatter(
                                x=gd['_period'],y=gd['Count'],name=cat,
                                mode='lines+markers',
                                line=dict(width=2.5,color=_PAL[gi%len(_PAL)],shape='spline'),
                                marker=dict(size=5,color=_PAL[gi%len(_PAL)],line=dict(width=1,color='#FFFFFF')),
                                hovertemplate=f'<b>{cat}</b><br>%{{x}}<br>Count: %{{y:,}}<extra></extra>',
                            ))
                        fig_ct2.update_layout(
                            font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10,r=20,t=10,b=10),height=400,
                            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',title='Count'),
                            legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,font=dict(size=10)),
                            hoverlabel=dict(bgcolor='#1E2D33',font_size=12,font_family='DM Sans',font_color='#E8E6DD'),
                        )
                        st.plotly_chart(fig_ct2,use_container_width=True,key="ct_fig_lines")

            with rpt[10]:
                # ═══ TOP MOVERS ═══
                st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#2D5F6E",18)}Top Movers — Period-over-Period</div>',unsafe_allow_html=True)

                _sys2={'Category','Subcategory','L3','L4','confidence','matched_rule','match_score','match_detail','Conversation_ID','Original_Text'}
                _dcols2=[]
                for c in out.columns:
                    if c in _sys2: continue
                    if pd.api.types.is_datetime64_any_dtype(out[c]): _dcols2.append(c)
                    elif out[c].dtype=='object':
                        try:
                            pd.to_datetime(out[c].dropna().head(20),format='mixed'); _dcols2.append(c)
                        except: pass

                if not _dcols2:
                    st.info("No date columns detected. Upload data with a date field to compare periods.")
                else:
                    tm1,tm2=st.columns([2,1])
                    with tm1: tm_date=st.selectbox("Date Column",_dcols2,key="tm_date")
                    with tm2: tm_level=st.selectbox("Level",["Category","Subcategory","L3","L4"],key="tm_level")

                    tm_df=out.copy()
                    tm_df[tm_date]=pd.to_datetime(tm_df[tm_date],errors='coerce')
                    tm_df=tm_df.dropna(subset=[tm_date])
                    tm_df=tm_df[(tm_df[tm_level]!='NA')&(tm_df[tm_level]!='Uncategorized')]

                    if tm_df.empty:
                        st.warning("No valid data for period comparison.")
                    else:
                        # Determine split point — median date
                        all_dates=tm_df[tm_date].sort_values()
                        mid_date=all_dates.iloc[len(all_dates)//2]

                        tm_split=st.date_input("Split Date (before = Period 1, after = Period 2)",
                            value=mid_date.date(),key="tm_split")
                        tm_split_dt=pd.Timestamp(tm_split)

                        p1=tm_df[tm_df[tm_date]<tm_split_dt]
                        p2=tm_df[tm_df[tm_date]>=tm_split_dt]

                        p1_label=f"Before {tm_split}"
                        p2_label=f"From {tm_split}"

                        st.markdown(f'<span class="badge b-info">{p1_label}: {len(p1):,} records</span> &nbsp; '
                            f'<span class="badge b-ok">{p2_label}: {len(p2):,} records</span>',unsafe_allow_html=True)

                        if len(p1)==0 or len(p2)==0:
                            st.warning("Both periods need records. Adjust the split date.")
                        else:
                            p1_counts=p1[tm_level].value_counts().reset_index()
                            p1_counts.columns=[tm_level,'Period_1']
                            p2_counts=p2[tm_level].value_counts().reset_index()
                            p2_counts.columns=[tm_level,'Period_2']

                            comp=p1_counts.merge(p2_counts,on=tm_level,how='outer').fillna(0)
                            comp['Period_1']=comp['Period_1'].astype(int)
                            comp['Period_2']=comp['Period_2'].astype(int)
                            comp['Change']=comp['Period_2']-comp['Period_1']
                            comp['Change_%']=((comp['Period_2']-comp['Period_1'])/comp['Period_1'].replace(0,1)*100).round(1)
                            comp=comp.sort_values('Change',ascending=False)

                            # Top gainers & losers
                            top_gain=comp.head(10)
                            top_loss=comp.tail(10).sort_values('Change')

                            st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#3D7A5F",18)}Top Gainers</div>',unsafe_allow_html=True)
                            fig_gain=go.Figure()
                            fig_gain.add_trace(go.Bar(
                                y=top_gain[tm_level],x=top_gain['Change'],orientation='h',
                                marker=dict(color=['#3D7A5F' if v>=0 else '#A04040' for v in top_gain['Change']],cornerradius=4),
                                text=[f"+{int(v):,}" if v>=0 else f"{int(v):,}" for v in top_gain['Change']],
                                textposition='outside',textfont=dict(size=11,family='DM Sans'),
                                customdata=top_gain[['Change_%']].values,
                                hovertemplate='<b>%{y}</b><br>Change: %{x:,}<br>Change %: %{customdata[0]:.1f}%<extra></extra>',
                            ))
                            fig_gain.update_layout(
                                font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10,r=20,t=40,b=10),height=max(300,len(top_gain)*32),
                                yaxis={'categoryorder':'total ascending'},showlegend=False,
                                title=dict(text="Volume Increase",font=dict(size=14,color='#3D7A5F'),x=0),
                                xaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',title='Change in Count'),
                            )
                            st.plotly_chart(fig_gain,use_container_width=True,key="tm_fig_gain")

                            st.markdown(f'<div class="sh">{IC.icon(IC.TRENDING,"#A04040",18)}Top Decliners</div>',unsafe_allow_html=True)
                            fig_loss=go.Figure()
                            fig_loss.add_trace(go.Bar(
                                y=top_loss[tm_level],x=top_loss['Change'],orientation='h',
                                marker=dict(color=['#3D7A5F' if v>=0 else '#A04040' for v in top_loss['Change']],cornerradius=4),
                                text=[f"+{int(v):,}" if v>=0 else f"{int(v):,}" for v in top_loss['Change']],
                                textposition='outside',textfont=dict(size=11,family='DM Sans'),
                                customdata=top_loss[['Change_%']].values,
                                hovertemplate='<b>%{y}</b><br>Change: %{x:,}<br>Change %: %{customdata[0]:.1f}%<extra></extra>',
                            ))
                            fig_loss.update_layout(
                                font_family="DM Sans",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10,r=20,t=40,b=10),height=max(300,len(top_loss)*32),
                                yaxis={'categoryorder':'total descending'},showlegend=False,
                                title=dict(text="Volume Decrease",font=dict(size=14,color='#A04040'),x=0),
                                xaxis=dict(showgrid=True,gridcolor='rgba(168,188,200,0.15)',title='Change in Count'),
                            )
                            st.plotly_chart(fig_loss,use_container_width=True,key="tm_fig_loss")

                            # Full comparison table
                            st.markdown("---")
                            st.markdown("**Full Period Comparison**")
                            display_comp=comp.copy()
                            display_comp.columns=[tm_level,p1_label,p2_label,'Change','Change %']
                            st.dataframe(display_comp,hide_index=True,use_container_width=True)

                            # Export
                            tm_buf=io.BytesIO()
                            display_comp.to_excel(tm_buf,index=False,engine='openpyxl')
                            tm_buf.seek(0)
                            st.download_button("Export Comparison",tm_buf.getvalue(),
                                f"top_movers_{tm_level}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,key="tm_dl")

    # ═══════════════════════════════════════════════════════════════════════════
    # NARRATIVE INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Narrative Intelligence":
        shdr(IC.TRENDING, "Narrative Intelligence")
        if 'out' not in st.session_state:
            st.info("Run analysis first to generate narratives.")
        else:
            out=st.session_state.out
            total=len(out)
            unc_n=int((out['Category']=='Uncategorized').sum())
            cat_n=total-unc_n
            coverage=cat_n/total*100 if total>0 else 0

            # Category stats
            cat_counts=out[out['Category']!='Uncategorized']['Category'].value_counts()
            top_cat=cat_counts.index[0] if len(cat_counts)>0 else "N/A"
            top_cat_pct=cat_counts.iloc[0]/total*100 if len(cat_counts)>0 else 0
            top3_cats=cat_counts.head(3)

            # Subcategory stats
            sub_counts=out[out['Subcategory']!='NA']['Subcategory'].value_counts()
            top_sub=sub_counts.index[0] if len(sub_counts)>0 else "N/A"

            # Confidence stats
            avg_conf=out['confidence'].mean() if 'confidence' in out.columns else 0
            low_conf_pct=(out['confidence']<0.3).mean()*100 if 'confidence' in out.columns else 0

            # Detect numeric metadata columns for insights
            _sys_n={'Category','Subcategory','L3','L4','confidence','matched_rule','match_score','match_detail','Conversation_ID','Original_Text'}
            num_meta=[c for c in out.columns if c not in _sys_n and pd.api.types.is_numeric_dtype(out[c])]
            # Also try coercing object columns
            for c in out.columns:
                if c in _sys_n or c in num_meta: continue
                if out[c].dtype=='object':
                    coerced=pd.to_numeric(out[c],errors='coerce')
                    if coerced.notna().sum()>total*0.3: num_meta.append(c)

            # ═══ EXECUTIVE SUMMARY NARRATIVE ═══
            st.markdown(f"""<div style="background:linear-gradient(135deg,#2D5F6E 0%,#3A7A8C 100%);border-radius:12px;padding:28px 32px;margin-bottom:24px">
            <span style="font-size:18px;font-weight:700;color:#FFFFFF">{IC.icon(IC.REPORT,'#FFFFFF',20)}Executive Summary</span>
            <p style="font-size:10px;color:#A8BCC8;text-transform:uppercase;letter-spacing:1.5px;margin:6px 0 0">Auto-generated from classification results</p>
            </div>""",unsafe_allow_html=True)

            # Build narrative paragraphs
            narrative_parts=[]

            # Para 1: Overview
            cov_verdict="excellent" if coverage>90 else "good" if coverage>75 else "moderate" if coverage>50 else "low"
            narrative_parts.append(
                f"The analysis processed **{total:,}** records, achieving **{coverage:.1f}%** classification coverage ({cov_verdict}). "
                f"**{cat_n:,}** records were successfully categorized across **{cat_counts.nunique()}** categories and "
                f"**{sub_counts.nunique()}** subcategories, while **{unc_n:,}** records ({100-coverage:.1f}%) remain uncategorized."
            )

            # Para 2: Top categories
            top3_text=', '.join([f"**{cat}** ({cnt:,}, {cnt/total*100:.1f}%)" for cat,cnt in top3_cats.items()])
            narrative_parts.append(
                f"The top drivers by volume are {top3_text}. "
                f"Together, the top 3 categories account for **{top3_cats.sum()/total*100:.1f}%** of all classified interactions."
            )

            # Para 3: Confidence
            if 'confidence' in out.columns:
                conf_verdict="high" if avg_conf>0.7 else "moderate" if avg_conf>0.4 else "low"
                narrative_parts.append(
                    f"Average classification confidence is **{avg_conf:.2f}** ({conf_verdict}). "
                    f"**{low_conf_pct:.1f}%** of records have low confidence (<0.3), suggesting potential rule refinement opportunities in those areas."
                )

            # Para 4: Metadata insights (if numeric columns exist)
            for mc in num_meta[:3]:
                mc_data=pd.to_numeric(out[mc],errors='coerce').dropna()
                if len(mc_data)>0:
                    mc_mean=mc_data.mean()
                    # Find category with highest and lowest average
                    mc_by_cat=out.copy()
                    mc_by_cat[mc]=pd.to_numeric(mc_by_cat[mc],errors='coerce')
                    mc_by_cat=mc_by_cat[(mc_by_cat['Category']!='Uncategorized')&(mc_by_cat['Category']!='NA')].dropna(subset=[mc])
                    if not mc_by_cat.empty:
                        cat_avg=mc_by_cat.groupby('Category')[mc].mean()
                        hi_cat=cat_avg.idxmax(); hi_val=cat_avg.max()
                        lo_cat=cat_avg.idxmin(); lo_val=cat_avg.min()
                        narrative_parts.append(
                            f"For **{mc}**, the overall average is **{mc_mean:,.2f}**. "
                            f"**{hi_cat}** has the highest average ({hi_val:,.2f}), while **{lo_cat}** has the lowest ({lo_val:,.2f}). "
                            f"The spread suggests {'significant variation' if (hi_val-lo_val)/mc_mean>0.5 else 'moderate variation' if (hi_val-lo_val)/mc_mean>0.2 else 'relatively consistent performance'} across categories."
                        )

            # Para 5: Recommendations
            recs=[]
            if coverage<80:
                recs.append(f"Classification coverage is at {coverage:.0f}% — review the **Uncategorized Analysis** in Reports to identify top unclassified keywords and create new rules.")
            if low_conf_pct>20:
                recs.append(f"{low_conf_pct:.0f}% of records have low confidence — consider adding more specific multi-term rules (AND/NEAR) to improve match precision.")
            if unc_n>total*0.15:
                recs.append(f"{unc_n:,} records are uncategorized — use **Concordance** to explore these records in context and build targeted rules.")
            if len(cat_counts)>0 and cat_counts.iloc[0]/total>0.3:
                recs.append(f"**{top_cat}** dominates at {top_cat_pct:.0f}% — consider splitting it into L2/L3 subcategories for more granular insights.")
            if not recs:
                recs.append("Classification coverage and confidence are strong. Continue monitoring for emerging categories via the **Top Movers** tab in Reports.")

            # Render narrative
            for p in narrative_parts:
                st.markdown(p)
                st.markdown("")

            # Recommendations section
            st.markdown(f'<div class="sh">{IC.icon(IC.ALERT,"#2D5F6E",18)}Recommendations</div>',unsafe_allow_html=True)
            for r in recs:
                st.markdown(f'<span style="color:var(--teal);font-weight:700">→</span> {r}',unsafe_allow_html=True)

            # ═══ KEY METRICS AT A GLANCE ═══
            st.markdown("---")
            st.markdown(f'<div class="sh">{IC.icon(IC.BAR,"#2D5F6E",18)}Key Metrics at a Glance</div>',unsafe_allow_html=True)

            km1,km2,km3,km4,km5,km6=st.columns(6)
            with km1: st.markdown(mcard("Records",f"{total:,}"),unsafe_allow_html=True)
            with km2: st.markdown(mcard("Coverage",f"{coverage:.1f}%","var(--success)" if coverage>80 else "var(--gold)"),unsafe_allow_html=True)
            with km3: st.markdown(mcard("Categories",str(cat_counts.nunique()),"var(--teal)"),unsafe_allow_html=True)
            with km4: st.markdown(mcard("Top Driver",top_cat[:15],"var(--slate)"),unsafe_allow_html=True)
            with km5: st.markdown(mcard("Avg Confidence",f"{avg_conf:.2f}","var(--success)" if avg_conf>0.6 else "var(--gold)"),unsafe_allow_html=True)
            with km6: st.markdown(mcard("Uncategorized",f"{unc_n:,}","var(--err)" if unc_n/total>0.15 else "var(--gold)"),unsafe_allow_html=True)

            # Metadata metrics
            if num_meta:
                st.markdown("**Metadata Metrics by Top Category**")
                nm_level=st.selectbox("Group By",["Category","Subcategory","L3","L4"],key="ni_group")
                nm_cols=st.multiselect("Metrics",num_meta,default=num_meta[:3],key="ni_metrics")
                if nm_cols:
                    nm_df=out[(out[nm_level]!='NA')&(out[nm_level]!='Uncategorized')].copy()
                    for mc in nm_cols:
                        nm_df[mc]=pd.to_numeric(nm_df[mc],errors='coerce')
                    nm_agg=nm_df.groupby(nm_level)[nm_cols].mean().round(2).reset_index()
                    nm_agg=nm_agg.sort_values(nm_cols[0],ascending=False).head(15)
                    st.dataframe(nm_agg,hide_index=True,use_container_width=True)

            # ═══ EXPORT NARRATIVE ═══
            st.markdown("---")
            full_narrative="EXECUTIVE SUMMARY — TextInsightMiner\n"+"="*50+"\n\n"
            full_narrative+="\n\n".join(p.replace("**","") for p in narrative_parts)
            full_narrative+="\n\nRECOMMENDATIONS\n"+"-"*30+"\n"
            full_narrative+="\n".join(f"• {r.replace('**','').replace('<span style=\"color:var(--teal);font-weight:700\">→</span> ','')}" for r in recs)
            full_narrative+=f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            e1,e2=st.columns(2)
            with e1:
                st.download_button("Export as Text",full_narrative,
                    f"narrative_{datetime.now().strftime('%Y%m%d')}.txt","text/plain",
                    use_container_width=True,key="ni_dl_txt")
            with e2:
                ni_buf=io.BytesIO()
                ni_excel_data={'Metric':[],'Value':[]}
                ni_excel_data['Metric'].extend(['Total Records','Categorized','Uncategorized','Coverage %','Categories','Subcategories','Top Category','Avg Confidence'])
                ni_excel_data['Value'].extend([total,cat_n,unc_n,f"{coverage:.1f}",cat_counts.nunique(),sub_counts.nunique(),top_cat,f"{avg_conf:.2f}"])
                for mc in num_meta[:5]:
                    mc_val=pd.to_numeric(out[mc],errors='coerce').mean()
                    ni_excel_data['Metric'].append(f"Avg {mc}")
                    ni_excel_data['Value'].append(f"{mc_val:.2f}")
                with pd.ExcelWriter(ni_buf,engine='openpyxl') as writer:
                    pd.DataFrame(ni_excel_data).to_excel(writer,sheet_name='Summary',index=False)
                    if len(cat_counts)>0:
                        cat_counts.reset_index().rename(columns={'index':'Category','count':'Count'} if 'index' in cat_counts.reset_index().columns else {}).to_excel(writer,sheet_name='Categories',index=False)
                ni_buf.seek(0)
                st.download_button("Export as Excel",ni_buf.getvalue(),
                    f"narrative_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,key="ni_dl_xlsx")

    # ═══════════════════════════════════════════════════════════════════════════
    # S3: RULE BUILDER
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Rule Builder":
        shdr(IC.TOOL, "Rule Builder")
        render_rule_builder(ind_rules)

    # ═══════════════════════════════════════════════════════════════════════════
    # S4: CONCORDANCE
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Concordance":
        shdr(IC.SEARCH, "Concordance (KWIC)")
        if 'out' not in st.session_state:
            st.info("Run analysis first.")
        else:
            out = st.session_state.out

            # ── Uncategorized filter toggle ──
            auto_uncat=st.session_state.pop('_conc_filter_uncat',False)
            prefill_kw=st.session_state.pop('_conc_prefill_kw','')
            unc_only=st.toggle("Uncategorized Records Only",value=auto_uncat,key="conc_unc_toggle",
                help="When enabled, searches only within uncategorized records")
            if unc_only:
                work_df=out[out['Category']=='Uncategorized'].copy()
                unc_count=len(work_df)
                st.markdown(f'<span class="badge b-warn">Filtered to {unc_count:,} uncategorized records</span>',unsafe_allow_html=True)
            else:
                work_df=out

            kc1,kc2,kc3 = st.columns([3,2,1])
            with kc1:
                default_kw=prefill_kw if prefill_kw else ""
                skw = st.text_input("Keyword or Phrase", value=default_kw, placeholder="e.g. cancel  or  cancel my subscription", key="ckw")
            with kc2:
                cat_opts=["All"]+sorted(work_df['Category'].unique().tolist())
                cf = st.selectbox("Category Filter", cat_opts, key="cc")
            with kc3: cx = st.slider("Context", 5, 25, 10, key="cx")

            # Auto-search if prefilled from Reports
            auto_search=bool(prefill_kw)
            if (st.button("Search", key="cb") or auto_search) and skw:
                ca = Concordance(work_df); r = ca.search(skw, cx, cf if cf != "All" else None)
                st.session_state.conc_r = r; st.session_state.conc_s = ca.stats(r)
                st.session_state.conc_co = ca.colloc(r); st.session_state.conc_ca = ca; st.session_state.conc_kw = skw
            if st.session_state.get('conc_r'):
                r = st.session_state.conc_r; s = st.session_state.conc_s
                cm1,cm2,cm3,cm4 = st.columns(4)
                with cm1: st.markdown(mcard("Matches", f"{s['n']:,}"), unsafe_allow_html=True)
                with cm2: st.markdown(mcard("Conversations", str(s['uc']), "var(--slate)"), unsafe_allow_html=True)
                with cm3: st.markdown(mcard("Categories", str(s['cats']), "var(--steel)"), unsafe_allow_html=True)
                with cm4: st.markdown(mcard("Avg/Conv", str(s['avg']), "var(--success)"), unsafe_allow_html=True)
                co = st.session_state.conc_co
                if co['l'] or co['r']:
                    with st.expander("Collocations"):
                        lc,rc = st.columns(2)
                        with lc:
                            for w,c in co['l']: st.markdown(f'<span class="tag tag-b">{w}</span> {c}', unsafe_allow_html=True)
                        with rc:
                            for w,c in co['r']: st.markdown(f'<span class="tag tag-p">{w}</span> {c}', unsafe_allow_html=True)
                dn = min(100, len(r))
                if len(r) > 100: dn = st.slider("Show", 10, min(500,len(r)), 50, step=10, key="cdn")
                for x in r[:dn]:
                    st.markdown(f'<div class="cl"><div class="cmeta">{x["Conversation_ID"]} | {x["Category"]}>{x["Subcategory"]}</div>{x["Left"]} <span class="ckw">{x["KW"]}</span> {x["Right"]}</div>', unsafe_allow_html=True)

                # ── Build Rule from Concordance ──
                st.markdown("---")
                conc_cols=st.columns([1,1,2])
                with conc_cols[0]: ef = st.radio("Export", ['csv','xlsx'], horizontal=True, key="cef")
                with conc_cols[1]:
                    ca = st.session_state.get('conc_ca'); kw = st.session_state.get('conc_kw','')
                    if ca: st.download_button(f"Download (.{ef})", ca.export(r,ef), f"conc_{kw}_{datetime.now().strftime('%Y%m%d')}.{ef}", use_container_width=True, key="conc_dl")
                with conc_cols[2]:
                    if st.button("Create Rule from this keyword",key="conc_to_rb",type="primary",use_container_width=True):
                        st.session_state._rb_prefill_term=kw
                        st.session_state._nav_target="Rule Builder"
                        st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # S5: AUDIT TRAIL
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Audit Trail":
        shdr(IC.EYE, "Audit Trail")
        if 'out' not in st.session_state:
            st.info("Run analysis first.")
        elif 'matched_rule' in st.session_state.out.columns:
            out = st.session_state.out
            ac = [c for c in ['Conversation_ID','Category','Subcategory','L3','L4','confidence','matched_rule','match_score','match_detail'] if c in out.columns]
            a1,a2 = st.columns(2)
            with a1: acat = st.selectbox("Category", ["All"]+sorted(out['Category'].unique().tolist()), key="at_cat")
            with a2: acf = st.selectbox("Confidence", ["All","High (>0.7)","Medium (0.3-0.7)","Low (<0.3)"], key="at_conf")
            ad = out.copy()
            if acat != "All": ad = ad[ad['Category']==acat]
            if "High" in acf: ad = ad[ad['confidence']>0.7]
            elif "Medium" in acf: ad = ad[(ad['confidence']>=0.3)&(ad['confidence']<=0.7)]
            elif "Low" in acf: ad = ad[ad['confidence']<0.3]
            st.dataframe(ad[ac].head(300), hide_index=True, use_container_width=True, height=400)

    # ═══════════════════════════════════════════════════════════════════════════
    # S6: RULE PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Rule Performance":
        shdr(IC.ACTIVITY, "Rule Performance")
        if 'out' not in st.session_state:
            st.info("Run analysis first.")
        else:
            out = st.session_state.out
            p1,p2,p3 = st.tabs(["Hit Rates","Run History","Uncategorized Sample"])
            with p1:
                rp = st.session_state.rh.get_rule_perf()
                if not rp.empty:
                    st.dataframe(rp.head(50), hide_index=True, use_container_width=True)
                    nf = set(r.rule_id for r in all_rules)-set(rp['rule_id'].tolist())
                    if nf: st.markdown(f'<span class="badge b-warn">{len(nf)} rules never fired</span>', unsafe_allow_html=True)
                else: st.info("No data yet.")
            with p2:
                h = st.session_state.rh.get_history()
                if not h.empty: st.dataframe(h, hide_index=True, use_container_width=True)
                else: st.info("No history.")
            with p3:
                sm = st.session_state.rh.get_uncat(out, 15)
                if not sm.empty: st.dataframe(sm, hide_index=True, use_container_width=True)
                else: st.success("All categorized.")

    # ═══════════════════════════════════════════════════════════════════════════
    # S7: PROJECTS
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "Projects":
        shdr(IC.GLOBE, "Projects")
        mapping_path = os.path.join(DOMAIN_PACKS_DIR, "master_company_industry_mapping.json")
        if 'project_mapping' not in st.session_state:
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, encoding='utf-8') as f: st.session_state.project_mapping = json.load(f)
                except: st.session_state.project_mapping = {"version":"1.0","industries":{}}
            else: st.session_state.project_mapping = {"version":"1.0","industries":{}}
        pm = st.session_state.project_mapping
        pt1, pt2 = st.tabs(["Add Project","View Projects"])
        with pt1:
            with st.form("add_project", clear_on_submit=True):
                pc1, pc2 = st.columns(2)
                with pc1: proj_name = st.text_input("Company *", placeholder="e.g. Netflix Chat")
                with pc2: domain_choice = st.selectbox("Domain", ["(select)","(new)"]+sorted(pm.get('industries',{}).keys()))
                new_domain = ""
                if domain_choice == "(new)": new_domain = st.text_input("New Domain")
                if st.form_submit_button("Register", type="primary") and proj_name.strip():
                    td = new_domain.strip() if domain_choice == "(new)" and new_domain.strip() else domain_choice
                    if td and td not in ("(select)","(new)"):
                        pm['industries'].setdefault(td, [])
                        if proj_name.strip() not in pm['industries'][td]:
                            pm['industries'][td].append(proj_name.strip())
                            try:
                                os.makedirs(DOMAIN_PACKS_DIR, exist_ok=True)
                                with open(mapping_path, 'w', encoding='utf-8') as f: json.dump(pm, f, indent=2)
                                st.success(f"Registered '{proj_name.strip()}' under '{td}'")
                            except Exception as e: st.error(str(e))
        with pt2:
            if pm.get('industries'):
                for d in sorted(pm['industries'].keys()):
                    tags = ' '.join(f'<span class="tag tag-b">{c}</span>' for c in pm['industries'][d])
                    st.markdown(f'<div class="rc"><strong>{d}</strong> ({len(pm["industries"][d])})<br>{tags}</div>', unsafe_allow_html=True)
                st.download_button("Export Mapping", json.dumps(pm, indent=2), "project_mapping.json", "application/json", use_container_width=True, key="proj_dl")

    # ── EXPORT ──
    st.markdown("---")
    if 'out' in st.session_state:
        out = st.session_state.out; of = st.session_state.get('of','csv')
        _export_key=f"{id(out)}_{len(out)}_{of}"
        if st.session_state.get('_export_cache_key')!=_export_key:
            ec = [c for c in ['Conversation_ID','Original_Text','Category','Subcategory','L3','L4','confidence','matched_rule'] if c in out.columns]
            st.session_state._export_cache_bytes=FH.save(pl.from_pandas(out[ec]), of)
            st.session_state._export_cache_key=_export_key
        eb=st.session_state._export_cache_bytes
        _fname=st.session_state.get('_uploaded_name','results')
        _fname_base=_fname.rsplit('.',1)[0] if '.' in _fname else _fname
        _, dc, _ = st.columns([1,2,1])
        with dc:
            st.download_button(f"Download {_fname_base}_classified.{of}", eb, f"{_fname_base}_classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{of}", type="primary", use_container_width=True, key="main_dl")

    st.markdown(f'<div style="text-align:center;color:#6B8A99;font-size:11px;padding:12px 0">{IC.icon(IC.PICKAXE,"#6B8A99",13)}TextInsightMiner v10.1 — Dig Deeper. Classify Smarter.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    is_app = st.session_state.get('page') == 'app'
    st.set_page_config(
        page_title="TextInsightMiner", layout="wide",
        initial_sidebar_state="expanded" if is_app else "collapsed",
        page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232D5F6E'><rect width='24' height='24' rx='4'/></svg>"
    )
    if is_app: main_app()
    else: render_landing()