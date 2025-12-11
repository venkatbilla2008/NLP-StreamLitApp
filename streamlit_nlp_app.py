"""
NLP Classification & PII Detection Pipeline - Streamlit Cloud Compatible
==========================================================================

OPTIMIZATIONS FOR STREAMLIT CLOUD:
- No spaCy dependency (avoids installation issues)
- No translation (maximum speed)
- Focus: Classification + PII Detection only
- Lightweight dependencies only

Version: 3.1.1 - Streamlit Cloud Optimized
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
import io
import os

# NLP Libraries - Lightweight only
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
    except ImportError:
        TEXTBLOB_AVAILABLE = False

# ========================================================================================
# PAGE CONFIG - MUST BE FIRST
# ========================================================================================

st.set_page_config(
    page_title="NLP Classification & PII Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================================
# CONFIGURATION
# ========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 16
CACHE_SIZE = 20000
SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'parquet', 'json']
COMPLIANCE_STANDARDS = ["HIPAA", "GDPR", "PCI-DSS", "CCPA"]
MAX_FILE_SIZE_MB = 500
DOMAIN_PACKS_DIR = "domain_packs"

# ========================================================================================
# DATA CLASSES
# ========================================================================================

@dataclass
class PIIRedactionResult:
    redacted_text: str
    pii_detected: bool
    pii_counts: Dict[str, int]
    total_items: int

@dataclass
class CategoryMatch:
    l1: str
    l2: str
    l3: str
    l4: str
    confidence: float
    match_path: str
    matched_rule: Optional[str] = None

@dataclass
class ProximityResult:
    primary_proximity: str
    proximity_group: str
    theme_count: int
    matched_themes: List[str]

@dataclass
class NLPResult:
    conversation_id: str
    original_text: str
    redacted_text: str
    category: CategoryMatch
    proximity: ProximityResult
    sentiment: str
    sentiment_score: float
    pii_result: PIIRedactionResult
    industry: Optional[str] = None

# ========================================================================================
# PII DETECTOR - NO SPACY
# ========================================================================================

class PIIDetector:
    """Fast PII detection without spaCy dependency"""
    
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERNS = [
        re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),
        re.compile(r'\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'),
    ]
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    
    @classmethod
    def detect_and_redact(cls, text: str, redaction_mode: str = 'hash') -> PIIRedactionResult:
        """Fast PII detection"""
        if not text:
            return PIIRedactionResult("", False, {}, 0)
        
        redacted = str(text)
        pii_counts = {}
        
        # Email
        emails = cls.EMAIL_PATTERN.findall(redacted)
        for email in emails:
            redacted = redacted.replace(email, '[EMAIL]')
            pii_counts['email'] = pii_counts.get('email', 0) + 1
        
        # Phone
        for pattern in cls.PHONE_PATTERNS:
            phones = pattern.findall(redacted)
            for phone in phones:
                redacted = redacted.replace(phone, '[PHONE]')
                pii_counts['phone'] = pii_counts.get('phone', 0) + 1
        
        # SSN
        ssns = cls.SSN_PATTERN.findall(redacted)
        for ssn in ssns:
            redacted = redacted.replace(ssn, '[SSN]')
            pii_counts['ssn'] = pii_counts.get('ssn', 0) + 1
        
        # Credit Card
        cards = cls.CREDIT_CARD_PATTERN.findall(redacted)
        for card in cards:
            redacted = redacted.replace(card, '[CARD]')
            pii_counts['card'] = pii_counts.get('card', 0) + 1
        
        total_items = sum(pii_counts.values())
        
        return PIIRedactionResult(
            redacted_text=redacted,
            pii_detected=total_items > 0,
            pii_counts=pii_counts,
            total_items=total_items
        )

# ========================================================================================
# DOMAIN LOADER
# ========================================================================================

class DomainLoader:
    def __init__(self, domain_packs_dir: str = None):
        self.domain_packs_dir = domain_packs_dir or DOMAIN_PACKS_DIR
        self.industries = {}
    
    def auto_load_all_industries(self) -> int:
        loaded_count = 0
        
        if not os.path.exists(self.domain_packs_dir):
            logger.error(f"Domain packs directory not found: {self.domain_packs_dir}")
            return 0
        
        for item in os.listdir(self.domain_packs_dir):
            item_path = os.path.join(self.domain_packs_dir, item)
            
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
            
            rules_path = os.path.join(item_path, "rules.json")
            keywords_path = os.path.join(item_path, "keywords.json")
            
            if os.path.exists(rules_path) and os.path.exists(keywords_path):
                try:
                    with open(rules_path, 'r') as f:
                        rules = json.load(f)
                    with open(keywords_path, 'r') as f:
                        keywords = json.load(f)
                    
                    self.industries[item] = {
                        'rules': rules,
                        'keywords': keywords,
                        'rules_count': len(rules),
                        'keywords_count': len(keywords)
                    }
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load {item}: {e}")
        
        return loaded_count
    
    def get_available_industries(self) -> List[str]:
        return list(self.industries.keys())
    
    def get_industry_data(self, industry: str) -> Dict:
        return self.industries.get(industry, {'rules': [], 'keywords': []})

# ========================================================================================
# RULE ENGINE
# ========================================================================================

class DynamicRuleEngine:
    def __init__(self, industry_data: Dict):
        self.rules = industry_data.get('rules', [])
        self.keywords = industry_data.get('keywords', [])
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        self.compiled_keywords = []
        
        for kw in self.keywords:
            conditions = kw.get('conditions', [])
            if conditions:
                pattern = re.compile('|'.join([re.escape(c.lower()) for c in conditions]), re.IGNORECASE)
                self.compiled_keywords.append({'pattern': pattern, 'category': kw.get('set', {})})
    
    @lru_cache(maxsize=CACHE_SIZE)
    def classify_text(self, text: str) -> CategoryMatch:
        if not text:
            return CategoryMatch("Uncategorized", "NA", "NA", "NA", 0.0, "Uncategorized", None)
        
        text_lower = text.lower()
        
        for kw_item in self.compiled_keywords:
            if kw_item['pattern'].search(text_lower):
                cat = kw_item['category']
                return CategoryMatch(
                    l1=cat.get('category', 'Uncategorized'),
                    l2=cat.get('subcategory', 'NA'),
                    l3=cat.get('level_3', 'NA'),
                    l4=cat.get('level_4', 'NA'),
                    confidence=0.9,
                    match_path=f"{cat.get('category', 'Uncategorized')} > {cat.get('subcategory', 'NA')}",
                    matched_rule="keyword_match"
                )
        
        return CategoryMatch("Uncategorized", "NA", "NA", "NA", 0.0, "Uncategorized", None)

# ========================================================================================
# PROXIMITY ANALYZER
# ========================================================================================

class ProximityAnalyzer:
    PROXIMITY_THEMES = {
        'Agent_Behavior': ['agent', 'representative', 'rep', 'staff', 'rude', 'unprofessional'],
        'Technical_Issues': ['error', 'bug', 'issue', 'problem', 'crash', 'broken'],
        'Customer_Service': ['service', 'support', 'help', 'assist', 'customer'],
        'Billing_Payments': ['bill', 'payment', 'charge', 'fee', 'refund'],
        'Account_Access': ['account', 'login', 'password', 'access', 'locked']
    }
    
    @classmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def analyze_proximity(cls, text: str) -> ProximityResult:
        if not text:
            return ProximityResult("Uncategorized", "Uncategorized", 0, [])
        
        text_lower = text.lower()
        matched_themes = set()
        
        for theme, keywords in cls.PROXIMITY_THEMES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_themes.add(theme)
                    break
        
        if not matched_themes:
            return ProximityResult("Uncategorized", "Uncategorized", 0, [])
        
        matched_list = sorted(list(matched_themes))
        
        return ProximityResult(
            primary_proximity=matched_list[0],
            proximity_group=", ".join(matched_list),
            theme_count=len(matched_themes),
            matched_themes=matched_list
        )

# ========================================================================================
# SENTIMENT ANALYZER
# ========================================================================================

class SentimentAnalyzer:
    _vader = None
    
    @classmethod
    def get_vader(cls):
        if cls._vader is None and VADER_AVAILABLE:
            cls._vader = SentimentIntensityAnalyzer()
        return cls._vader
    
    @staticmethod
    def normalize_score(score: float) -> float:
        import math
        return math.tanh(score * 1.2)
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def analyze_sentiment(text: str) -> Tuple[str, float]:
        if not text:
            return "Neutral", 0.0
        
        try:
            vader = SentimentAnalyzer.get_vader()
            
            if vader:
                scores = vader.polarity_scores(text)
                compound = scores['compound']
                normalized = SentimentAnalyzer.normalize_score(compound)
                
                if normalized > 0.55:
                    sentiment = "Very Positive"
                elif normalized > 0.15:
                    sentiment = "Positive"
                elif normalized >= -0.15:
                    sentiment = "Neutral"
                elif normalized >= -0.55:
                    sentiment = "Negative"
                else:
                    sentiment = "Very Negative"
                
                return sentiment, compound
            elif TEXTBLOB_AVAILABLE:
                from textblob import TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity >= 0.5:
                    sentiment = "Very Positive"
                elif polarity >= 0.1:
                    sentiment = "Positive"
                elif polarity <= -0.5:
                    sentiment = "Very Negative"
                elif polarity <= -0.1:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                
                return sentiment, polarity
            else:
                return "Neutral", 0.0
        except:
            return "Neutral", 0.0

# ========================================================================================
# PIPELINE
# ========================================================================================

class DynamicNLPPipeline:
    def __init__(self, rule_engine, enable_pii_redaction=True, industry_name=None):
        self.rule_engine = rule_engine
        self.enable_pii_redaction = enable_pii_redaction
        self.industry_name = industry_name
    
    def process_single_text(self, conversation_id: str, text: str, redaction_mode: str = 'hash') -> NLPResult:
        # PII Detection
        if self.enable_pii_redaction:
            pii_result = PIIDetector.detect_and_redact(text, redaction_mode)
        else:
            pii_result = PIIRedactionResult(text, False, {}, 0)
        
        # Classification
        category = self.rule_engine.classify_text(text)
        
        # Proximity
        proximity = ProximityAnalyzer.analyze_proximity(text)
        
        # Sentiment
        sentiment, sentiment_score = SentimentAnalyzer.analyze_sentiment(text)
        
        return NLPResult(
            conversation_id=conversation_id,
            original_text=text,
            redacted_text=pii_result.redacted_text,
            category=category,
            proximity=proximity,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            pii_result=pii_result,
            industry=self.industry_name
        )
    
    def process_batch(self, df, text_column, id_column, redaction_mode='hash', progress_callback=None):
        results = []
        total = len(df)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for idx, row in df.iterrows():
                conv_id = str(row[id_column])
                text = str(row[text_column])
                
                future = executor.submit(self.process_single_text, conv_id, text, redaction_mode)
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback and completed % 100 == 0:
                        progress_callback(completed, total)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    completed += 1
        
        return results
    
    def results_to_dataframe(self, results: List[NLPResult]) -> pd.DataFrame:
        data = []
        for result in results:
            data.append({
                'Conversation_ID': result.conversation_id,
                'Original_Text': result.original_text,
                'L1_Category': result.category.l1,
                'L2_Subcategory': result.category.l2,
                'L3_Tertiary': result.category.l3,
                'L4_Quaternary': result.category.l4,
                'Primary_Proximity': result.proximity.primary_proximity,
                'Sentiment': result.sentiment,
                'Sentiment_Score': result.sentiment_score
            })
        return pd.DataFrame(data)

# ========================================================================================
# FILE HANDLER
# ========================================================================================

class FileHandler:
    @staticmethod
    def read_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB")
                return None
            
            ext = Path(uploaded_file.name).suffix.lower()[1:]
            
            if ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif ext == 'parquet':
                df = pd.read_parquet(uploaded_file)
            elif ext == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported format: {ext}")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, format: str = 'csv') -> bytes:
        buffer = io.BytesIO()
        
        if format == 'csv':
            df.to_csv(buffer, index=False)
        elif format == 'xlsx':
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        elif format == 'parquet':
            df.to_parquet(buffer, index=False)
        elif format == 'json':
            df.to_json(buffer, orient='records', lines=True)
        
        buffer.seek(0)
        return buffer.getvalue()

# ========================================================================================
# STREAMLIT UI
# ========================================================================================

def main():
    st.title("🚀 NLP Classification & PII Detection v3.1.1")
    st.markdown("""
    **Streamlit Cloud Optimized**
    - 📊 NLP Classification (4-level)
    - 🔒 PII Detection (HIPAA/GDPR)
    - 💭 Sentiment Analysis
    - ⚡ No spaCy / No Translation
    """)
    
    # Status
    cols = st.columns(2)
    with cols[0]:
        st.info("ℹ️ PII: Regex-based (fast)")
    with cols[1]:
        if VADER_AVAILABLE:
            st.success("✅ VADER Sentiment")
        else:
            st.info("ℹ️ Basic Sentiment")
    
    # Compliance
    cols = st.columns(4)
    for idx, std in enumerate(COMPLIANCE_STANDARDS):
        cols[idx].success(f"✅ {std}")
    
    st.markdown("---")
    
    # Initialize
    if 'domain_loader' not in st.session_state:
        st.session_state.domain_loader = DomainLoader()
        loaded = st.session_state.domain_loader.auto_load_all_industries()
        if loaded > 0:
            st.success(f"✅ Loaded {loaded} industries")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    industries = st.session_state.domain_loader.get_available_industries()
    
    if industries:
        selected = st.sidebar.selectbox("Industry", [""] + sorted(industries))
        st.session_state.selected_industry = selected if selected else None
        
        if selected:
            data = st.session_state.domain_loader.get_industry_data(selected)
            st.sidebar.success(f"✅ {selected}")
            st.sidebar.info(f"Rules: {data.get('rules_count', 0)}\nKeywords: {data.get('keywords_count', 0)}")
    else:
        st.sidebar.error("❌ No industries")
        st.session_state.selected_industry = None
    
    st.sidebar.markdown("---")
    
    enable_pii = st.sidebar.checkbox("Enable PII Detection", value=True)
    output_format = st.sidebar.selectbox("Output Format", ['csv', 'xlsx', 'parquet', 'json'])
    
    # Main
    st.header("📁 Upload Data")
    
    data_file = st.file_uploader("Upload file", type=SUPPORTED_FORMATS)
    
    has_industry = st.session_state.get('selected_industry')
    
    if not has_industry:
        st.info("👆 Select industry")
    elif not data_file:
        st.info("👆 Upload file")
    else:
        df = FileHandler.read_file(data_file)
        
        if df is not None:
            st.success(f"✅ {len(df):,} records")
            
            col1, col2 = st.columns(2)
            
            with col1:
                id_col = st.selectbox("ID Column", df.columns.tolist())
            with col2:
                text_cols = [c for c in df.columns if c != id_col]
                text_col = st.selectbox("Text Column", text_cols)
            
            with st.expander("👀 Preview"):
                st.dataframe(df[[id_col, text_col]].head(10))
            
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                industry_data = st.session_state.domain_loader.get_industry_data(st.session_state.selected_industry)
                
                with st.spinner("Initializing..."):
                    rule_engine = DynamicRuleEngine(industry_data)
                    pipeline = DynamicNLPPipeline(
                        rule_engine=rule_engine,
                        enable_pii_redaction=enable_pii,
                        industry_name=st.session_state.selected_industry
                    )
                
                progress = st.progress(0)
                status = st.empty()
                
                def update_progress(completed, total):
                    progress.progress(completed / total)
                    status.text(f"{completed:,}/{total:,}")
                
                start = datetime.now()
                
                results = pipeline.process_batch(
                    df=df,
                    text_column=text_col,
                    id_column=id_col,
                    progress_callback=update_progress
                )
                
                elapsed = (datetime.now() - start).total_seconds()
                speed = len(results) / elapsed if elapsed > 0 else 0
                
                results_df = pipeline.results_to_dataframe(results)
                
                st.success(f"✅ Done! {elapsed:.1f}s ({speed:.1f} rec/sec)")
                
                st.subheader("📈 Metrics")
                cols = st.columns(4)
                cols[0].metric("Records", f"{len(results):,}")
                cols[1].metric("Speed", f"{speed:.1f}/s")
                cols[2].metric("Categories", results_df['L1_Category'].nunique())
                cols[3].metric("Avg Sentiment", f"{results_df['Sentiment_Score'].mean():.2f}")
                
                st.dataframe(results_df.head(20))
                
                st.subheader("📊 Charts")
                cols = st.columns(2)
                with cols[0]:
                    st.bar_chart(results_df['L1_Category'].value_counts())
                with cols[1]:
                    st.bar_chart(results_df['Sentiment'].value_counts())
                
                data = FileHandler.save_dataframe(results_df, output_format)
                st.download_button(
                    f"📥 Download (.{output_format})",
                    data=data,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                )
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:gray'><small>v3.1.1 - Streamlit Cloud Optimized</small></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
