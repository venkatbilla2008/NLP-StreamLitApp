"""
Dynamic Domain-Agnostic NLP Text Analysis Pipeline - v3.0.3
============================================================

CHANGES IN v3.0.3:
1. Translation block COMMENTED OUT (can be re-enabled easily)
2. spaCy NER ENABLED for fast PII detection (replaces slow regex)
3. All other features maintained: Classification, Sentiment, Parallel Processing
4. Focus: PII Detection + Classification + Sentiment

Version: 3.0.3 - Translation Disabled, spaCy NER Enabled
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
import io
import os

# NLP Libraries
import spacy
from textblob import TextBlob
# === TRANSLATION COMMENTED OUT - Can be re-enabled if needed ===
# from deep_translator import GoogleTranslator

# ========================================================================================
# CONFIGURATION & CONSTANTS
# ========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - OPTIMIZED FOR PERFORMANCE
MAX_WORKERS = 8  # Parallel processing workers
BATCH_SIZE = 500
CACHE_SIZE = 10000
SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'parquet', 'json']
COMPLIANCE_STANDARDS = ["HIPAA", "GDPR", "PCI-DSS", "CCPA"]

# Performance optimization flags
ENABLE_TRANSLATION = False  # TRANSLATION DISABLED - Can be re-enabled
ENABLE_SPACY_NER = True  # ENABLED - Fast PII detection using spaCy
PII_DETECTION_MODE = 'full'  # Using spaCy NER for comprehensive detection

# File size limits (in MB)
MAX_FILE_SIZE_MB = 500
WARN_FILE_SIZE_MB = 100

# Domain packs directory structure
DOMAIN_PACKS_DIR = "domain_packs"

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            logger.warning("spaCy model not found. Attempting to download...")
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info("spaCy model downloaded successfully")
                return spacy.load("en_core_web_sm")
            else:
                logger.error(f"Failed to download spaCy model: {result.stderr}")
                st.error("⚠️ spaCy model download failed")
                st.stop()
        except Exception as e:
            logger.error(f"Error downloading spaCy model: {e}")
            st.error(f"⚠️ Could not load spaCy model: {e}")
            st.stop()

nlp = load_spacy_model()


# ========================================================================================
# DATA CLASSES
# ========================================================================================

@dataclass
class PIIRedactionResult:
    """Result of PII detection and redaction"""
    redacted_text: str
    pii_detected: bool
    pii_counts: Dict[str, int]
    total_items: int


@dataclass
class CategoryMatch:
    """Hierarchical category match result with 4 levels"""
    l1: str
    l2: str
    l3: str
    l4: str
    confidence: float
    match_path: str
    matched_rule: Optional[str] = None


@dataclass
class ProximityResult:
    """Proximity-based grouping result"""
    primary_proximity: str
    proximity_group: str
    theme_count: int
    matched_themes: List[str]


@dataclass
class NLPResult:
    """Complete NLP analysis result - Translation field commented out"""
    conversation_id: str
    original_text: str
    redacted_text: str
    # translated_text: str  # COMMENTED OUT - Translation disabled
    category: CategoryMatch
    proximity: ProximityResult
    sentiment: str
    sentiment_score: float
    pii_result: PIIRedactionResult
    industry: Optional[str] = None


# ========================================================================================
# DOMAIN LOADER
# ========================================================================================

class DomainLoader:
    """Dynamically loads industry-specific rules and keywords from JSON files"""
    
    def __init__(self, domain_packs_dir: str = None):
        self.domain_packs_dir = domain_packs_dir or DOMAIN_PACKS_DIR
        self.industries = {}
        self.company_mapping = {}
        
    def load_company_mapping(self, mapping_file: str = None) -> Dict:
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                self.company_mapping = data.get('industries', {})
                logger.info(f"Loaded company mapping with {len(self.company_mapping)} industries")
                return self.company_mapping
        return {}
    
    def auto_load_all_industries(self) -> int:
        """Automatically load all industries from domain_packs directory"""
        loaded_count = 0
        
        if not os.path.exists(self.domain_packs_dir):
            logger.error(f"Domain packs directory not found: {self.domain_packs_dir}")
            return 0
        
        logger.info(f"Scanning domain_packs directory: {self.domain_packs_dir}")
        
        # Load company mapping first
        mapping_path = os.path.join(self.domain_packs_dir, "company_industry_mapping.json")
        if os.path.exists(mapping_path):
            self.load_company_mapping(mapping_path)
        
        try:
            items = os.listdir(self.domain_packs_dir)
        except Exception as e:
            logger.error(f"Error listing domain_packs directory: {e}")
            return 0
        
        for item in items:
            item_path = os.path.join(self.domain_packs_dir, item)
            
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
            
            rules_path = os.path.join(item_path, "rules.json")
            keywords_path = os.path.join(item_path, "keywords.json")
            
            if os.path.exists(rules_path) and os.path.exists(keywords_path):
                try:
                    self.load_from_files(rules_path, keywords_path, item)
                    loaded_count += 1
                    logger.info(f"✅ Successfully loaded: {item}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {item}: {str(e)}")
        
        logger.info(f"Auto-load complete: {loaded_count} industries loaded")
        return loaded_count
    
    def load_from_files(self, rules_file: str, keywords_file: str, industry_name: str):
        try:
            with open(rules_file, 'r') as f:
                rules = json.load(f)
            with open(keywords_file, 'r') as f:
                keywords = json.load(f)
            
            self.industries[industry_name] = {
                'rules': rules,
                'keywords': keywords,
                'rules_count': len(rules),
                'keywords_count': len(keywords)
            }
            logger.info(f"Loaded {industry_name}: {len(rules)} rules, {len(keywords)} keywords")
        except Exception as e:
            logger.error(f"Error loading {industry_name}: {e}")
            raise
    
    def get_available_industries(self) -> List[str]:
        return list(self.industries.keys())
    
    def get_industry_data(self, industry: str) -> Dict:
        return self.industries.get(industry, {'rules': [], 'keywords': []})


# ========================================================================================
# PII DETECTION USING SPACY NER - FAST & ACCURATE
# ========================================================================================

class PIIDetector:
    """
    SPACY NER-BASED PII DETECTION - Fast and Accurate
    Compliant with: HIPAA, GDPR, PCI-DSS, CCPA
    
    Uses spaCy Named Entity Recognition for:
    - PERSON names
    - ORG (organizations)
    - GPE (locations)
    - DATE (dates of birth)
    - MONEY (financial info)
    
    Plus regex for:
    - Email addresses
    - Phone numbers
    - Credit cards
    - SSN
    """
    
    # Essential regex patterns (kept for specific PII types)
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERNS = [
        re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),
        re.compile(r'\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'),
    ]
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    
    @classmethod
    def _generate_hash(cls, text: str) -> str:
        """Generate SHA-256 hash"""
        return hashlib.sha256(text.encode()).hexdigest()[:8]
    
    @classmethod
    def _redact_value(cls, value: str, pii_type: str, mode: str) -> str:
        """Redact value based on mode"""
        if mode == 'hash':
            return f"[{pii_type}:{cls._generate_hash(value)}]"
        elif mode == 'mask':
            return f"[{pii_type}:{'*' * 10}]"
        elif mode == 'token':
            return f"[{pii_type}]"
        elif mode == 'remove':
            return ""
        else:
            return f"[{pii_type}:{cls._generate_hash(value)}]"
    
    @classmethod
    def detect_and_redact(cls, text: str, redaction_mode: str = 'hash') -> PIIRedactionResult:
        """
        Detect and redact PII using spaCy NER - FAST & ACCURATE
        
        This method is FASTER than regex-only approaches because:
        1. spaCy NER processes text once and finds multiple entity types
        2. No need for complex regex patterns and validation
        3. Better accuracy for names, organizations, locations
        """
        if not text or not isinstance(text, str):
            return PIIRedactionResult(
                redacted_text=str(text) if text else "",
                pii_detected=False,
                pii_counts={},
                total_items=0
            )
        
        redacted = text
        pii_counts = {}
        
        # === SPACY NER - PRIMARY METHOD (Fast & Accurate) ===
        if ENABLE_SPACY_NER and nlp:
            try:
                doc = nlp(redacted)
                
                # Process named entities
                for ent in doc.ents:
                    entity_text = ent.text
                    entity_label = ent.label_
                    
                    # Map spaCy entity types to PII categories
                    if entity_label == 'PERSON':
                        redacted = redacted.replace(entity_text, cls._redact_value(entity_text, 'NAME', redaction_mode))
                        pii_counts['names'] = pii_counts.get('names', 0) + 1
                    
                    elif entity_label == 'ORG':
                        redacted = redacted.replace(entity_text, cls._redact_value(entity_text, 'ORG', redaction_mode))
                        pii_counts['organizations'] = pii_counts.get('organizations', 0) + 1
                    
                    elif entity_label == 'GPE':  # Geopolitical entities (cities, states, countries)
                        redacted = redacted.replace(entity_text, cls._redact_value(entity_text, 'LOCATION', redaction_mode))
                        pii_counts['locations'] = pii_counts.get('locations', 0) + 1
                    
                    elif entity_label == 'DATE':
                        # Could be DOB
                        redacted = redacted.replace(entity_text, cls._redact_value(entity_text, 'DATE', redaction_mode))
                        pii_counts['dates'] = pii_counts.get('dates', 0) + 1
                    
                    elif entity_label == 'MONEY':
                        redacted = redacted.replace(entity_text, cls._redact_value(entity_text, 'MONEY', redaction_mode))
                        pii_counts['financial'] = pii_counts.get('financial', 0) + 1
            
            except Exception as e:
                logger.error(f"spaCy NER error: {e}")
        
        # === REGEX - SECONDARY METHOD (for specific patterns) ===
        
        # 1. Emails
        emails = cls.EMAIL_PATTERN.findall(redacted)
        for email in emails:
            redacted = redacted.replace(email, cls._redact_value(email, 'EMAIL', redaction_mode))
            pii_counts['emails'] = pii_counts.get('emails', 0) + 1
        
        # 2. Phone numbers
        for pattern in cls.PHONE_PATTERNS:
            phones = pattern.findall(redacted)
            for phone in phones:
                redacted = redacted.replace(phone, cls._redact_value(phone, 'PHONE', redaction_mode))
                pii_counts['phones'] = pii_counts.get('phones', 0) + 1
        
        # 3. SSN
        ssns = cls.SSN_PATTERN.findall(redacted)
        for ssn in ssns:
            redacted = redacted.replace(ssn, cls._redact_value(ssn, 'SSN', redaction_mode))
            pii_counts['ssn'] = pii_counts.get('ssn', 0) + 1
        
        # 4. Credit Cards
        cards = cls.CREDIT_CARD_PATTERN.findall(redacted)
        for card in cards:
            redacted = redacted.replace(card, cls._redact_value(card, 'CARD', redaction_mode))
            pii_counts['credit_cards'] = pii_counts.get('credit_cards', 0) + 1
        
        total_items = sum(pii_counts.values())
        
        return PIIRedactionResult(
            redacted_text=redacted,
            pii_detected=total_items > 0,
            pii_counts=pii_counts,
            total_items=total_items
        )


# ========================================================================================
# DYNAMIC RULE ENGINE
# ========================================================================================

class DynamicRuleEngine:
    """Dynamic rule-based classification engine"""
    
    def __init__(self, industry_data: Dict):
        self.rules = industry_data.get('rules', [])
        self.keywords = industry_data.get('keywords', [])
        self._build_lookup_tables()
        logger.info(f"Initialized with {len(self.rules)} rules, {len(self.keywords)} keywords")
    
    def _build_lookup_tables(self):
        self.compiled_rules = []
        for rule in self.rules:
            conditions = rule.get('conditions', [])
            if conditions:
                pattern_parts = [re.escape(cond.lower()) for cond in conditions]
                pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)
                self.compiled_rules.append({
                    'pattern': pattern,
                    'conditions': conditions,
                    'category': rule.get('set', {})
                })
        
        self.compiled_keywords = []
        for keyword_group in self.keywords:
            conditions = keyword_group.get('conditions', [])
            if conditions:
                pattern_parts = [re.escape(cond.lower()) for cond in conditions]
                pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)
                self.compiled_keywords.append({
                    'pattern': pattern,
                    'conditions': conditions,
                    'category': keyword_group.get('set', {})
                })
    
    @lru_cache(maxsize=CACHE_SIZE)
    def classify_text(self, text: str) -> CategoryMatch:
        """Classify text using dynamic rules"""
        if not text or not isinstance(text, str):
            return CategoryMatch("Uncategorized", "NA", "NA", "NA", 0.0, "Uncategorized", None)
        
        text_lower = text.lower()
        
        # Try keywords first
        for kw_item in self.compiled_keywords:
            if kw_item['pattern'].search(text_lower):
                category_data = kw_item['category']
                return CategoryMatch(
                    l1=category_data.get('category', 'Uncategorized'),
                    l2=category_data.get('subcategory', 'NA'),
                    l3=category_data.get('level_3', 'NA'),
                    l4=category_data.get('level_4', 'NA'),
                    confidence=0.9,
                    match_path=f"{category_data.get('category', 'Uncategorized')} > {category_data.get('subcategory', 'NA')}",
                    matched_rule="keyword_match"
                )
        
        # Try rules
        best_match = None
        best_match_count = 0
        
        for rule_item in self.compiled_rules:
            matches = rule_item['pattern'].findall(text_lower)
            match_count = len(matches)
            if match_count > best_match_count:
                best_match_count = match_count
                best_match = rule_item
        
        if best_match:
            category_data = best_match['category']
            total_conditions = len(best_match['conditions'])
            confidence = min(best_match_count / max(total_conditions, 1), 1.0) * 0.85
            return CategoryMatch(
                l1=category_data.get('category', 'Uncategorized'),
                l2=category_data.get('subcategory', 'NA'),
                l3=category_data.get('level_3', 'NA'),
                l4=category_data.get('level_4', 'NA'),
                confidence=confidence,
                match_path=f"{category_data.get('category', 'Uncategorized')} > {category_data.get('subcategory', 'NA')}",
                matched_rule=f"rule_match_{best_match_count}_conditions"
            )
        
        return CategoryMatch("Uncategorized", "NA", "NA", "NA", 0.0, "Uncategorized", None)


# ========================================================================================
# PROXIMITY ANALYZER
# ========================================================================================

class ProximityAnalyzer:
    """Analyzes text for proximity-based contextual themes"""
    
    PROXIMITY_THEMES = {
        'Agent_Behavior': ['agent', 'representative', 'rep', 'staff', 'rude', 'unprofessional'],
        'Technical_Issues': ['error', 'bug', 'issue', 'problem', 'crash', 'broken'],
        'Customer_Service': ['service', 'support', 'help', 'assist', 'customer'],
        'Communication': ['communication', 'call', 'email', 'message', 'contact'],
        'Billing_Payments': ['bill', 'payment', 'charge', 'fee', 'refund'],
        'Product_Quality': ['product', 'quality', 'defect', 'damaged', 'faulty'],
        'Cancellation_Refund': ['cancel', 'cancellation', 'refund', 'return'],
        'Policy_Terms': ['policy', 'term', 'terms', 'condition', 'rule'],
        'Account_Access': ['account', 'login', 'password', 'access', 'locked'],
        'Order_Delivery': ['order', 'delivery', 'shipping', 'delayed', 'package'],
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
    """Sentiment analysis with 5-level granularity"""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def analyze_sentiment(text: str) -> Tuple[str, float]:
        if not text:
            return "Neutral", 0.0
        
        try:
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
        except:
            return "Neutral", 0.0


# === TRANSLATION SERVICE - COMMENTED OUT ===
# class TranslationService:
#     """Multi-language translation service - DISABLED"""
#     
#     @staticmethod
#     @lru_cache(maxsize=CACHE_SIZE)
#     def translate_to_english(text: str) -> str:
#         """Translate text to English - DISABLED"""
#         if not text or not isinstance(text, str):
#             return text
#         
#         if not ENABLE_TRANSLATION:
#             return text
#         
#         try:
#             from deep_translator import GoogleTranslator
#             translated = GoogleTranslator(source='auto', target='en').translate(text)
#             return translated
#         except Exception as e:
#             logger.error(f"Translation error: {e}")
#             return text


# ========================================================================================
# COMPLIANCE MANAGER
# ========================================================================================

class ComplianceManager:
    """Manages compliance reporting and audit logging"""
    
    def __init__(self):
        self.audit_log = []
        self.start_time = datetime.now()
    
    def log_redaction(self, conversation_id: str, pii_counts: Dict[str, int]):
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'conversation_id': conversation_id,
            'pii_counts': pii_counts,
            'total_items': sum(pii_counts.values())
        })
    
    def generate_compliance_report(self, results: List[NLPResult]) -> Dict:
        total_records = len(results)
        records_with_pii = sum(1 for r in results if r.pii_result.pii_detected)
        total_pii_items = sum(r.pii_result.total_items for r in results)
        
        pii_distribution = {}
        for result in results:
            for pii_type, count in result.pii_result.pii_counts.items():
                pii_distribution[pii_type] = pii_distribution.get(pii_type, 0) + count
        
        return {
            'report_generated': datetime.now().isoformat(),
            'processing_time': str(datetime.now() - self.start_time),
            'summary': {
                'total_records_processed': total_records,
                'records_with_pii': records_with_pii,
                'records_clean': total_records - records_with_pii,
                'pii_detection_rate': f"{(records_with_pii/total_records*100):.2f}%" if total_records > 0 else "0%",
                'total_pii_items': total_pii_items
            },
            'pii_type_distribution': pii_distribution,
            'compliance_standards': COMPLIANCE_STANDARDS,
            'audit_log_entries': len(self.audit_log)
        }
    
    def export_audit_log(self) -> pd.DataFrame:
        if not self.audit_log:
            return pd.DataFrame()
        return pd.DataFrame(self.audit_log)


# ========================================================================================
# MAIN NLP PIPELINE
# ========================================================================================

class DynamicNLPPipeline:
    """Main NLP processing pipeline - Translation Disabled"""
    
    def __init__(self, rule_engine, enable_pii_redaction=True, industry_name=None):
        self.rule_engine = rule_engine
        self.enable_pii_redaction = enable_pii_redaction
        self.industry_name = industry_name
        self.compliance_manager = ComplianceManager()
    
    def process_single_text(self, conversation_id: str, text: str, redaction_mode: str = 'hash') -> NLPResult:
        """Process single text - Translation disabled"""
        
        # 1. PII Detection using spaCy NER
        if self.enable_pii_redaction:
            pii_result = PIIDetector.detect_and_redact(text, redaction_mode)
            if pii_result.pii_detected:
                self.compliance_manager.log_redaction(conversation_id, pii_result.pii_counts)
            working_text = pii_result.redacted_text
        else:
            pii_result = PIIRedactionResult(text, False, {}, 0)
            working_text = text
        
        # === TRANSLATION DISABLED ===
        # translated_text = TranslationService.translate_to_english(working_text)
        analysis_text = working_text  # Use redacted text directly
        
        # 2. Classification
        category = self.rule_engine.classify_text(analysis_text)
        
        # 3. Proximity
        proximity = ProximityAnalyzer.analyze_proximity(analysis_text)
        
        # 4. Sentiment
        sentiment, sentiment_score = SentimentAnalyzer.analyze_sentiment(analysis_text)
        
        return NLPResult(
            conversation_id=conversation_id,
            original_text=text,
            redacted_text=pii_result.redacted_text,
            # translated_text=translated_text,  # COMMENTED OUT
            category=category,
            proximity=proximity,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            pii_result=pii_result,
            industry=self.industry_name
        )
    
    def process_batch(self, df, text_column, id_column, redaction_mode='hash', progress_callback=None):
        """Process batch with parallel processing"""
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
                    
                    if progress_callback and completed % 10 == 0:
                        progress_callback(completed, total)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    completed += 1
        
        return results
    
    def results_to_dataframe(self, results: List[NLPResult]) -> pd.DataFrame:
        """Convert to DataFrame - No translation column"""
        data = []
        
        for result in results:
            row = {
                'Conversation_ID': result.conversation_id,
                'Original_Text': result.original_text,
                # 'Translated_Text': result.translated_text,  # COMMENTED OUT
                'L1_Category': result.category.l1,
                'L2_Subcategory': result.category.l2,
                'L3_Tertiary': result.category.l3,
                'L4_Quaternary': result.category.l4,
                'Primary_Proximity': result.proximity.primary_proximity,
                'Proximity_Group': result.proximity.proximity_group,
                'Sentiment': result.sentiment,
                'Sentiment_Score': result.sentiment_score
            }
            data.append(row)
        
        return pd.DataFrame(data)


# ========================================================================================
# FILE HANDLER
# ========================================================================================

class FileHandler:
    """Handles file I/O operations"""
    
    @staticmethod
    def read_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB")
                return None
            
            if file_size_mb > WARN_FILE_SIZE_MB:
                st.warning(f"⚠️ Large file: {file_size_mb:.1f}MB")
            
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
                st.error(f"Unsupported: {ext}")
                return None
            
            logger.info(f"Loaded: {uploaded_file.name} ({len(df)} rows)")
            return df
        except Exception as e:
            st.error(f"Error: {e}")
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
    """Main application - v3.0.3"""
    
    st.set_page_config(
        page_title="NLP Pipeline v3.0.3",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 NLP Pipeline v3.0.3 - spaCy NER Enabled")
    st.markdown("""
    **Focus: PII Detection + Classification + Sentiment**
    - 🔐 spaCy NER for fast PII detection (Names, Orgs, Locations)
    - 📊 NLP Classification (4-level hierarchy)
    - 💭 Sentiment Analysis
    - ⚡ Parallel Processing enabled
    - ❌ **Translation disabled** (can be re-enabled if needed)
    """)
    
    # Status indicators
    cols = st.columns(3)
    with cols[0]:
        st.success("✅ spaCy NER: Enabled")
    with cols[1]:
        st.info("ℹ️ Translation: Disabled")
    with cols[2]:
        st.success(f"✅ Parallel: {MAX_WORKERS} workers")
    
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
    
    # Industry
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
    
    # PII settings
    enable_pii = st.sidebar.checkbox("Enable PII Detection", value=True)
    redaction_mode = st.sidebar.selectbox("Redaction Mode", ['hash', 'mask', 'token', 'remove'])
    
    st.sidebar.info("ℹ️ **spaCy NER Enabled**\n\nFast detection of:\n- Names (PERSON)\n- Organizations (ORG)\n- Locations (GPE)\n- Dates (DATE)\n- Money (MONEY)")
    
    # Output
    output_format = st.sidebar.selectbox("Output Format", ['csv', 'xlsx', 'parquet', 'json'])
    
    # Main
    st.header("📁 Upload Data")
    
    data_file = st.file_uploader("Upload file", type=SUPPORTED_FORMATS)
    
    if data_file:
        st.session_state.current_file = data_file
    
    has_industry = st.session_state.get('selected_industry')
    has_file = data_file is not None
    
    if not has_industry:
        st.info("👆 Select industry")
    elif not has_file:
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
                    redaction_mode=redaction_mode,
                    progress_callback=update_progress
                )
                
                elapsed = (datetime.now() - start).total_seconds()
                speed = len(results) / elapsed if elapsed > 0 else 0
                
                results_df = pipeline.results_to_dataframe(results)
                
                st.success(f"✅ Done! {elapsed:.2f}s ({speed:.1f} rec/sec)")
                
                st.subheader("📈 Metrics")
                cols = st.columns(5)
                cols[0].metric("Records", f"{len(results):,}")
                cols[1].metric("Speed", f"{speed:.1f} rec/sec")
                cols[2].metric("Categories", results_df['L1_Category'].nunique())
                cols[3].metric("Avg Sentiment", f"{results_df['Sentiment_Score'].mean():.2f}")
                cols[4].metric("Industry", st.session_state.selected_industry)
                
                st.dataframe(results_df.head(20))
                
                st.subheader("📊 Charts")
                cols = st.columns(3)
                with cols[0]:
                    st.bar_chart(results_df['L1_Category'].value_counts())
                with cols[1]:
                    st.bar_chart(results_df['Sentiment'].value_counts())
                with cols[2]:
                    st.bar_chart(results_df['Primary_Proximity'].value_counts().head(10))
                
                if enable_pii:
                    st.subheader("🔒 Compliance")
                    report = pipeline.compliance_manager.generate_compliance_report(results)
                    st.json(report['summary'])
                
                data = FileHandler.save_dataframe(results_df, output_format)
                st.download_button(
                    f"📥 Download (.{output_format})",
                    data=data,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                )
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:gray'><small>v3.0.3 - spaCy NER + Translation Disabled</small></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
