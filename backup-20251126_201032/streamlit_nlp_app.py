"""
🎯 NLP Text Classification Dashboard - Streamlit Version
Analyzes transcripts for sentiment, category, and subcategory classification
Outputs results in Parquet format for efficient storage
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import io

# NLP Libraries
from textblob import TextBlob
from afinn import Afinn
from langdetect import detect, DetectorFactory, LangDetectException

warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

# ============================================================
# 🔧 NLTK Data Setup (for Streamlit Cloud)
# ============================================================
import nltk
import ssl

# Handle SSL certificate verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data on first run (cached)"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown', quiet=True)
    return True

# Download NLTK data (runs once, then cached)
download_nltk_data()

# ============================================================
# 📊 Configuration
# ============================================================
class Config:
    """Application configuration settings"""
    # Reduce threads for Streamlit Cloud free tier (1 CPU core)
    NUM_THREADS = min(4, os.cpu_count() or 4)
    MAX_FILE_SIZE_MB = 100
    MAX_ROWS = 50000
    SENTIMENT_WEIGHTS = {'textblob': 0.6, 'afinn': 0.4}
    SENTIMENT_THRESHOLDS = {
        'very_negative': -0.75,
        'negative': -0.25,
        'positive': 0.25,
        'very_positive': 0.75
    }
    MAX_PREVIEW_ROWS = 100
    OUTPUT_DIR = "nlp_results"
    MIN_TEXT_LENGTH = 3
    CHUNK_SIZE = 1000  # For progress updates


# ============================================================
# 🔑 Category and Subcategory Keywords
# ============================================================
TOPIC_KEYWORDS = {
    "login issue": [
        "login", "log in", "sign in", "sign-in", "signin", "sign out", "sign-out", "signout",
        "password", "forgot password", "reset password", "authentication",
        "verify account", "verification code", "2fa", "two-factor", "two factor",
        "unable to access account", "can't log in", "cannot login"
    ],
    "account issue": [
        "account", "profile", "username", "display name",
        "linked account", "merge account", "multiple accounts",
        "email change", "update details", "account disabled",
        "account locked", "deactivate account", "delete account"
    ],
    "playback issue": [
        "playback", "stream", "music not playing", "song not playing",
        "track skipped", "buffering", "lag", "pause", "stuck",
        "stops suddenly", "won't play", "audio issue", "no sound",
        "silence", "volume problem", "audio quality"
    ],
    "device issue": [
        "bluetooth", "speaker", "carplay", "android auto", "smart tv",
        "echo", "alexa", "chromecast", "airplay", "headphones",
        "device not showing", "device disconnected", "connection issue"
    ],
    "content restriction": [
        "song not available", "track unavailable", "region restriction",
        "country restriction", "not licensed", "greyed out", "removed song",
        "can't find song", "missing track"
    ],
    "ad issue": [
        "ads", "advertisement", "too many ads", "ad volume",
        "ad playing", "premium ads", "commercials", "ad frequency"
    ],
    "recommendation issue": [
        "recommendations", "discover weekly", "radio", "algorithm",
        "curated", "autoplay", "song suggestions", "not relevant",
        "bad recommendations"
    ],
    "ui issue": [
        "interface", "layout", "design", "dark mode", "theme",
        "buttons not working", "search not working", "filter not working",
        "navigation", "menu"
    ],
    "general feedback": [
        "suggestion", "feedback", "recommend", "love spotify",
        "like app", "app improvement", "feature request", "enhancement"
    ],
    "network failure": [
        "network", "connectivity", "internet", "server",
        "connection failed", "offline", "not connecting",
        "spotify down", "timeout", "dns", "proxy", "vpn"
    ],
    "app crash": [
        "crash", "crashed", "app closed", "stopped working", "freeze",
        "freezing", "hang", "bug", "error message", "glitch",
        "unresponsive", "not responding"
    ],
    "performance issue": [
        "slow", "lag", "delay", "performance", "loading", "slow loading",
        "takes forever", "laggy"
    ],
    "data sync issue": [
        "sync", "not syncing", "listening history", "recently played",
        "activity feed", "spotify connect", "data lost", "missing data",
        "playlist not syncing"
    ],
    "subscription issue": [
        "subscription", "plan", "premium", "cancel", "renew",
        "billing", "charged", "payment", "refund", "invoice",
        "upgrade", "downgrade", "free trial", "family plan",
        "student plan", "gift card", "promo code", "spotify wrapped",
        "card", "payment failed"
    ],
}

SUBCATEGORY_KEYWORDS = {
    "subscription issue": {
        "payment": ["refund", "charged", "billing", "invoice", "payment", "payment failed", "card declined"],
        "cancel": ["cancel", "unsubscribe", "stop subscription", "end subscription"],
        "upgrade": ["upgrade", "family plan", "student plan", "premium", "switch plan"],
    },
    "account issue": {
        "login": ["login", "password", "signin", "sign in", "authentication"],
        "profile": ["profile", "email", "username", "display name", "account settings"],
    },
    "device issue": {
        "mobile": ["phone", "android", "iphone", "ios", "mobile app"],
        "car": ["carplay", "android auto", "car", "vehicle"],
        "smart_device": ["alexa", "echo", "chromecast", "smart tv", "airplay"],
    },
}

# Pre-compile regex patterns for better performance
CONSUMER_PATTERN_PRIMARY = re.compile(
    r"(?i)Consumer:\s*(.*?)(?=\s*\|\s*\d{4}-\d{2}-\d{2}|$|\s*\|\s*Agent:)",
    re.IGNORECASE
)
CONSUMER_PATTERN_FALLBACK = re.compile(
    r"(?i)Consumer:\s*(.*?)(?=\||$)",
    re.IGNORECASE
)


# ============================================================
# 🛠️ Helper Functions
# ============================================================

def is_english(text: str) -> bool:
    """
    Detect if text is English with proper error handling.
    
    Args:
        text: Input text to check
        
    Returns:
        bool: True if text is English, False otherwise
    """
    if not text or len(text.strip()) < Config.MIN_TEXT_LENGTH:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        # Too short or unclear language
        return False
    except Exception as e:
        # Log unexpected errors
        st.warning(f"Language detection error: {e}")
        return False


def extract_consumer_text(transcript: str) -> str:
    """
    Extract consumer text from transcript using multiple regex patterns.
    
    Args:
        transcript: Full conversation transcript
        
    Returns:
        str: Extracted consumer text
    """
    if not isinstance(transcript, str):
        return ""
    
    # Try primary pattern
    parts = CONSUMER_PATTERN_PRIMARY.findall(transcript + " ")
    
    # Fallback to simpler pattern
    if not parts:
        parts = CONSUMER_PATTERN_FALLBACK.findall(transcript + "|")
    
    return " ".join(p.strip() for p in parts if p.strip())


def hybrid_sentiment(text: str, af: Afinn) -> str:
    """
    Perform hybrid sentiment analysis using TextBlob and AFINN.
    
    Args:
        text: Input text to analyze
        af: AFINN analyzer instance
        
    Returns:
        str: Sentiment classification
    """
    if not text or not is_english(text):
        return ""
    
    # Check minimum length
    if len(text.split()) < Config.MIN_TEXT_LENGTH:
        return "neutral"
    
    # TextBlob sentiment
    tb_score = TextBlob(text).sentiment.polarity
    
    # AFINN sentiment (normalized)
    af_score = af.score(text) / 5.0
    
    # Weighted combination
    score = (Config.SENTIMENT_WEIGHTS['textblob'] * tb_score + 
             Config.SENTIMENT_WEIGHTS['afinn'] * af_score)
    
    # Classification
    thresholds = Config.SENTIMENT_THRESHOLDS
    if score <= thresholds['very_negative']:
        return "very negative"
    elif score <= thresholds['negative']:
        return "negative"
    elif score >= thresholds['very_positive']:
        return "very positive"
    elif score >= thresholds['positive']:
        return "positive"
    else:
        return "neutral"


def predict_category(text: str) -> Tuple[str, int]:
    """
    Predict category with improved word boundary matching and confidence score.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (category, confidence_score)
    """
    text_lower = text.lower()
    best_match, best_score = "", 0
    
    for category, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Use word boundaries for short keywords to avoid false matches
            if len(keyword.split()) == 1 and len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    score += 1
            else:
                if keyword in text_lower:
                    # Weight multi-word phrases higher
                    score += 1.5 if ' ' in keyword else 1
        
        if score > best_score:
            best_score = score
            best_match = category
    
    return (best_match if best_score > 0 else "", int(best_score))


def predict_subcategory(category: str, text: str) -> Tuple[str, int]:
    """
    Predict subcategory based on category and text.
    
    Args:
        category: Predicted main category
        text: Input text
        
    Returns:
        Tuple of (subcategory, confidence_score)
    """
    if not category or category not in SUBCATEGORY_KEYWORDS:
        return ("", 0)
    
    text_lower = text.lower()
    best_match, best_score = "", 0
    
    for subcategory, keywords in SUBCATEGORY_KEYWORDS[category].items():
        score = 0
        for keyword in keywords:
            if len(keyword.split()) == 1 and len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    score += 1
            else:
                if keyword in text_lower:
                    score += 1.5 if ' ' in keyword else 1
        
        if score > best_score:
            best_score = score
            best_match = subcategory
    
    return (best_match, int(best_score))


def apply_rules(text: str, preds: Dict) -> Dict:
    """
    Apply rule-based overrides for special cases.
    
    Args:
        text: Input text
        preds: Current predictions dictionary
        
    Returns:
        Updated predictions dictionary
    """
    text_lower = text.lower()
    
    # Rule 1: Payment/refund issues
    if any(k in text_lower for k in ["refund", "charged", "billing", "payment failed"]):
        preds["category"] = "subscription issue"
        preds["subcategory"] = "payment"
        if "refund" in text_lower or "charged" in text_lower:
            preds["sentiment"] = "negative"
    
    # Rule 2: Cancellation
    elif "cancel" in text_lower and "subscription" in text_lower:
        preds["category"] = "subscription issue"
        preds["subcategory"] = "cancel"
    
    return preds


# ============================================================
# 🔄 Core Processing Function
# ============================================================

def process_row(row: Dict, af: Afinn) -> Dict:
    """
    Process a single transcript row and classify it.
    
    Args:
        row: Dictionary containing 'Conversation Id' and 'transcripts'
        af: AFINN analyzer instance
        
    Returns:
        dict: Processed row with classification fields
    """
    conversation_id = row.get("Conversation Id", "")
    transcript = str(row.get("transcripts", ""))
    consumer_text = extract_consumer_text(transcript)

    # Handle empty or non-English text
    if not consumer_text.strip() or not is_english(consumer_text):
        return {
            "Conversation Id": conversation_id,
            "Consumer_Text": consumer_text,
            "Category": "",
            "Subcategory": "",
            "Sentiment": "",
            "Category_Confidence": 0,
            "Subcategory_Confidence": 0,
            "Needs_Review": True,
        }

    # Predict category and subcategory
    category, cat_confidence = predict_category(consumer_text)
    subcategory, subcat_confidence = predict_subcategory(category, consumer_text)
    sentiment = hybrid_sentiment(consumer_text, af)
    
    # Build predictions dictionary
    preds = {
        "category": category,
        "subcategory": subcategory,
        "sentiment": sentiment,
        "category_confidence": cat_confidence,
        "subcategory_confidence": subcat_confidence,
    }
    
    # Apply rule-based overrides
    preds = apply_rules(consumer_text, preds)

    return {
        "Conversation Id": conversation_id,
        "Consumer_Text": consumer_text,
        "Category": preds["category"],
        "Subcategory": preds["subcategory"],
        "Sentiment": preds["sentiment"],
        "Category_Confidence": preds["category_confidence"],
        "Subcategory_Confidence": preds["subcategory_confidence"],
        "Needs_Review": preds["category_confidence"] < 2,
    }


# ============================================================
# 📊 Data Validation
# ============================================================

def validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate input dataframe structure and content.
    
    Args:
        df: Input dataframe to validate
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    required_cols = ["Conversation Id", "transcripts"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {', '.join(missing_cols)}")
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("❌ Input file is empty")
    
    # Check row count limit
    if len(df) > Config.MAX_ROWS:
        raise ValueError(f"❌ Too many rows: {len(df):,} (max: {Config.MAX_ROWS:,})")
    
    # Check for all-null transcripts
    null_count = df["transcripts"].isnull().sum()
    if null_count == len(df):
        raise ValueError("❌ All transcript values are empty")
    
    # Warn about high null percentage
    if null_count > len(df) * 0.5:
        st.warning(f"⚠️ Warning: {null_count:,}/{len(df):,} ({null_count/len(df)*100:.1f}%) transcripts are empty")


# ============================================================
# 🚀 Main Pipeline Function
# ============================================================

def run_pipeline(df: pd.DataFrame, progress_bar=None, status_text=None) -> pd.DataFrame:
    """
    Run the NLP classification pipeline with parallel processing.
    
    Args:
        df: Input dataframe with transcripts
        progress_bar: Streamlit progress bar object
        status_text: Streamlit status text object
        
    Returns:
        DataFrame with classification results
    """
    start_time = time.time()
    af = Afinn()
    
    # Convert dataframe to list of dictionaries
    rows = df.to_dict("records")
    total_rows = len(rows)
    
    results = []
    processed = 0
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=Config.NUM_THREADS) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_row, row, af): row for row in rows}
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            results.append(future.result())
            processed += 1
            
            # Update progress
            if progress_bar and processed % 10 == 0:
                progress = processed / total_rows
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"Processing: {processed:,}/{total_rows:,} rows ({progress*100:.1f}%)")
    
    # Final progress update
    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        elapsed = time.time() - start_time
        status_text.text(f"✅ Completed! Processed {total_rows:,} rows in {elapsed:.2f}s ({total_rows/elapsed:.1f} rows/sec)")
    
    # Create output dataframe
    out_df = pd.DataFrame(results)
    
    # Add row numbers
    out_df.insert(0, 'Row_Number', range(1, len(out_df) + 1))
    
    return out_df


# ============================================================
# 🎨 Streamlit UI
# ============================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="NLP Text Classification Dashboard",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stDownloadButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🎯 NLP Text Classification Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="File must contain 'Conversation Id' and 'transcripts' columns"
        )
        
        st.markdown("---")
        
        # Settings
        st.subheader("🔧 Settings")
        num_threads = st.slider(
            "Number of threads",
            min_value=1,
            max_value=os.cpu_count() or 4,
            value=Config.NUM_THREADS,
            help="More threads = faster processing (if you have the CPU cores)"
        )
        Config.NUM_THREADS = num_threads
        
        preview_rows = st.slider(
            "Preview rows",
            min_value=10,
            max_value=500,
            value=Config.MAX_PREVIEW_ROWS,
            step=10,
            help="Number of rows to display in preview"
        )
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.info(
            "This app classifies customer service transcripts by:\n"
            "- **Category**: Main issue type\n"
            "- **Subcategory**: Specific issue\n"
            "- **Sentiment**: Customer emotion\n"
            "\n**Output format**: Parquet for efficient storage"
        )
        
        # System info
        with st.expander("🖥️ System Info"):
            st.text(f"Python: {os.sys.version.split()[0]}")
            st.text(f"CPU Cores: {os.cpu_count()}")
            st.text(f"Max File Size: {Config.MAX_FILE_SIZE_MB} MB")
            st.text(f"Max Rows: {Config.MAX_ROWS:,}")
    
    # Main content
    if uploaded_file is None:
        # Welcome screen
        st.info("👈 **Get Started:** Upload a CSV or Excel file using the sidebar")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📝 Step 1")
            st.write("Upload your dataset containing conversation transcripts")
        
        with col2:
            st.markdown("### 🚀 Step 2")
            st.write("Click 'Run Analysis' to process the data")
        
        with col3:
            st.markdown("### 📥 Step 3")
            st.write("Download results in Parquet format")
        
        st.markdown("---")
        
        # Sample data format
        st.subheader("📋 Required Data Format")
        sample_df = pd.DataFrame({
            "Conversation Id": ["CONV_001", "CONV_002"],
            "transcripts": [
                "Consumer: I can't login to my account | Agent: Let me help you",
                "Consumer: Songs are not playing | Agent: I'll check that"
            ]
        })
        st.dataframe(sample_df, use_container_width=True)
        
        # Categories info
        with st.expander("📚 Supported Categories"):
            cols = st.columns(2)
            categories = list(TOPIC_KEYWORDS.keys())
            mid = len(categories) // 2
            
            with cols[0]:
                for cat in categories[:mid]:
                    st.write(f"• {cat}")
            
            with cols[1]:
                for cat in categories[mid:]:
                    st.write(f"• {cat}")
    
    else:
        # File info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Check file size
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)")
            return
        
        # Load data
        try:
            with st.spinner("📖 Reading file..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
            
            # Validate data
            validate_input_dataframe(df)
            
            # Show preview
            with st.expander("👀 Data Preview (First 5 rows)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Run analysis button
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
            
            if run_button:
                # Create output directory
                os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run pipeline
                try:
                    result_df = run_pipeline(df, progress_bar, status_text)
                    
                    # Generate timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save as Parquet
                    parquet_filename = f"sentiment_output_{timestamp}.parquet"
                    parquet_path = os.path.join(Config.OUTPUT_DIR, parquet_filename)
                    result_df.to_parquet(parquet_path, index=False, compression='snappy')
                    
                    # Also save as CSV for compatibility
                    csv_filename = f"sentiment_output_{timestamp}.csv"
                    csv_path = os.path.join(Config.OUTPUT_DIR, csv_filename)
                    result_df.to_csv(csv_path, index=False)
                    
                    st.success("🎉 Analysis Complete!")
                    
                    # Display statistics
                    st.markdown("---")
                    st.subheader("📊 Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Rows", f"{len(result_df):,}")
                    
                    with col2:
                        valid_count = (result_df['Category'] != '').sum()
                        st.metric("Valid Classifications", f"{valid_count:,}")
                    
                    with col3:
                        needs_review = result_df['Needs_Review'].sum()
                        st.metric("Needs Review", f"{needs_review:,}", 
                                 delta=f"{needs_review/len(result_df)*100:.1f}%")
                    
                    with col4:
                        file_size_kb = os.path.getsize(parquet_path) / 1024
                        st.metric("File Size", f"{file_size_kb:.1f} KB")
                    
                    # Category distribution
                    st.markdown("---")
                    st.subheader("📈 Category Distribution")
                    
                    category_counts = result_df[result_df['Category'] != '']['Category'].value_counts()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.bar_chart(category_counts)
                    
                    with col2:
                        st.dataframe(
                            pd.DataFrame({
                                'Category': category_counts.index,
                                'Count': category_counts.values,
                                'Percentage': (category_counts.values / len(result_df) * 100).round(1)
                            }).reset_index(drop=True),
                            use_container_width=True
                        )
                    
                    # Sentiment distribution
                    st.markdown("---")
                    st.subheader("💭 Sentiment Distribution")
                    
                    sentiment_counts = result_df[result_df['Sentiment'] != '']['Sentiment'].value_counts()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create a color-coded bar chart
                        sentiment_df = pd.DataFrame({
                            'Sentiment': sentiment_counts.index,
                            'Count': sentiment_counts.values
                        })
                        st.bar_chart(sentiment_counts)
                    
                    with col2:
                        st.dataframe(
                            pd.DataFrame({
                                'Sentiment': sentiment_counts.index,
                                'Count': sentiment_counts.values,
                                'Percentage': (sentiment_counts.values / len(result_df) * 100).round(1)
                            }).reset_index(drop=True),
                            use_container_width=True
                        )
                    
                    # Results preview
                    st.markdown("---")
                    st.subheader(f"🔍 Results Preview (First {preview_rows} rows)")
                    st.dataframe(result_df.head(preview_rows), use_container_width=True)
                    
                    # Download section
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    # Parquet download
                    with col1:
                        with open(parquet_path, 'rb') as f:
                            st.download_button(
                                label="📦 Download Parquet File",
                                data=f,
                                file_name=parquet_filename,
                                mime="application/octet-stream",
                                use_container_width=True,
                                help="Recommended: Smaller file size, faster loading"
                            )
                    
                    # CSV download
                    with col2:
                        csv_data = result_df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV File",
                            data=csv_data,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True,
                            help="Alternative: Compatible with Excel"
                        )
                    
                    # File info
                    parquet_size = os.path.getsize(parquet_path) / 1024
                    csv_size = os.path.getsize(csv_path) / 1024
                    compression_ratio = (1 - parquet_size / csv_size) * 100
                    
                    st.info(f"💡 **Parquet is {compression_ratio:.1f}% smaller** ({parquet_size:.1f}KB vs {csv_size:.1f}KB)")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
        
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)


# ============================================================
# 🚀 Application Entry Point
# ============================================================

if __name__ == "__main__":
    main()
