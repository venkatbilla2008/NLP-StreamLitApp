# Code Review: Mercury NLP Text Classification Dashboard

## Overall Assessment

**Rating: 7.5/10** - This is a well-structured application with good functionality, but there are several areas for improvement in error handling, code organization, and user experience.

---

## ✅ Strengths

### 1. **Good Code Organization**
- Clear section separation with descriptive headers
- Logical flow from configuration → imports → processing → execution
- Well-structured keyword dictionaries for classification

### 2. **Robust Text Processing**
- Multi-method sentiment analysis (TextBlob + AFINN hybrid)
- Language detection to filter non-English text
- Consumer text extraction with regex patterns
- Rule-based overrides for special cases

### 3. **Performance Optimization**
- Parallel processing using ThreadPoolExecutor
- Configurable thread count (NUM_THREADS = 8)
- SpaCy model optimization (disabled unnecessary components)

### 4. **User Experience**
- Clear emoji-enhanced UI labels
- Helpful error messages and progress indicators
- Multiple download options provided
- Preview of results (first 100 rows)

---

## 🔧 Areas for Improvement

### 1. **File Upload Handling** (Critical)

**Issue:** Complex file detection logic with fallback attempts
```python
# Current approach tries multiple methods
if original_name.lower().endswith('.xlsx'):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        error_msg = f"Excel read error: {str(e)}"
```

**Recommendation:**
```python
def load_dataframe(file_path, original_name):
    """Load dataframe with clear error messages."""
    try:
        if original_name.lower().endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif original_name.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            # Try pandas auto-detection
            return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read file '{original_name}': {str(e)}")
```

### 2. **Error Handling** (Important)

**Issue:** Broad exception catching loses valuable debugging information
```python
try:
    return detect(text) == "en"
except Exception:
    return False
```

**Recommendation:**
```python
def is_english(text):
    """Detect if text is English with proper error handling."""
    if not text or len(text.strip()) < 3:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        # Too short or unclear language
        return False
    except Exception as e:
        # Log unexpected errors for debugging
        print(f"Language detection error: {e}")
        return False
```

### 3. **SpaCy Model Check** (Important)

**Issue:** The app loads spaCy but doesn't actually use it anywhere
```python
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
except OSError:
    print("⚠️ spaCy model 'en_core_web_sm' not found.")
    nlp = None
```

**Recommendation:**
- Either remove spaCy entirely if unused
- Or use it for lemmatization/tokenization:
```python
def preprocess_text(text):
    """Preprocess text using spaCy for better keyword matching."""
    if nlp is None:
        return text.lower()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
```

### 4. **Keyword Matching Logic** (Medium Priority)

**Issue:** Simple keyword counting doesn't account for context or word boundaries
```python
matches = sum(k in text_lower for k in keywords)
```

**Problems:**
- "blog" matches "blogging"
- "ad" matches "add", "bad", "sad"
- No weighting by keyword importance

**Recommendation:**
```python
import re

def predict_category_improved(text):
    """Predict category with word boundary matching and weights."""
    text_lower = text.lower()
    best_match, best_score = "", 0
    
    for category, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Use word boundaries for short keywords
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
    
    return best_match if best_score > 0 else ""
```

### 5. **Configuration Management** (Medium Priority)

**Issue:** Hard-coded values scattered throughout
```python
NUM_THREADS = 8
score = 0.6 * tb_score + 0.4 * af_score
if score <= -0.75: return "very negative"
```

**Recommendation:**
```python
# At the top of the file
class Config:
    NUM_THREADS = os.cpu_count() or 4
    SENTIMENT_WEIGHTS = {'textblob': 0.6, 'afinn': 0.4}
    SENTIMENT_THRESHOLDS = {
        'very_negative': -0.75,
        'negative': -0.25,
        'positive': 0.25,
        'very_positive': 0.75
    }
    MAX_PREVIEW_ROWS = 100
```

### 6. **Regex Extraction** (Medium Priority)

**Issue:** Complex regex patterns are hard to maintain
```python
parts = re.findall(r"(?i)Consumer:\s*(.*?)(?=\s*\|\s*\d{4}-\d{2}-\d{2}|$|\s*\|\s*Agent:)", 
                   transcript + " ")
```

**Recommendation:**
```python
# Define patterns as constants with documentation
CONSUMER_PATTERN_PRIMARY = re.compile(
    r"(?i)Consumer:\s*(.*?)(?=\s*\|\s*\d{4}-\d{2}-\d{2}|$|\s*\|\s*Agent:)",
    re.IGNORECASE
)
CONSUMER_PATTERN_FALLBACK = re.compile(
    r"(?i)Consumer:\s*(.*?)(?=\||$)",
    re.IGNORECASE
)

def extract_consumer_text(transcript):
    """Extract consumer text from transcript using multiple patterns."""
    if not isinstance(transcript, str):
        return ""
    
    # Try primary pattern
    parts = CONSUMER_PATTERN_PRIMARY.findall(transcript + " ")
    
    # Fallback to simpler pattern
    if not parts:
        parts = CONSUMER_PATTERN_FALLBACK.findall(transcript + "|")
    
    return " ".join(p.strip() for p in parts if p.strip())
```

### 7. **Output File Management** (Low Priority)

**Issue:** Files accumulate without cleanup
```python
output_filename = f"sentiment_output_{timestamp}.csv"
```

**Recommendation:**
```python
# Add cleanup or organization
OUTPUT_DIR = "nlp_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_filename = f"sentiment_output_{timestamp}.csv"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# Optional: Clean old files
def cleanup_old_files(directory, days=7):
    """Remove files older than specified days."""
    cutoff = time.time() - (days * 86400)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
            os.remove(filepath)
```

### 8. **Data Validation** (Medium Priority)

**Issue:** Minimal validation of input data
```python
if "Conversation Id" not in df.columns or "transcripts" not in df.columns:
    raise ValueError("Input file must contain 'Conversation Id' and 'transcripts' columns.")
```

**Recommendation:**
```python
def validate_input_dataframe(df):
    """Validate input dataframe structure and content."""
    # Check required columns
    required_cols = ["Conversation Id", "transcripts"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    # Check for all-null transcripts
    null_count = df["transcripts"].isnull().sum()
    if null_count == len(df):
        raise ValueError("All transcript values are empty")
    
    # Warn about high null percentage
    if null_count > len(df) * 0.5:
        print(f"⚠️ Warning: {null_count}/{len(df)} ({null_count/len(df)*100:.1f}%) transcripts are empty")
    
    return True
```

### 9. **Memory Efficiency** (Low Priority)

**Issue:** Loading entire result list into memory
```python
results = list(executor.map(process_row, df.to_dict("records")))
```

**Recommendation for large files:**
```python
# Process in chunks for very large files
CHUNK_SIZE = 1000

def run_pipeline_chunked(uploaded_file):
    # ... file loading code ...
    
    chunks = [df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
    
    all_results = []
    for chunk_num, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {chunk_num}/{len(chunks)}...")
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            results = list(executor.map(process_row, chunk.to_dict("records")))
        all_results.extend(results)
    
    return pd.DataFrame(all_results)
```

---

## 🐛 Potential Bugs

### 1. **Empty Category Handling**
```python
def predict_subcategory(category, text):
    if not category or category not in SUBCATEGORY_KEYWORDS:
        return ""
```
**Issue:** Empty string returned, but might be better to return `None` or "unknown"

### 2. **Sentiment for Short Text**
```python
def hybrid_sentiment(text):
    if not text or not is_english(text):
        return ""
```
**Issue:** Very short texts might give unreliable sentiment scores

**Fix:**
```python
if not text or len(text.split()) < 3:
    return "neutral"  # or "insufficient_text"
```

### 3. **Rule Override Side Effects**
```python
def apply_rules(text, preds):
    for cond, override in RULE_OVERRIDES:
        if cond(text):
            preds = override(preds)
    return preds
```
**Issue:** If multiple rules match, later rules override earlier ones. Might be intentional but should be documented.

---

## 📊 Performance Considerations

### Current Performance:
- ✅ Parallel processing implemented
- ✅ SpaCy components disabled
- ⚠️ No caching (for repeated analyses)
- ⚠️ All regex compiled at runtime

### Optimization Suggestions:

```python
# 1. Pre-compile regex patterns (DONE partially)
PATTERNS = {
    'consumer_primary': re.compile(r"(?i)Consumer:\s*(.*?)(?=\s*\|\s*\d{4}-\d{2}-\d{2}|$|\s*\|\s*Agent:)"),
    'consumer_fallback': re.compile(r"(?i)Consumer:\s*(.*?)(?=\||$)")
}

# 2. Cache language detection for repeated texts
from functools import lru_cache

@lru_cache(maxsize=1000)
def is_english_cached(text):
    return is_english(text)

# 3. Batch sentiment analysis
afinn_scores = [af.score(text) for text in texts]  # Vectorize if possible
```

---

## 🎨 UI/UX Improvements

### Current State:
- ✅ Clear emoji labels
- ✅ Progress indicators
- ⚠️ No progress bar for long-running tasks
- ⚠️ Multiple download instructions might confuse users

### Recommendations:

```python
# 1. Add processing progress
for idx, row in enumerate(df.to_dict("records")):
    if idx % 100 == 0:
        print(f"Progress: {idx}/{len(df)} rows ({idx/len(df)*100:.1f}%)")
    process_row(row)

# 2. Simplify download instructions
mr.Markdown("### 📥 Download Your Results")
mr.Markdown(f"**File saved as:** `{output_filename}`")

if os.path.exists(output_path):
    file_size = os.path.getsize(output_path) / 1024  # KB
    mr.Markdown(f"**File size:** {file_size:.1f} KB")
    mr.Markdown(f"**Location:** `{output_path}`")

# 3. Add summary statistics
mr.Markdown("### 📈 Processing Summary")
mr.Markdown(f"- **Total rows:** {len(df_result)}")
mr.Markdown(f"- **Valid English texts:** {(df_result['Category'] != '').sum()}")
mr.Markdown(f"- **Processing time:** {elapsed:.2f} seconds")
mr.Markdown(f"- **Rows per second:** {len(df_result)/elapsed:.1f}")
```

---

## 🔒 Security Considerations

1. **File Upload Safety:**
   - ✅ Limited to CSV/XLSX formats
   - ⚠️ No file size limit check
   - ⚠️ No malicious content scanning

2. **Path Traversal:**
   - ⚠️ Output filename uses timestamp but no sanitization
   - **Fix:** Use `os.path.basename()` on any user input

3. **Resource Exhaustion:**
   - ⚠️ Large files could crash the app
   - **Fix:** Add file size limit and row count limit

```python
MAX_FILE_SIZE_MB = 100
MAX_ROWS = 50000

file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
if file_size_mb > MAX_FILE_SIZE_MB:
    raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")

if len(df) > MAX_ROWS:
    raise ValueError(f"Too many rows: {len(df)} (max: {MAX_ROWS})")
```

---

## 📝 Documentation Improvements

### Add Docstrings:
```python
def process_row(row):
    """
    Process a single transcript row and classify it.
    
    Args:
        row (dict): Dictionary containing 'Conversation Id' and 'transcripts'
    
    Returns:
        dict: Processed row with added classification fields:
            - Consumer_Text: Extracted consumer portion
            - Category: Predicted main category
            - Subcategory: Predicted subcategory (if applicable)
            - Sentiment: Sentiment analysis result
    
    Example:
        >>> row = {'Conversation Id': '123', 'transcripts': 'Consumer: Login failed'}
        >>> result = process_row(row)
        >>> result['Category']
        'login issue'
    """
```

---

## 🎯 Testing Recommendations

### Unit Tests Needed:
```python
def test_extract_consumer_text():
    test_cases = [
        ("Consumer: Hello | Agent: Hi", "Hello"),
        ("Consumer: Help me | 2024-01-01 | Agent: Sure", "Help me"),
        ("No consumer text", ""),
    ]
    for input_text, expected in test_cases:
        assert extract_consumer_text(input_text) == expected

def test_sentiment_analysis():
    assert hybrid_sentiment("I love this!") in ["positive", "very positive"]
    assert hybrid_sentiment("This is terrible") in ["negative", "very negative"]
    assert hybrid_sentiment("It's okay") == "neutral"
```

---

## 🚀 Quick Wins (Easy Improvements)

1. **Add logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

2. **Add row counter in output:**
   ```python
   out_df.insert(0, 'Row_Number', range(1, len(out_df) + 1))
   ```

3. **Add confidence scores:**
   ```python
   return {
       ...,
       "Category_Confidence": best_score,
       "Needs_Review": best_score < 2  # Flag low-confidence predictions
   }
   ```

4. **Add category distribution:**
   ```python
   mr.Markdown("### 📊 Category Distribution")
   category_counts = df_result['Category'].value_counts()
   for category, count in category_counts.items():
       mr.Markdown(f"- **{category}**: {count} ({count/len(df_result)*100:.1f}%)")
   ```

---

## Summary

### Priority Fixes:
1. ✅ **HIGH**: Improve error handling and validation
2. ✅ **HIGH**: Fix keyword matching (word boundaries)
3. ✅ **MEDIUM**: Remove unused spaCy or implement properly
4. ✅ **MEDIUM**: Add configuration management
5. ✅ **LOW**: Add file size limits and cleanup

### Recommended Next Steps:
1. Implement improved keyword matching
2. Add comprehensive input validation
3. Create unit tests for core functions
4. Add processing progress indicators
5. Document the category/subcategory taxonomy
6. Consider ML model for better classification (if needed)

**Overall**: This is a functional application that works well for its intended purpose. The main improvements should focus on robustness, maintainability, and user experience rather than core functionality changes.
