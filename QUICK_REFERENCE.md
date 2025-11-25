# 🚀 Quick Reference Guide

## Installation (One-Time Setup)

### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

### Windows
```cmd
setup.bat
```

### Manual Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

---

## Running the App

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Start app
streamlit run streamlit_nlp_app.py
```

**App URL:** http://localhost:8501

---

## Quick Usage

1. **Upload File** → Browse in sidebar
2. **Click "Run Analysis"** → Wait for progress bar
3. **Download Results** → Choose Parquet or CSV

---

## File Requirements

### Required Columns
- `Conversation Id` (string)
- `transcripts` (string)

### Transcript Format
```
Consumer: <message> | Agent: <reply>
Consumer: <message> | 2024-01-15 | Agent: <reply>
```

### Limits
- Max file size: **100 MB**
- Max rows: **50,000**
- File types: **CSV, XLSX**

---

## Output Columns

| Column | Description |
|--------|-------------|
| `Row_Number` | Sequential number |
| `Conversation Id` | Original ID |
| `Consumer_Text` | Extracted text |
| `Category` | Main category |
| `Subcategory` | Sub-category |
| `Sentiment` | Sentiment (5 levels) |
| `Category_Confidence` | Match count |
| `Subcategory_Confidence` | Sub-match count |
| `Needs_Review` | Low confidence flag |

---

## Categories (15 Total)

```
• login issue          • account issue
• playback issue       • device issue
• content restriction  • ad issue
• recommendation issue • ui issue
• general feedback     • network failure
• app crash            • performance issue
• data sync issue      • subscription issue
```

---

## Sentiment Levels

```
very negative  →  negative  →  neutral  →  positive  →  very positive
    (≤-0.75)      (≤-0.25)    (-0.25 to     (≥0.25)      (≥0.75)
                                 0.25)
```

---

## Common Commands

### Generate Sample Data
```bash
python generate_sample_data.py
```

### Read Parquet in Python
```python
import pandas as pd
df = pd.read_parquet('sentiment_output_*.parquet')
```

### Check File Size
```bash
ls -lh nlp_results/
```

### View Output Directory
```bash
cd nlp_results/
ls -la
```

---

## Keyboard Shortcuts (in Streamlit)

- `R` - Rerun app
- `C` - Clear cache
- `S` - Open settings
- `?` - Show keyboard shortcuts

---

## Troubleshooting Quick Fixes

### App Won't Start
```bash
pip install --upgrade streamlit
```

### Import Error
```bash
pip install -r requirements.txt
```

### Port Already in Use
```bash
streamlit run streamlit_nlp_app.py --server.port 8502
```

### Clear Cache
```bash
streamlit cache clear
```

---

## Configuration Quick Edit

Edit `streamlit_nlp_app.py`:

```python
class Config:
    NUM_THREADS = 8           # Change to your CPU cores
    MAX_FILE_SIZE_MB = 100    # Increase if needed
    MAX_ROWS = 50000          # Increase if needed
```

---

## Performance Tips

- ✅ Use **Parquet** output (30-50% smaller)
- ✅ Increase **threads** if you have CPU cores
- ✅ Close **other apps** when processing
- ✅ Split **large files** into batches
- ❌ Don't process during **high system load**

---

## File Size Comparison

| Rows | CSV Size | Parquet Size | Savings |
|------|----------|--------------|---------|
| 1K   | ~300 KB  | ~150 KB      | 50%     |
| 10K  | ~3 MB    | ~1.5 MB      | 50%     |
| 50K  | ~15 MB   | ~8 MB        | 47%     |

---

## Processing Speed Guide

| Rows  | Threads=4 | Threads=8 | Threads=16 |
|-------|-----------|-----------|------------|
| 1K    | 3-5s      | 2-3s      | 1-2s       |
| 10K   | 15-25s    | 10-15s    | 5-10s      |
| 50K   | 60-90s    | 40-60s    | 25-40s     |

*Actual speed varies by CPU, text length, and system load*

---

## URLs & Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Pandas Docs:** https://pandas.pydata.org
- **Parquet Format:** https://parquet.apache.org

---

## Emergency Commands

### Kill Streamlit Process
```bash
# Linux/Mac
pkill -f streamlit

# Windows
taskkill /F /IM streamlit.exe
```

### Reset Everything
```bash
rm -rf venv nlp_results __pycache__ .streamlit
./setup.sh
```

---

## Advanced Options

### Run on Different Port
```bash
streamlit run streamlit_nlp_app.py --server.port 8502
```

### Run on Network
```bash
streamlit run streamlit_nlp_app.py --server.address 0.0.0.0
```

### Disable Auto-Reload
```bash
streamlit run streamlit_nlp_app.py --server.runOnSave false
```

---

## Getting Help

1. Check **README.md** for detailed docs
2. Look at **terminal output** for errors
3. Try **sample data generator** to test
4. Check **requirements.txt** versions

---

**Version:** 1.0.0  
**Last Updated:** November 2024
