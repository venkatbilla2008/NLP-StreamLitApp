# 🎯 NLP Text Classification Dashboard

A modern Streamlit application for analyzing customer service transcripts using Natural Language Processing. Classifies transcripts by category, subcategory, and sentiment with efficient Parquet output format.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 📊 **Multi-threaded Processing**: Parallel processing for fast analysis
- 🎯 **15+ Categories**: Comprehensive classification (login, playback, subscription, etc.)
- 💭 **Sentiment Analysis**: Hybrid approach using TextBlob + AFINN
- 📦 **Parquet Output**: Efficient storage with 30-50% size reduction vs CSV
- 🔍 **Confidence Scores**: Know which predictions need review
- 📈 **Interactive Visualizations**: Category and sentiment distributions
- 🚀 **User-Friendly Interface**: Drag-and-drop file upload with progress tracking

## 🔧 Requirements

- Python 3.11 or higher
- 4GB+ RAM (for large datasets)
- Multi-core CPU (recommended for parallel processing)

## 📥 Installation

### Method 1: Using pip

```bash
# Clone or download the files
cd /path/to/app

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data (first time only)
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n nlp-app python=3.11
conda activate nlp-app

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

## 🚀 Quick Start

1. **Start the application:**

```bash
streamlit run streamlit_nlp_app.py
```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

3. **Upload your data:**
   - Click "Browse files" in the sidebar
   - Upload a CSV or Excel file
   - Required columns: `Conversation Id`, `transcripts`

4. **Run analysis:**
   - Click "🚀 Run Analysis" button
   - Wait for processing (progress bar shows status)
   - View results and download files

## 📋 Input Data Format

Your input file must have these columns:

| Column Name | Type | Description | Required |
|-------------|------|-------------|----------|
| `Conversation Id` | String | Unique identifier for each conversation | ✅ Yes |
| `transcripts` | String | Full conversation transcript | ✅ Yes |

### Example Input Format

```csv
Conversation Id,transcripts
CONV_001,"Consumer: I can't login to my account | Agent: Let me help you with that | 2024-01-15"
CONV_002,"Consumer: Songs are not playing properly | Agent: I'll check your account"
CONV_003,"Consumer: Need refund for duplicate charge | Agent: Processing your request"
```

### Transcript Format

The app automatically extracts consumer text from transcripts in these formats:

- `Consumer: <text> | Agent: <text>`
- `Consumer: <text> | 2024-01-15 | Agent: <text>`
- `Consumer: <text> | <other content>`

## 📊 Output Format

### Parquet File (Recommended)

The app generates a Parquet file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Row_Number` | Integer | Sequential row number |
| `Conversation Id` | String | Original conversation ID |
| `Consumer_Text` | String | Extracted consumer text |
| `Category` | String | Main issue category |
| `Subcategory` | String | Specific subcategory |
| `Sentiment` | String | Sentiment classification |
| `Category_Confidence` | Integer | Keyword match count |
| `Subcategory_Confidence` | Integer | Subcategory match count |
| `Needs_Review` | Boolean | Low confidence flag (< 2 matches) |

### Why Parquet?

- **30-50% smaller** file size than CSV
- **Faster** read/write operations
- **Preserves data types** (no type inference needed)
- **Column-oriented** storage for efficient queries
- Compatible with **Pandas, Spark, DuckDB**

## 📚 Supported Categories

### Main Categories (15)

1. **Login Issue** - Authentication, password, 2FA problems
2. **Account Issue** - Profile, settings, account management
3. **Playback Issue** - Music not playing, buffering, audio problems
4. **Device Issue** - Bluetooth, speakers, smart devices
5. **Content Restriction** - Unavailable songs, region locks
6. **Ad Issue** - Advertisement frequency, volume
7. **Recommendation Issue** - Algorithm, playlists, radio
8. **UI Issue** - Interface, navigation, design
9. **General Feedback** - Suggestions, feature requests
10. **Network Failure** - Connectivity, server issues
11. **App Crash** - Crashes, freezes, bugs
12. **Performance Issue** - Slow loading, lag
13. **Data Sync Issue** - Sync problems, data loss
14. **Subscription Issue** - Billing, payments, plans

### Subcategories

#### Subscription Issue
- `payment` - Refunds, billing, payment failures
- `cancel` - Cancellation requests
- `upgrade` - Plan upgrades, family/student plans

#### Account Issue
- `login` - Login and authentication
- `profile` - Profile and settings

#### Device Issue
- `mobile` - Phone apps (Android/iOS)
- `car` - CarPlay, Android Auto
- `smart_device` - Alexa, Chromecast, smart TVs

## 💭 Sentiment Analysis

The app uses a **hybrid approach** combining:

1. **TextBlob** (60% weight) - Grammar-based sentiment
2. **AFINN** (40% weight) - Lexicon-based scoring

### Sentiment Labels

- `very negative` - Score ≤ -0.75
- `negative` - Score ≤ -0.25
- `neutral` - -0.25 < Score < 0.25
- `positive` - Score ≥ 0.25
- `very positive` - Score ≥ 0.75

## ⚙️ Configuration

### Adjustable Settings (in sidebar)

- **Number of threads**: 1 to CPU core count (default: all cores)
- **Preview rows**: 10 to 500 (default: 100)

### Hard Limits (in code)

```python
MAX_FILE_SIZE_MB = 100    # Maximum upload size
MAX_ROWS = 50,000         # Maximum rows to process
MIN_TEXT_LENGTH = 3       # Minimum words for analysis
CHUNK_SIZE = 1000         # Progress update frequency
```

To modify these limits, edit the `Config` class in `streamlit_nlp_app.py`.

## 🎨 User Interface

### Main Features

1. **Sidebar**
   - File upload
   - Settings configuration
   - System information

2. **Main Panel**
   - Data preview
   - Processing progress
   - Results statistics
   - Category/sentiment charts
   - Download buttons

3. **Results Display**
   - Total rows processed
   - Valid classifications count
   - Items needing review
   - File size comparison

## 📈 Performance Tips

### For Large Datasets

1. **Increase thread count** (if you have CPU cores)
2. **Process in batches** (split large files)
3. **Use Parquet format** (faster than CSV)
4. **Close other applications** (free up RAM)

### Expected Processing Speed

- Small files (< 1,000 rows): **Instant** (< 5 seconds)
- Medium files (1,000 - 10,000 rows): **Fast** (5-30 seconds)
- Large files (10,000 - 50,000 rows): **Moderate** (30-120 seconds)

*Note: Speed depends on CPU cores, text length, and system load*

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

#### 2. NLTK Data Missing

```bash
LookupError: Resource 'tokenizers/punkt' not found
```

**Solution:**
```bash
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

#### 3. File Upload Fails

```
Error: File too large
```

**Solution:**
- Split large files into smaller chunks
- Or increase `MAX_FILE_SIZE_MB` in config

#### 4. Memory Error

```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce file size
- Close other applications
- Increase system RAM
- Lower `NUM_THREADS`

#### 5. Column Not Found

```
ValueError: Missing required columns: transcripts
```

**Solution:**
- Check column names match exactly: `Conversation Id`, `transcripts`
- No extra spaces or different capitalization

## 🛠️ Advanced Usage

### Reading Parquet Files

#### Using Pandas
```python
import pandas as pd

# Read parquet file
df = pd.read_parquet('sentiment_output_20240115_143022.parquet')
print(df.head())

# Filter by category
login_issues = df[df['Category'] == 'login issue']

# Filter by sentiment
negative_feedback = df[df['Sentiment'].isin(['negative', 'very negative'])]
```

#### Using DuckDB (SQL queries)
```python
import duckdb

# Query parquet directly
result = duckdb.sql("""
    SELECT Category, COUNT(*) as count
    FROM 'sentiment_output_20240115_143022.parquet'
    GROUP BY Category
    ORDER BY count DESC
""").df()
print(result)
```

#### Using Polars (faster alternative to Pandas)
```python
import polars as pl

# Read with Polars
df = pl.read_parquet('sentiment_output_20240115_143022.parquet')

# Fast filtering
needs_review = df.filter(pl.col('Needs_Review') == True)
```

### Batch Processing

Process multiple files:

```python
import os
import pandas as pd

input_dir = 'input_files/'
output_dir = 'output_files/'

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        print(f"Processing {filename}...")
        # Load and process
        df = pd.read_csv(os.path.join(input_dir, filename))
        # ... run your analysis ...
        # Save results
        output_name = filename.replace('.csv', '.parquet')
        result_df.to_parquet(os.path.join(output_dir, output_name))
```

### Custom Categories

To add custom categories, edit the `TOPIC_KEYWORDS` dictionary:

```python
TOPIC_KEYWORDS = {
    # ... existing categories ...
    "custom_category": [
        "keyword1",
        "keyword2",
        "multi word phrase",
    ],
}
```

## 📝 File Structure

```
.
├── streamlit_nlp_app.py    # Main application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── nlp_results/            # Output directory (auto-created)
    ├── sentiment_output_*.parquet
    └── sentiment_output_*.csv
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more sophisticated ML models (BERT, transformers)
- [ ] Multi-language support
- [ ] Real-time processing
- [ ] Database integration
- [ ] API endpoint
- [ ] Batch processing UI
- [ ] Export to multiple formats (JSON, Excel with formatting)

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🐛 Known Issues

1. Very short texts (< 3 words) may not classify accurately
2. Mixed-language transcripts may be filtered out
3. Progress bar updates in batches (not per-row)

## 📞 Support

If you encounter issues:

1. Check this README's troubleshooting section
2. Verify your Python version: `python --version`
3. Check installed packages: `pip list`
4. Look at Streamlit logs in the terminal

## 🎓 Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [TextBlob](https://textblob.readthedocs.io/) - Sentiment analysis
- [AFINN](https://github.com/fnielsen/afinn) - Sentiment lexicon
- [langdetect](https://github.com/Mimino666/langdetect) - Language detection
- [PyArrow](https://arrow.apache.org/docs/python/) - Parquet support

---

**Version:** 1.0.0  
**Last Updated:** November 2024  
**Python Required:** 3.11+
