# ✅ Getting Started Checklist

Follow these steps to get your NLP dashboard running in 5 minutes!

---

## 📋 Pre-Installation Checklist

- [ ] Python 3.11 or higher installed
  ```bash
  python3 --version
  # Should show 3.11.x or higher
  ```

- [ ] pip installed and updated
  ```bash
  pip --version
  ```

- [ ] At least 4GB free RAM
- [ ] Terminal/Command Prompt access

---

## 🚀 Installation Checklist (One-Time)

### Option A: Automatic Setup (Recommended)

**Linux/Mac:**
- [ ] Navigate to project directory
  ```bash
  cd /path/to/project
  ```

- [ ] Make setup script executable
  ```bash
  chmod +x setup.sh
  ```

- [ ] Run setup script
  ```bash
  ./setup.sh
  ```

- [ ] Wait for "Setup complete!" message

**Windows:**
- [ ] Navigate to project directory
  ```cmd
  cd C:\path\to\project
  ```

- [ ] Run setup script
  ```cmd
  setup.bat
  ```

- [ ] Wait for "Setup complete!" message

### Option B: Manual Setup

- [ ] Create virtual environment
  ```bash
  python3 -m venv venv
  ```

- [ ] Activate virtual environment
  ```bash
  # Linux/Mac:
  source venv/bin/activate
  
  # Windows:
  venv\Scripts\activate
  ```

- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Download NLTK data
  ```bash
  python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
  ```

- [ ] Create output directory
  ```bash
  mkdir nlp_results
  ```

---

## 🎯 First Run Checklist

- [ ] Virtual environment is activated
  ```bash
  # You should see (venv) in your prompt
  ```

- [ ] Start the application
  ```bash
  streamlit run streamlit_nlp_app.py
  ```

- [ ] Browser opens to http://localhost:8501
  - [ ] If not, manually open the URL shown in terminal

- [ ] You see the welcome screen with "Get Started" message

---

## 📊 Test Run Checklist

### Generate Sample Data

- [ ] Open new terminal (keep Streamlit running)

- [ ] Activate virtual environment
  ```bash
  source venv/bin/activate  # or venv\Scripts\activate
  ```

- [ ] Generate sample data
  ```bash
  python generate_sample_data.py
  ```

- [ ] Enter number of rows (try 100 first)

- [ ] Choose format (csv is fine for testing)

- [ ] Note the generated filename

### Upload and Process

- [ ] In Streamlit app, click "Browse files" in sidebar

- [ ] Select the generated sample file

- [ ] File info appears (size, row count)

- [ ] Click "🚀 Run Analysis" button

- [ ] Progress bar shows processing status

- [ ] "✅ Processing Complete!" message appears

### Review Results

- [ ] Check Results Summary metrics:
  - [ ] Total rows processed
  - [ ] Valid classifications count
  - [ ] Items needing review
  - [ ] File size

- [ ] View Category Distribution chart

- [ ] View Sentiment Distribution chart

- [ ] Scroll through Results Preview table

### Download Results

- [ ] Click "📦 Download Parquet File" button
  - [ ] File downloads successfully
  - [ ] Note the file size (should be ~50% smaller than CSV)

- [ ] Click "📄 Download CSV File" button
  - [ ] File downloads successfully

- [ ] Open downloaded file in Excel/Pandas to verify
  ```python
  import pandas as pd
  df = pd.read_parquet('sentiment_output_*.parquet')
  print(df.head())
  ```

---

## 🎨 Configuration Checklist

- [ ] Try adjusting "Number of threads" slider
  - [ ] More threads = faster processing (if you have CPU cores)

- [ ] Try adjusting "Preview rows" slider
  - [ ] Shows more/fewer rows in preview

- [ ] Check "System Info" in sidebar
  - [ ] Python version
  - [ ] CPU cores available
  - [ ] Max file size
  - [ ] Max rows

---

## ✅ Production Use Checklist

### Prepare Your Data

- [ ] Your file is CSV or Excel format

- [ ] File has these columns (exact names):
  - [ ] `Conversation Id`
  - [ ] `transcripts`

- [ ] File size is under 100 MB
  - [ ] If larger, split into smaller files

- [ ] Row count is under 50,000
  - [ ] If larger, process in batches

- [ ] Transcripts include consumer text:
  ```
  Consumer: <message> | Agent: <reply>
  ```

### Process Your Data

- [ ] Upload your real data file

- [ ] Review data preview (first 5 rows)

- [ ] Verify columns are correct

- [ ] Click "Run Analysis"

- [ ] Wait for processing to complete

- [ ] Review results statistics

- [ ] Check category distribution (does it make sense?)

- [ ] Check sentiment distribution (looks reasonable?)

### Save and Analyze Results

- [ ] Download Parquet file (recommended)
  - [ ] Smaller file size
  - [ ] Faster to load later

- [ ] Or download CSV file
  - [ ] Compatible with Excel
  - [ ] Easy to share

- [ ] Open in analysis tool:
  ```python
  import pandas as pd
  df = pd.read_parquet('your_results.parquet')
  
  # Analyze results
  print(df['Category'].value_counts())
  print(df['Sentiment'].value_counts())
  
  # Find items needing review
  review = df[df['Needs_Review'] == True]
  print(f"{len(review)} items need review")
  ```

---

## 🐛 Troubleshooting Checklist

### If App Won't Start

- [ ] Virtual environment activated?
  ```bash
  source venv/bin/activate
  ```

- [ ] Dependencies installed?
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Port 8501 available?
  ```bash
  # Try different port:
  streamlit run streamlit_nlp_app.py --server.port 8502
  ```

### If Upload Fails

- [ ] File is CSV or XLSX?

- [ ] Columns are named correctly?
  - `Conversation Id` and `transcripts`

- [ ] File size under 100 MB?

- [ ] Row count under 50,000?

### If Processing Fails

- [ ] Check terminal for error messages

- [ ] Try sample data first (to rule out data issues)

- [ ] Check system resources (RAM, CPU)

- [ ] Try reducing thread count

### If Results Look Wrong

- [ ] Check input data quality
  - Empty transcripts?
  - Non-English text?
  - Malformed data?

- [ ] Review "Needs_Review" column
  - Low confidence predictions flagged

- [ ] Check "Category_Confidence" scores
  - Low scores = uncertain predictions

---

## 📚 Documentation Checklist

After first successful run, review these docs:

- [ ] **README.md**
  - Complete installation guide
  - Detailed usage instructions
  - Troubleshooting section
  - Advanced usage examples

- [ ] **QUICK_REFERENCE.md**
  - One-page cheat sheet
  - Common commands
  - Quick fixes

- [ ] **MERCURY_VS_STREAMLIT.md**
  - Feature comparison
  - Migration guide
  - Why Streamlit is better

- [ ] **PROJECT_SUMMARY.md**
  - Complete overview
  - What's included
  - Next steps

---

## 🎓 Learning Checklist

### Understand the Output

- [ ] Know what each column means:
  - `Category` - Main issue type
  - `Subcategory` - Specific issue
  - `Sentiment` - Emotion (5 levels)
  - `Category_Confidence` - How certain (keyword count)
  - `Needs_Review` - Flag for manual review

- [ ] Understand sentiment scale:
  - very negative → negative → neutral → positive → very positive

- [ ] Know the 15 categories:
  - login, account, playback, device, content restriction,
  - ad, recommendation, ui, feedback, network, crash,
  - performance, sync, subscription

### Analyze Results

- [ ] Use Pandas to explore:
  ```python
  import pandas as pd
  df = pd.read_parquet('results.parquet')
  
  # Most common issues
  print(df['Category'].value_counts())
  
  # Negative sentiment issues
  negative = df[df['Sentiment'].isin(['negative', 'very negative'])]
  print(negative['Category'].value_counts())
  
  # Low confidence predictions
  review = df[df['Category_Confidence'] < 2]
  print(f"Need review: {len(review)} items")
  ```

---

## 🚀 Advanced Usage Checklist

### Customization

- [ ] Add custom categories (edit `TOPIC_KEYWORDS`)
- [ ] Adjust sentiment thresholds (edit `Config`)
- [ ] Modify file size limits (edit `Config`)
- [ ] Change thread count default (edit `Config`)

### Batch Processing

- [ ] Process multiple files
- [ ] Combine results
- [ ] Generate reports

### Integration

- [ ] Export to database
- [ ] Create API endpoint
- [ ] Schedule automatic processing
- [ ] Connect to BI tools

---

## ✅ Success Criteria

You're ready when you can:

- [x] Start the app without errors
- [x] Upload and process sample data
- [x] Download results in both formats
- [x] Understand the output columns
- [x] Process your real data successfully
- [x] Analyze results with Pandas

---

## 🎉 Congratulations!

If you've checked all the boxes above, you're ready to use the NLP dashboard for production work!

### Next Steps:
1. Process your real customer service data
2. Analyze patterns and trends
3. Share insights with your team
4. Customize categories for your use case
5. Explore advanced features

### Need Help?
- Check README.md for detailed docs
- Review QUICK_REFERENCE.md for commands
- Look at error messages in terminal
- Test with sample data to isolate issues

---

**Happy Analyzing! 📊🎯**
