# 🔄 Update Your GitHub Repository - Complete Guide

## Your Repository
**GitHub:** https://github.com/venkatbilla2008/nlp-text-pipeline

## What We'll Do
1. ✅ Clone your existing repository
2. ✅ Remove all old files
3. ✅ Add only essential files (4 files total!)
4. ✅ Push updated code to GitHub
5. ✅ Streamlit Cloud will auto-deploy

**Time needed:** 10-15 minutes  
**Files needed:** 4 essential files only

---

## 📋 Essential Files You Need

### 1. streamlit_nlp_app.py ✅
**Status:** You provided this (your modified code with translation)  
**Action:** Save your code as `streamlit_nlp_app.py`

### 2. requirements.txt ✅
**Status:** Need to update (add googletrans)  
**Action:** Use the updated version provided below

### 3. README.md ✅
**Status:** Need to update (mention translation feature)  
**Action:** Use the updated version provided below

### 4. .gitignore ✅
**Status:** Auto-created  
**Action:** Script will create this

---

## 🚀 Method 1: Automated (Easiest - Recommended)

### Step 1: Download Files

Download these 4 files to a folder on your computer:

1. **Your modified app code** (save as `streamlit_nlp_app.py`)
2. [requirements_with_translation.txt](computer:///mnt/user-data/outputs/requirements_with_translation.txt) - Rename to `requirements.txt`
3. [README_with_translation.md](computer:///mnt/user-data/outputs/README_with_translation.md) - Rename to `README.md`
4. [update_github_repo.sh](computer:///mnt/user-data/outputs/update_github_repo.sh) - Update script

### Step 2: Run the Script

```bash
# Make script executable
chmod +x update_github_repo.sh

# Run it
./update_github_repo.sh
```

The script will:
- Clone your repository
- Backup old files
- Remove unnecessary files
- Add your new files
- Push to GitHub

**Done!** Skip to "After Pushing" section below.

---

## 🔧 Method 2: Manual Steps (Full Control)

### Step 1: Clone Your Repository

```bash
# Clone your existing repo
git clone https://github.com/venkatbilla2008/nlp-text-pipeline.git

# Navigate into it
cd nlp-text-pipeline

# Check current files
ls -la
```

### Step 2: Backup Current Files (Optional)

```bash
# Create backup
mkdir ../backup-old-files
cp -r * ../backup-old-files/

# You can delete this later
echo "Backup created at ../backup-old-files"
```

### Step 3: Remove ALL Old Files

```bash
# Remove everything except .git folder
rm -rf *

# Remove hidden files too (except .git)
rm -f .*
# Don't worry - this keeps .git safe

# Verify only .git remains
ls -A
# Should only show .git
```

### Step 4: Add Your Modified Files

Now copy these 4 files into the `nlp-text-pipeline` folder:

#### File 1: streamlit_nlp_app.py

**Copy your modified code** (from the document you shared) and save as `streamlit_nlp_app.py`

#### File 2: requirements.txt

Create `requirements.txt` with this content:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
textblob>=0.17.0
afinn>=0.1
langdetect>=1.0.9
googletrans==4.0.0rc1
openpyxl>=3.1.0
pyarrow>=14.0.0
python-snappy>=0.6.1
```

**Important:** Added `googletrans==4.0.0rc1` for translation support!

#### File 3: README.md

Create `README.md` - you can use the template provided or create your own:

```markdown
# 🎯 NLP Text Classification with Translation

Streamlit app for analyzing customer service transcripts with automatic language detection and translation.

## Features
- 15 Category Classification
- 5-Level Sentiment Analysis
- 🌍 Multi-language Support (NEW!)
- Automatic Translation to English
- Parquet Output

## Usage
1. Upload CSV/Excel file (columns: `Conversation Id`, `transcripts`)
2. Enable translation in sidebar
3. Click "Run Analysis"
4. Download results

## Live Demo
[Launch App](https://venkatbilla2008-nlp-text-pipeline.streamlit.app)

## Local Development
\`\`\`bash
pip install -r requirements.txt
streamlit run streamlit_nlp_app.py
\`\`\`
```

#### File 4: .gitignore

Create `.gitignore`:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
.Python

# Virtual Environment
venv/
env/

# Streamlit
.streamlit/secrets.toml

# Output files
nlp_results/
*.parquet
*.csv

# IDE
.vscode/
.idea/
.DS_Store

# Logs
*.log
EOF
```

#### Optional: .streamlit/config.toml

```bash
# Create .streamlit directory
mkdir -p .streamlit

# Create config
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
maxUploadSize = 200
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
```

### Step 5: Verify Files

```bash
# Check files are there
ls -la

# Should show:
# streamlit_nlp_app.py  ✓
# requirements.txt      ✓
# README.md            ✓
# .gitignore           ✓
# .streamlit/          ✓ (optional)
```

### Step 6: Git Add, Commit, Push

```bash
# Add all files
git add .

# Check what will be committed
git status

# Should show:
# new file: streamlit_nlp_app.py
# new file: requirements.txt
# new file: README.md
# new file: .gitignore

# Commit
git commit -m "Update: NLP app with translation support"

# Push to GitHub
git push origin main
```

**If you get an error about remote changes:**

```bash
# Pull first
git pull origin main --allow-unrelated-histories

# Then push
git push origin main
```

**If you want to force overwrite (use carefully!):**

```bash
git push origin main --force
```

### Step 7: Verify on GitHub

1. Go to: https://github.com/venkatbilla2008/nlp-text-pipeline
2. Check files are there:
   - ✅ streamlit_nlp_app.py
   - ✅ requirements.txt
   - ✅ README.md
   - ✅ .gitignore

---

## 🎯 After Pushing to GitHub

### Streamlit Cloud Will Auto-Deploy

1. **Wait 2-3 minutes** - Streamlit Cloud detects the change
2. **Auto-redeploy starts** - Watch at https://share.streamlit.io
3. **Check logs** - Go to your app → Manage app → Logs
4. **Test when ready** - Open your app URL

### If App Exists Already

Your app at Streamlit Cloud will **automatically redeploy** with the new code!

**Check deployment:**
1. Go to https://share.streamlit.io
2. Click on your app
3. Click "Manage app"
4. View "Logs" tab
5. Wait for "Your app is live!" message

### If You Need to Create New App

1. Go to https://share.streamlit.io
2. Click "New app"
3. **Repository:** venkatbilla2008/nlp-text-pipeline
4. **Branch:** main
5. **Main file:** streamlit_nlp_app.py
6. Click "Deploy!"

---

## ✅ Verification Checklist

### Before Pushing
- [ ] `streamlit_nlp_app.py` has your modified code
- [ ] `requirements.txt` includes `googletrans==4.0.0rc1`
- [ ] `README.md` mentions translation feature
- [ ] `.gitignore` created
- [ ] All 4 files in repository folder

### After Pushing
- [ ] Files visible on GitHub
- [ ] No error messages on GitHub
- [ ] Streamlit Cloud shows "Redeploying"
- [ ] App logs show successful installation
- [ ] Can access app URL

### App Testing
- [ ] App loads successfully
- [ ] Can upload file
- [ ] Translation toggle works
- [ ] Analysis runs without errors
- [ ] Can download results
- [ ] Translated_Text column appears in output

---

## 🐛 Troubleshooting

### Issue 1: "googletrans" Installation Fails

**Symptoms:** Streamlit Cloud shows "ERROR: Could not find a version"

**Fix:**
```bash
# Update requirements.txt to use this version:
googletrans==4.0.0rc1

# Or try alternative:
googletrans==3.1.0a0
```

### Issue 2: Translation Rate Limit

**Symptoms:** Too many translation requests fail

**Fix:** In sidebar settings:
- Increase "Translation delay" to 1.0 seconds
- Reduce "Number of threads" to 2

### Issue 3: Git Push Rejected

**Symptoms:** "Updates were rejected"

**Fix:**
```bash
# Option 1: Pull and merge
git pull origin main
git push origin main

# Option 2: Force push (careful!)
git push origin main --force
```

### Issue 4: NLTK Data Error

**Symptoms:** "Resource 'punkt' not found"

**Solution:** Already handled! Your code has:
```python
@st.cache_resource
def download_nltk_data():
    ...
```

This auto-downloads on first run.

### Issue 5: Old Files Still Showing

**If old files still show on GitHub:**

```bash
# Remove them explicitly
git rm old_file.py
git commit -m "Remove old files"
git push origin main
```

---

## 📊 File Comparison: Before vs After

### Before (Old Repository)
```
❌ mercury_app_code_review.md
❌ MERCURY_VS_STREAMLIT.md
❌ PROJECT_SUMMARY.md
❌ setup.sh
❌ setup.bat
❌ ...and many more docs
✅ streamlit_nlp_app.py (old version)
✅ requirements.txt (old)
```

### After (Clean Repository)
```
✅ streamlit_nlp_app.py (with translation!)
✅ requirements.txt (with googletrans)
✅ README.md (updated)
✅ .gitignore
✅ .streamlit/config.toml (optional)
```

**Result:** Clean, professional repository with only what's needed!

---

## 🎉 Success Indicators

You're successful when:

✅ **GitHub shows 4-5 files only**  
✅ **Streamlit Cloud shows "Your app is live!"**  
✅ **App has translation toggle in sidebar**  
✅ **Can upload file and see translation**  
✅ **Output has "Translated_Text" column**  
✅ **No errors in Streamlit logs**  

---

## 📥 Quick Download Links

### Essential Files to Download:

1. **Updated requirements.txt:**  
   [Download requirements_with_translation.txt](computer:///mnt/user-data/outputs/requirements_with_translation.txt)  
   *(Rename to `requirements.txt`)*

2. **Updated README.md:**  
   [Download README_with_translation.md](computer:///mnt/user-data/outputs/README_with_translation.md)  
   *(Rename to `README.md`)*

3. **Automated update script:**  
   [Download update_github_repo.sh](computer:///mnt/user-data/outputs/update_github_repo.sh)  
   *(Makes everything automatic!)*

4. **Your modified app:**  
   Save your code from the document as `streamlit_nlp_app.py`

---

## 🚀 Quick Commands (Copy-Paste)

### Complete Update in One Go:

```bash
# Clone repository
git clone https://github.com/venkatbilla2008/nlp-text-pipeline.git
cd nlp-text-pipeline

# Backup old files
mkdir ../backup && cp -r * ../backup/

# Remove old files
rm -rf *

# Now copy your 4 files here:
# - streamlit_nlp_app.py
# - requirements.txt
# - README.md
# - .gitignore

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
venv/
.streamlit/secrets.toml
nlp_results/
*.parquet
*.csv
.DS_Store
*.log
EOF

# Git operations
git add .
git commit -m "Update: NLP app with translation support"
git push origin main

# Done! Check GitHub and Streamlit Cloud
```

---

## 📞 Need Help?

### Common Questions

**Q: Will this delete my old work?**  
A: Old files are removed from GitHub, but you can keep backup locally.

**Q: How long does deployment take?**  
A: 2-5 minutes for first deploy, 1-2 minutes for updates.

**Q: Do I need to redeploy manually?**  
A: No! Streamlit Cloud auto-deploys when you push to GitHub.

**Q: What if translation is slow?**  
A: Increase translation delay in sidebar (0.5s → 1.0s).

**Q: Can I keep documentation files locally?**  
A: Yes! Keep them on your computer, just don't push to GitHub.

---

## ✅ Final Checklist

Complete update checklist:

- [ ] Cloned repository locally
- [ ] Backed up old files (optional)
- [ ] Removed all old files
- [ ] Added 4 essential files
- [ ] Verified files are correct
- [ ] Committed changes
- [ ] Pushed to GitHub successfully
- [ ] Verified files on github.com
- [ ] Checked Streamlit Cloud deployment
- [ ] Tested app with translation
- [ ] Downloaded results successfully

---

**Your repository:** https://github.com/venkatbilla2008/nlp-text-pipeline  
**Streamlit Cloud:** https://share.streamlit.io

**You're all set! Your app with translation support will be live in minutes! 🎉**
