# 🚀 GitHub & Streamlit Cloud Deployment Guide

Complete step-by-step guide to deploy your NLP Text Classification Dashboard to GitHub and Streamlit Cloud.

---

## 📋 Quick Overview

**What we'll do:**
1. Push your code to GitHub
2. Connect GitHub to Streamlit Cloud
3. Deploy your app (it will be live!)
4. Get a public URL to share

**Time required:** 15-20 minutes  
**Cost:** FREE (both GitHub and Streamlit Cloud are free)

---

## ✅ Prerequisites

### You Need:
- ✅ GitHub account (you have this!)
- ⬜ Git installed on your computer
- ⬜ Your project files downloaded locally

### Check Git Installation:
```bash
git --version
```

Should show: `git version 2.x.x`

### Install Git if needed:

**Mac:**
```bash
brew install git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install git
```

**Windows:**
Download from: https://git-scm.com/download/win

---

## 🐙 Part 1: Push to GitHub

### Step 1: Create Repository on GitHub

1. Go to **https://github.com**
2. Click the **"+"** icon (top right)
3. Select **"New repository"**

4. **Fill in details:**
   ```
   Repository name: nlp-text-classification
   Description: NLP Text Classification Dashboard with Sentiment Analysis
   
   Visibility: ● Public (REQUIRED for free Streamlit Cloud)
   
   ☐ Do NOT check "Add a README file"
   ☐ Do NOT add .gitignore
   ☐ Do NOT choose a license
   
   (We already have these files!)
   ```

5. Click **"Create repository"**

6. **Save your repository URL** (you'll need it):
   ```
   https://github.com/YOUR_USERNAME/nlp-text-classification
   ```

### Step 2: Prepare Your Files Locally

1. **Open terminal/command prompt**

2. **Navigate to where you saved the files:**
   ```bash
   cd /path/to/your/downloaded/files
   ```

3. **Verify files are there:**
   ```bash
   ls
   ```
   
   You should see:
   - streamlit_nlp_app.py
   - requirements.txt
   - README.md
   - setup.sh
   - setup.bat
   - generate_sample_data.py
   - All the .md documentation files

### Step 3: Initialize Git

```bash
# Initialize git repository
git init

# Add all files
git add .

# Check what will be committed
git status
```

You should see all your files listed in green.

### Step 4: Commit Your Files

```bash
git commit -m "Initial commit: NLP Text Classification Dashboard"
```

### Step 5: Connect to GitHub

**Replace YOUR_USERNAME with your actual GitHub username:**

```bash
git remote add origin https://github.com/YOUR_USERNAME/nlp-text-classification.git

# Set branch name to main
git branch -M main
```

### Step 6: Push to GitHub

```bash
git push -u origin main
```

**You'll be prompted for credentials:**
- **Username:** Your GitHub username
- **Password:** Your Personal Access Token (see below)

### Getting a Personal Access Token (PAT)

GitHub no longer accepts passwords for git operations. You need a token:

1. Go to: **https://github.com/settings/tokens**
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. **Settings:**
   ```
   Note: Streamlit App Deployment
   Expiration: 90 days (or your choice)
   
   Select scopes:
   ✅ repo (check ALL repo access)
   ```
4. Click **"Generate token"**
5. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
6. Use this token as your password when git asks

### Step 7: Verify on GitHub

1. Go to: `https://github.com/YOUR_USERNAME/nlp-text-classification`
2. You should see all your files!
3. README.md should be displayed at the bottom
4. Click through files to verify everything uploaded

✅ **Part 1 Complete!** Your code is now on GitHub!

---

## ☁️ Part 2: Deploy to Streamlit Cloud

### Step 1: Sign Up for Streamlit Cloud

1. Go to: **https://share.streamlit.io**

2. Click **"Sign up"** or **"Get started"**

3. **Choose "Continue with GitHub"**
   - Authorize Streamlit to access your GitHub
   - Grant permissions when asked

4. **Complete your profile**
   - Verify your email if prompted
   - Fill in any required information

### Step 2: Create New App

1. From **Streamlit Cloud dashboard**, click **"New app"**

2. **Fill in deployment settings:**
   ```
   Repository: YOUR_USERNAME/nlp-text-classification
   Branch: main
   Main file path: streamlit_nlp_app.py
   
   App URL (optional): Choose a custom name or use default
   Example: my-nlp-classifier.streamlit.app
   ```

3. **Click "Advanced settings" (optional but recommended):**
   ```
   Python version: 3.11
   ```

4. Click **"Deploy!"**

### Step 3: Watch Deployment

You'll see a live log showing:
1. ✅ Cloning repository
2. ✅ Installing Python 3.11
3. ✅ Installing dependencies from requirements.txt
4. ✅ Downloading NLTK data
5. ✅ Starting app

**This takes 2-5 minutes on first deployment.**

### Step 4: Your App is Live!

Once complete, you'll see:
```
🎉 Your app is live at:
https://your-app-name.streamlit.app
```

**Test it:**
1. Click the URL
2. Try uploading a file
3. Run the analysis
4. Verify everything works

✅ **Part 2 Complete!** Your app is now live on the internet!

---

## 🎨 Part 3: Optional Configuration

### Add Streamlit Configuration

Create better app settings:

1. **In your local project folder, create:**
   ```bash
   mkdir -p .streamlit
   ```

2. **Create config file:**
   ```bash
   cat > .streamlit/config.toml << 'EOF'
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#262730"
   font = "sans serif"
   
   [server]
   maxUploadSize = 200
   enableXsrfProtection = true
   
   [browser]
   gatherUsageStats = false
   EOF
   ```

3. **Push to GitHub:**
   ```bash
   git add .streamlit/config.toml
   git commit -m "Add Streamlit configuration"
   git push
   ```

4. **Streamlit Cloud will auto-redeploy** (takes 1-2 minutes)

### Fix NLTK Data Download Issue

If you get "NLTK data not found" errors:

**Add to the TOP of streamlit_nlp_app.py** (after imports):

```python
import nltk
import ssl

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
```

Then push changes:
```bash
git add streamlit_nlp_app.py
git commit -m "Fix NLTK data download"
git push
```

---

## 🔄 Updating Your App

### Make Changes Locally

1. **Edit your files:**
   ```bash
   # Edit any file
   nano streamlit_nlp_app.py
   ```

2. **Test locally:**
   ```bash
   streamlit run streamlit_nlp_app.py
   ```

### Push Changes

```bash
# Add changed files
git add .

# Commit with descriptive message
git commit -m "Update: improved category detection"

# Push to GitHub
git push
```

**Streamlit Cloud automatically redeploys!** (1-2 minutes)

---

## 🐛 Troubleshooting

### Issue 1: "Permission denied" when pushing to GitHub

**Solution:** Use Personal Access Token as password (see Part 1, Step 6)

### Issue 2: "Requirements file not found"

**Check:**
```bash
# Verify file exists
ls -la requirements.txt

# Should show: requirements.txt in root directory
```

**Fix:**
```bash
git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

### Issue 3: "Module not found" errors in Streamlit Cloud

**Check requirements.txt has all dependencies:**
```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
textblob>=0.17.0
afinn>=0.1
langdetect>=1.0.9
openpyxl>=3.1.0
pyarrow>=14.0.0
python-snappy>=0.6.1
```

**Update and push:**
```bash
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Issue 4: NLTK data errors

**Solution:** Add NLTK download code (see "Fix NLTK Data" in Part 3)

### Issue 5: App is slow on Streamlit Cloud

**Solutions:**

1. **Add caching to your app:**
   ```python
   @st.cache_data
   def load_data(file):
       return pd.read_csv(file)
   ```

2. **Reduce thread count for cloud:**
   ```python
   # In streamlit_nlp_app.py, modify Config class:
   import os
   
   class Config:
       # Use fewer threads on Streamlit Cloud
       NUM_THREADS = 2 if os.environ.get("STREAMLIT_CLOUD") else 8
   ```

### Issue 6: Cannot push to GitHub (authentication fails)

**Option 1: Use HTTPS with token**
```bash
git push
Username: YOUR_USERNAME
Password: YOUR_PERSONAL_ACCESS_TOKEN
```

**Option 2: Use SSH** (recommended for frequent pushes)
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/nlp-text-classification.git

# Now push works without password
git push
```

---

## 📱 Sharing Your App

### Your App URL
```
https://your-app-name.streamlit.app
```

### Ways to Share:

1. **Direct link** - Share the URL directly
2. **QR Code** - Generate QR code for mobile access
3. **Social media** - Share on LinkedIn, Twitter, etc.
4. **Email** - Send link to colleagues
5. **Portfolio** - Add to your portfolio website

### Make it Look Professional:

**Update README.md on GitHub** with a live demo link:
```markdown
# NLP Text Classification Dashboard

🚀 **[Live Demo](https://your-app-name.streamlit.app)**

Analyze customer service transcripts with sentiment analysis...
```

---

## 📊 Monitoring Your App

### View Analytics

1. Go to **Streamlit Cloud dashboard**
2. Click on your app
3. Click **"Analytics"** tab

See:
- Number of viewers
- Usage over time
- Peak usage times
- Geographic distribution

### View Logs

1. In app dashboard, click **"Manage app"**
2. Click **"Logs"** tab
3. See real-time app logs and errors

### Restart App

If app becomes unresponsive:
1. Click **"Manage app"**
2. Click **"⋮" (three dots)**
3. Click **"Reboot app"**

---

## ✅ Complete Deployment Checklist

### GitHub Setup
- [ ] Repository created on GitHub
- [ ] Git initialized locally (`git init`)
- [ ] Files added (`git add .`)
- [ ] Initial commit made (`git commit`)
- [ ] Remote added (`git remote add origin`)
- [ ] Code pushed (`git push`)
- [ ] Files visible on GitHub.com

### Streamlit Cloud Setup
- [ ] Streamlit Cloud account created
- [ ] GitHub account connected
- [ ] New app created
- [ ] Repository selected
- [ ] Main file set to `streamlit_nlp_app.py`
- [ ] App deployed successfully
- [ ] App URL works
- [ ] Can upload files
- [ ] Analysis works correctly

### Post-Deployment
- [ ] Shared URL with team/friends
- [ ] Bookmarked Streamlit dashboard
- [ ] Tested all features
- [ ] Checked logs for errors
- [ ] Updated README with live demo link

---

## 🎯 Quick Commands Reference

### Git Basics
```bash
# Check status
git status

# Add all files
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# Pull latest changes
git pull

# View remote URL
git remote -v
```

### Local Testing
```bash
# Test app locally
streamlit run streamlit_nlp_app.py

# Check Python version
python --version

# List installed packages
pip list
```

### Streamlit Cloud
```
# All done through web interface at:
https://share.streamlit.io

# View your apps
Dashboard → My apps

# View logs
App → Manage app → Logs

# Restart app
App → Manage app → ⋮ → Reboot
```

---

## 🚀 What You've Accomplished

After completing this guide, you have:

✅ **Code on GitHub** (version controlled, backed up)  
✅ **Live web app** (accessible from anywhere)  
✅ **Public URL** (share with anyone)  
✅ **Free hosting** (no cost!)  
✅ **Auto-deployment** (push to GitHub = auto-update)  
✅ **Professional portfolio piece** (show to employers!)  

**Your app is now:**
- 🌍 Accessible worldwide
- 📱 Works on any device
- 🔄 Automatically updates
- 📊 Has analytics
- 💰 Costs $0

---

## 🎓 Next Steps

### Immediate
1. ✅ Share your app URL
2. ✅ Test with real data
3. ✅ Get feedback from users

### This Week
1. Customize the UI
2. Add more features
3. Improve performance
4. Update documentation

### Advanced
1. Add authentication
2. Connect to database
3. Add more ML models
4. Create API endpoint
5. Custom domain

---

## 📞 Support & Resources

### Streamlit
- **Docs:** https://docs.streamlit.io
- **Forum:** https://discuss.streamlit.io
- **Gallery:** https://streamlit.io/gallery
- **Cheat Sheet:** https://docs.streamlit.io/library/cheatsheet

### GitHub
- **Docs:** https://docs.github.com
- **Learning:** https://skills.github.com

### Your App
- **Dashboard:** https://share.streamlit.io
- **Logs:** In app settings
- **Analytics:** In app dashboard

---

## 💡 Pro Tips

### 1. Use Meaningful Commit Messages
```bash
# Bad
git commit -m "update"

# Good
git commit -m "Add confidence scoring to category predictions"
```

### 2. Test Locally Before Pushing
```bash
streamlit run streamlit_nlp_app.py
# Make sure it works, then push
```

### 3. Keep README Updated
Update README.md with:
- Live demo link
- Screenshots
- Usage instructions
- Features list

### 4. Monitor Your App
Check logs regularly:
- Look for errors
- Monitor performance
- Track usage patterns

### 5. Optimize for Cloud
```python
# Use caching
@st.cache_data
def expensive_computation():
    ...

# Reduce threads on cloud
NUM_THREADS = 2  # Instead of 8
```

---

## 🎉 Success!

**Congratulations!** Your NLP Text Classification Dashboard is now:

✨ **Live on the internet**  
✨ **Hosted on Streamlit Cloud (free!)**  
✨ **Version controlled on GitHub**  
✨ **Ready to share with the world**  

**Your Live App:**
```
🌐 https://your-app-name.streamlit.app
```

**Your GitHub Repo:**
```
📁 https://github.com/YOUR_USERNAME/nlp-text-classification
```

Share these links and show off your work! 🚀

---

**Need help?** Check the troubleshooting section or Streamlit documentation.

**Want to improve?** See the "Next Steps" section above.

**Ready to deploy?** Follow the checklist and you'll be live in 20 minutes!

---

**Version:** 1.0.0  
**Last Updated:** November 2024  
**Deployment Time:** 15-20 minutes  
**Cost:** FREE
