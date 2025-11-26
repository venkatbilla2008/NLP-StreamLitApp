#!/bin/bash
# update_github_repo.sh
# Script to update your GitHub repository with only essential files

echo "=========================================="
echo "🚀 Update GitHub Repository"
echo "=========================================="
echo ""

# Configuration
REPO_URL="https://github.com/venkatbilla2008/nlp-text-pipeline.git"
REPO_DIR="nlp-text-pipeline"
BACKUP_DIR="../backup-$(date +%Y%m%d_%H%M%S)"

echo "Repository: $REPO_URL"
echo "Local directory: $REPO_DIR"
echo ""

# Step 1: Clone or navigate to repository
if [ -d "$REPO_DIR" ]; then
    echo "✓ Repository directory exists"
    cd "$REPO_DIR"
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi

echo "✓ In repository directory: $(pwd)"
echo ""

# Step 2: Backup existing files
echo "📦 Creating backup of current files..."
mkdir -p "$BACKUP_DIR"
cp -r * "$BACKUP_DIR/" 2>/dev/null
echo "✓ Backup created at: $BACKUP_DIR"
echo ""

# Step 3: Remove all files (keep .git)
echo "🗑️  Removing old files..."
find . -maxdepth 1 -not -name ".git" -not -name "." -not -name ".." -exec rm -rf {} +
echo "✓ Old files removed"
echo ""

# Step 4: Check for required files
echo "📋 Checking for required files..."
echo ""
echo "Please ensure you have these files ready:"
echo "  1. streamlit_nlp_app.py (your modified code)"
echo "  2. requirements.txt (with googletrans)"
echo "  3. README.md"
echo ""

read -p "Do you have all files ready? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Please prepare the files first, then run this script again"
    exit 1
fi

echo ""
echo "📁 Please copy your files now:"
echo ""
echo "Copy to: $(pwd)"
echo ""
echo "Required files:"
echo "  • streamlit_nlp_app.py"
echo "  • requirements.txt"
echo "  • README.md"
echo ""

read -p "Press Enter when files are copied..."
echo ""

# Step 5: Create .gitignore
echo "✓ Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyc
.Python
*.so

# Virtual Environment
venv/
env/
ENV/
.venv

# Streamlit
.streamlit/secrets.toml

# Output files
nlp_results/
*.parquet
*.csv
*.xlsx

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Logs
*.log

# OS
Thumbs.db
EOF
echo "✓ .gitignore created"
echo ""

# Step 6: Create Streamlit config (optional)
echo "✓ Creating .streamlit/config.toml..."
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
echo "✓ Streamlit config created"
echo ""

# Step 7: Verify files
echo "📋 Verifying files..."
echo ""

if [ ! -f "streamlit_nlp_app.py" ]; then
    echo "❌ ERROR: streamlit_nlp_app.py not found!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found!"
    exit 1
fi

if [ ! -f "README.md" ]; then
    echo "⚠️  WARNING: README.md not found (recommended)"
fi

echo "✓ Essential files verified"
echo ""

# Step 8: Check git status
echo "📊 Current files:"
ls -lh
echo ""

# Step 9: Git add, commit, push
echo "🔧 Git operations..."
echo ""

# Add all files
git add .

# Show status
echo "Git status:"
git status --short
echo ""

# Commit
read -p "Enter commit message (default: 'Update: NLP app with translation support'): " commit_msg
commit_msg=${commit_msg:-"Update: NLP app with translation support"}

git commit -m "$commit_msg"
echo "✓ Changes committed"
echo ""

# Push
echo "📤 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS!"
    echo "=========================================="
    echo ""
    echo "Your repository has been updated!"
    echo ""
    echo "🌐 View on GitHub:"
    echo "   https://github.com/venkatbilla2008/nlp-text-pipeline"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Go to https://share.streamlit.io"
    echo "   2. Click your app (or create new)"
    echo "   3. It will auto-redeploy (1-2 minutes)"
    echo "   4. Test your app with translation!"
    echo ""
    echo "=========================================="
else
    echo ""
    echo "❌ Push failed! Check error messages above."
    echo ""
    echo "Common fixes:"
    echo "  • git pull origin main (if remote has changes)"
    echo "  • git push origin main --force (if you want to overwrite)"
    echo ""
fi
