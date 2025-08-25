# 🚀 Deployment Guide for Advanced Stock Analyzer Pro

This guide will help you deploy your stock analysis application to Streamlit Cloud and GitHub.

## Prerequisites

1. **GitHub Account**: Create an account at [github.com](https://github.com)
2. **Streamlit Cloud Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website
1. Go to [github.com](https://github.com) and log in
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., "advanced-stock-analyzer")
4. Add a description: "AI-powered stock analysis application with real-time data and technical indicators"
5. Choose "Public" or "Private" visibility
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### Option B: Using GitHub CLI (if installed)
```bash
# Install GitHub CLI first if not installed
# Windows: winget install --id GitHub.cli
# macOS: brew install gh
# Linux: sudo apt install gh

gh auth login
gh repo create advanced-stock-analyzer --description "AI-powered stock analysis application" --public
```

## Step 2: Push Code to GitHub

After creating the repository, follow the instructions to push your existing repository:

```bash
# Add the remote origin (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/advanced-stock-analyzer.git

# Rename branch to main (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: "advanced-stock-analyzer"
5. Select branch: "main"
6. Main file path: "advanced_stock_analyzer.py"
7. Click "Deploy"

## Step 4: Configure Environment (Optional)

If you need to add environment variables or secrets:

### For Streamlit Cloud:
1. Go to your app on Streamlit Cloud
2. Click "Settings" (gear icon)
3. Add any required secrets in the "Secrets" section

### For Local Development:
Create a `.streamlit/secrets.toml` file (not committed to Git):

```toml
# API keys or other secrets
SECRET_KEY = "your-secret-key-here"
```

## Step 5: Test Your Deployment

1. Visit your Streamlit Cloud URL (e.g., `https://your-app-name.streamlit.app`)
2. Test all features to ensure everything works
3. Check the logs in Streamlit Cloud dashboard for any errors

## Alternative Deployment Options

### Heroku Deployment
```bash
# Create a Procfile
echo "web: streamlit run advanced_stock_analyzer.py --server.port=\$PORT" > Procfile

# Add to requirements.txt
echo "gunicorn" >> requirements.txt

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "advanced_stock_analyzer.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Ensure all dependencies are in `requirements.txt`
2. **Port already in use**: Streamlit runs on port 8501 by default
3. **API rate limits**: Implement proper error handling for external APIs
4. **Large file uploads**: Check if any large files need to be in `.gitignore`

### Debugging:
- Check Streamlit Cloud logs in the dashboard
- Test locally first: `streamlit run advanced_stock_analyzer.py`
- Use `print()` statements for debugging

## Maintenance

1. **Updates**: Push changes to GitHub, Streamlit Cloud will auto-deploy
2. **Monitoring**: Check Streamlit Cloud dashboard for usage and errors
3. **Dependencies**: Keep `requirements.txt` updated with all necessary packages

## Security Considerations

- Never commit API keys or secrets to GitHub
- Use environment variables for sensitive information
- Implement rate limiting for API calls
- Validate user inputs to prevent injection attacks

---

**Note**: This application is for educational purposes. Always follow best practices for production deployment and security.
