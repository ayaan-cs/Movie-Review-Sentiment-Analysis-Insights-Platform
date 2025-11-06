# Streamlit Dashboard Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main analysis script first** (to generate models and results):
   ```bash
   python imdb_sentiment_analysis.py
   ```
   This will create:
   - `models/best_model.pkl` - Trained ML model
   - `models/tfidf_vectorizer.pkl` - Text vectorizer
   - `analysis_results/` - All analysis outputs

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`

## 📱 Using the Dashboard

### 1. Predict Sentiment Page
- Enter any movie review text
- Click "Analyze Sentiment"
- View predictions from both ML and Transformer models
- See text analysis metrics (word count, polarity, subjectivity)

### 2. Analysis Results Page
- View model performance comparison
- See interactive visualizations
- Browse sample processed data

### 3. About Page
- Learn about the project
- View project structure
- See deployment information

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Easiest & Recommended) ⭐⭐⭐

**Steps:**

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Streamlit dashboard"
   git push origin main
   ```

2. **Go to [streamlit.io](https://streamlit.io)**
   - Sign up/Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Your app is live!** 🎉
   - Free hosting
   - Automatic updates on git push
   - Public URL to share

**Requirements:**
- Repository must be public (free tier)
- `requirements.txt` must be in root
- `app.py` must be the main file

### Option 2: Heroku

**Steps:**

1. **Create `Procfile`**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `setup.sh`**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

4. **Scale dyno**:
   ```bash
   heroku ps:scale web=1
   ```

**Cost:** Free tier available, then $7+/month

### Option 3: Railway

**Steps:**

1. **Connect GitHub repository** to Railway
2. **Set start command**:
   ```
   streamlit run app.py --server.port=$PORT
   ```
3. **Deploy** - Railway auto-detects Python apps

**Cost:** $5+/month

### Option 4: Docker

**Create `Dockerfile`**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run**:
```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

## 📋 Pre-Deployment Checklist

- [ ] Run `python imdb_sentiment_analysis.py` to generate models
- [ ] Verify `models/` folder exists with `.pkl` files
- [ ] Verify `analysis_results/` folder exists
- [ ] Test app locally: `streamlit run app.py`
- [ ] Check all visualizations load correctly
- [ ] Test prediction functionality
- [ ] Verify `requirements.txt` includes all dependencies

## 🔧 Troubleshooting

### Issue: "Model not found" error
**Solution**: Run the main analysis script first to generate models:
```bash
python imdb_sentiment_analysis.py
```

### Issue: Transformer model loading slowly
**Solution**: This is normal on first run. The model downloads from Hugging Face (~250MB). Subsequent runs will be faster.

### Issue: NLTK data not found
**Solution**: The app automatically downloads NLTK data, but if it fails:
```python
import nltk
nltk.download('all')
```

### Issue: App crashes on deployment
**Solution**: 
- Check that all files are in repository
- Verify `requirements.txt` is up to date
- Check logs for specific errors
- Ensure enough memory (Streamlit Cloud provides 1GB free)

### Issue: Slow predictions
**Solution**: 
- Use GPU for transformer model (if available)
- Reduce batch size for transformer
- Consider caching more aggressively

## 🎨 Customization

### Change Theme
Edit `app.py` and add to `st.set_page_config`:
```python
st.set_page_config(
    theme={
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "font": "sans serif"
    }
)
```

### Add More Pages
Create new sections in the sidebar radio button:
```python
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Predict", "📈 Results", "📋 About", "🆕 New Page"]
)
```

### Add Authentication
Use Streamlit's built-in authentication or external libraries like `streamlit-authenticator`

## 📊 Performance Tips

1. **Use caching**: Already implemented with `@st.cache_resource` and `@st.cache_data`
2. **Lazy load models**: Models load only when needed
3. **Batch predictions**: For multiple reviews, process in batches
4. **Optimize visualizations**: Use Plotly for interactive charts

## 🔒 Security Considerations

- **Don't commit sensitive data**: Use environment variables
- **Rate limiting**: Consider adding rate limits for production
- **Input validation**: Validate user inputs
- **Error handling**: Don't expose internal errors to users

## 📈 Monitoring

### Streamlit Cloud
- Built-in analytics
- View app usage
- Error tracking

### Custom Monitoring
- Add logging
- Track predictions
- Monitor model performance
- Set up alerts

## 🚀 Next Steps

1. **Deploy to Streamlit Cloud** (easiest)
2. **Share your app URL** with others
3. **Collect feedback** and improve
4. **Add more features** (user authentication, history, etc.)
5. **Scale up** if needed

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Example Apps](https://streamlit.io/gallery)

---

**Happy Deploying! 🎉**

