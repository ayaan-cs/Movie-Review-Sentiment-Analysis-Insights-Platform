# Setup Instructions

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Movie-Review-Sentiment-Analysis-Insights-Platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis script** (generates models and results)
   ```bash
   python imdb_sentiment_analysis.py
   ```
   This will:
   - Download the IMDB dataset (40,000 reviews)
   - Train machine learning models
   - Generate visualizations and reports
   - Save models to `models/` folder
   - Create analysis results in `analysis_results/` folder
   
   **Expected time**: 15-20 minutes on CPU, 8-12 minutes on GPU

5. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`

## What Gets Generated

After running `imdb_sentiment_analysis.py`, you'll have:

- **`models/`** folder:
  - `best_model.pkl` - Trained Logistic Regression model
  - `tfidf_vectorizer.pkl` - Text vectorizer

- **`analysis_results/`** folder:
  - 6 interactive HTML visualizations
  - 6 static PNG images
  - `FINAL_REPORT.md` - Comprehensive analysis report
  - `model_comparison.csv` - Model performance metrics
  - `sample_processed_data.csv` - Sample of processed data

## Troubleshooting

### Models not found
If you see "ML model not available" in the Streamlit app, make sure you've run:
```bash
python imdb_sentiment_analysis.py
```

### Missing dependencies
If you get import errors, ensure all packages are installed:
```bash
pip install -r requirements.txt
```

### NLTK data issues
The script automatically downloads NLTK data, but if it fails:
```python
import nltk
nltk.download('all')
```

## Next Steps

- Read `README.md` for project overview
- Check `QUICK_START.md` for detailed usage guide
- See `STREAMLIT_DEPLOYMENT.md` to deploy the app online

