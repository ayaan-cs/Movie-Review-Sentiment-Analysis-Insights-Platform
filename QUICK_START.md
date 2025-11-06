# 🚀 Quick Start Guide

This guide will help you get the IMDB Sentiment Analysis project up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- 2GB+ free disk space

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Movie-Review-Sentiment-Analysis-Insights-Platform
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages:
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- transformers, torch (deep learning)
- nltk, textblob (NLP)
- matplotlib, seaborn, plotly (visualization)
- scipy (statistics)

### 4. Run the Analysis
```bash
python imdb_sentiment_analysis.py
```

## What to Expect

When you run the script, it will:

1. **Download NLTK resources** (~10 seconds)
   - punkt, stopwords, wordnet, etc.

2. **Load IMDB dataset** (~30 seconds)
   - Downloads 50,000 movie reviews from Hugging Face

3. **Preprocess data** (~2-3 minutes)
   - Text cleaning, tokenization, lemmatization
   - Feature engineering

4. **Generate EDA visualizations** (~1-2 minutes)
   - Creates 6 HTML and 6 PNG visualizations

5. **Train ML models** (~5-8 minutes)
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - Gradient Boosting

6. **Run transformer model** (~3-5 minutes)
   - DistilBERT predictions on 1000 samples

7. **Topic modeling** (~2-3 minutes)
   - LDA with 5 topics

8. **Statistical analysis** (~30 seconds)
   - T-tests, chi-square tests, correlations

9. **Generate final report** (~10 seconds)
   - Comprehensive markdown report

**Total time: 15-20 minutes (CPU) or 8-12 minutes (GPU)**

## Output Files

All results are saved in `analysis_results/` folder:

### Interactive HTML Visualizations
- `01_sentiment_distribution.html`
- `02_word_count_by_sentiment.html`
- `03_polarity_vs_subjectivity.html`
- `04_review_length_distribution.html`
- `10_model_comparison.html`
- `11_roc_curves.html`
- `13_transformer_confidence.html`
- `14_topic_distribution.html`
- `15_topic_sentiment.html`

### Static PNG Images
- `05_statistical_features.png`
- `06_correlation_matrix.png`
- `07_wordclouds.png`
- `08_top_bigrams.png`
- `09_top_trigrams.png`
- `12_confusion_matrix.png`

### Reports & Data
- `FINAL_REPORT.md` - Comprehensive analysis report
- `sample_processed_data.csv` - Sample of processed data
- `model_comparison.csv` - Model performance metrics

## Viewing Results

### HTML Files
Simply open any `.html` file in your web browser:
```bash
# On Linux
xdg-open analysis_results/01_sentiment_distribution.html

# On Mac
open analysis_results/01_sentiment_distribution.html

# On Windows
start analysis_results/01_sentiment_distribution.html
```

### PNG Files
Open with any image viewer or include in presentations.

### Markdown Report
View in any markdown viewer or text editor:
```bash
cat analysis_results/FINAL_REPORT.md
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you've installed all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory"
**Solution**: The transformer model is run on a sample of 1000 reviews. If still having issues, reduce `sample_size` in the script.

### Issue: NLTK data download fails
**Solution**: Manually download NLTK data:
```python
import nltk
nltk.download('all')
```

### Issue: Slow execution
**Solution**:
- Use GPU if available (automatically detected)
- Reduce sample sizes in the script
- Run on a machine with more RAM

### Issue: Hugging Face download slow
**Solution**: Set Hugging Face token (optional):
```bash
export HUGGING_FACE_TOKEN="your_token_here"
```

## Next Steps

After running the analysis:

1. **Explore visualizations** in `analysis_results/` folder
2. **Read the comprehensive report** (`FINAL_REPORT.md`)
3. **Modify parameters** in the script to experiment
4. **Try different models** from Hugging Face
5. **Extend the analysis** with your own features

## Customization

### Change Number of Topics (LDA)
```python
n_topics = 5  # Change to 3, 7, 10, etc.
```

### Use Different Transformer Model
```python
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="roberta-base-sentiment",  # or any other model
    device=0 if torch.cuda.is_available() else -1
)
```

### Adjust TF-IDF Parameters
```python
tfidf = TfidfVectorizer(
    max_features=5000,  # Increase for more features
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)  # Change to (1, 3) for trigrams
)
```

### Increase Transformer Sample Size
```python
sample_size = 1000  # Increase to 2000, 5000, etc. (slower)
```

## Performance Tips

1. **Use GPU**: Automatically detected for transformer models
2. **Close other applications**: Free up RAM
3. **Use SSD**: Faster data I/O
4. **Increase batch size**: If you have more RAM

## Support

If you encounter any issues:

1. Check the error message carefully
2. Review this guide
3. Check README.md for more details
4. Open an issue on GitHub

## Happy Analyzing! 🎉

Your comprehensive sentiment analysis results will be ready in ~15 minutes!
