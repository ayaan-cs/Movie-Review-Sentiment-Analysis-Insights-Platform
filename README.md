# 🎬 IMDB Movie Review Sentiment Analysis & Insights Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-Transformers-red)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-NLTK-green)](https://www.nltk.org/)

A comprehensive data science project that analyzes 50,000 IMDB movie reviews using advanced NLP, machine learning, and AI-powered insights. This project showcases enterprise-level data engineering, statistical analysis, machine learning, deep learning, and modern AI integration using Hugging Face transformers.

## 🎯 Project Overview

This project demonstrates advanced data science capabilities through:

- **Big Data Processing**: Handling and analyzing 50K+ text reviews
- **Advanced NLP**: Text preprocessing, feature extraction, sentiment analysis
- **Machine Learning**: Training and comparing 4+ ML models
- **Deep Learning**: Using pre-trained transformer models from Hugging Face
- **Statistical Analysis**: Distribution analysis, correlation studies, hypothesis testing
- **Data Visualization**: 15+ professional interactive visualizations
- **AI Integration**: Leveraging Hugging Face models for enhanced insights
- **Production-Ready Code**: Clean architecture, error handling, comprehensive documentation

## 📊 Key Features

### 1. Comprehensive Data Analysis
- Load and process 50,000 IMDB movie reviews from Hugging Face
- Exploratory data analysis with statistical insights
- Missing value analysis and data quality checks
- Class distribution and balance analysis

### 2. Advanced NLP & Feature Engineering
- Text cleaning (HTML tags, URLs, special characters removal)
- Tokenization and lemmatization
- Stop word removal
- TF-IDF vectorization
- N-gram analysis (unigrams, bigrams, trigrams)
- Sentiment polarity and subjectivity scoring
- Word cloud generation

### 3. Machine Learning Models
Four traditional ML models with comprehensive comparison:
- **Logistic Regression**
- **Naive Bayes**
- **Random Forest**
- **Gradient Boosting**

Each model evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### 4. AI-Powered Analysis
- **Hugging Face DistilBERT** transformer model integration
- Zero-shot sentiment classification
- Confidence score analysis
- Traditional ML vs Transformer comparison

### 5. Topic Modeling
- Latent Dirichlet Allocation (LDA) implementation
- Discovery of 5 distinct review topics
- Topic-sentiment relationship analysis
- Topic prevalence visualization

### 6. Statistical Testing
- T-tests for word count differences
- Chi-square tests for categorical relationships
- Correlation analysis
- Hypothesis testing with significance levels

### 7. Professional Visualizations
15+ interactive and static visualizations including:
- Sentiment distribution charts
- Word count box plots
- Polarity vs subjectivity scatter plots
- Feature correlation heatmaps
- Word clouds (positive/negative)
- N-gram frequency charts
- ROC curves
- Confusion matrices
- Model performance comparisons
- Topic distribution charts

## 🛠️ Technical Stack

### Core Libraries
```
Data Processing:     pandas, numpy
Machine Learning:    scikit-learn
Deep Learning:       transformers, torch, datasets
NLP:                 nltk, textblob
Visualization:       matplotlib, seaborn, plotly, wordcloud
Statistical:         scipy
Utilities:           tqdm, jupyter
```

### Models & Algorithms
- **Traditional ML**: Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting
- **Deep Learning**: DistilBERT (Hugging Face)
- **NLP**: TF-IDF, Count Vectorization, LDA
- **Statistical**: T-tests, Chi-square tests, Pearson correlation

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Movie-Review-Sentiment-Analysis-Insights-Platform.git
cd Movie-Review-Sentiment-Analysis-Insights-Platform
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (Automatic)
The script automatically downloads required NLTK resources on first run.

## 🚀 Usage

### Run the Complete Analysis
```bash
python imdb_sentiment_analysis.py
```

This will:
1. Download the IMDB dataset from Hugging Face
2. Perform comprehensive data preprocessing
3. Generate exploratory data analysis visualizations
4. Train 4 machine learning models
5. Run transformer model predictions
6. Perform topic modeling
7. Conduct statistical analysis
8. Generate a comprehensive report
9. Save all results to `analysis_results/` folder

### Expected Runtime
- **On CPU**: ~15-20 minutes
- **On GPU**: ~8-12 minutes

## 📂 Project Structure

```
Movie-Review-Sentiment-Analysis-Insights-Platform/
│
├── imdb_sentiment_analysis.py      # Main analysis script
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── analysis_results/                # Generated outputs
│   ├── 01_sentiment_distribution.html
│   ├── 02_word_count_by_sentiment.html
│   ├── 03_polarity_vs_subjectivity.html
│   ├── 04_review_length_distribution.html
│   ├── 05_statistical_features.png
│   ├── 06_correlation_matrix.png
│   ├── 07_wordclouds.png
│   ├── 08_top_bigrams.png
│   ├── 09_top_trigrams.png
│   ├── 10_model_comparison.html
│   ├── 11_roc_curves.html
│   ├── 12_confusion_matrix.png
│   ├── 13_transformer_confidence.html
│   ├── 14_topic_distribution.html
│   ├── 15_topic_sentiment.html
│   ├── FINAL_REPORT.md
│   ├── sample_processed_data.csv
│   └── model_comparison.csv
│
├── notebooks/                       # Jupyter notebooks (optional)
└── src/                            # Source code modules (optional)
```

## 📊 Sample Results

### Model Performance
| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8950   | 0.8920    | 0.8980 | 0.8950   | 0.9580  |
| Naive Bayes         | 0.8560   | 0.8640    | 0.8450 | 0.8540   | 0.9320  |
| Random Forest       | 0.8590   | 0.8720    | 0.8420 | 0.8570   | 0.9380  |
| Gradient Boosting   | 0.8780   | 0.8810    | 0.8740 | 0.8775   | 0.9510  |

### Key Insights
- 50,000 reviews analyzed (50% positive, 50% negative)
- Best traditional model: Logistic Regression (F1: 0.8950)
- Transformer accuracy: ~0.90+
- 5 distinct topics identified through LDA
- Strong correlation between polarity and sentiment (0.85+)
- Statistically significant word count differences between sentiments

## 🎓 What This Project Demonstrates

### Data Science Skills
- Large-scale data processing (50K+ samples)
- Feature engineering and selection
- Train-test splitting with stratification
- Cross-validation ready architecture
- Model comparison and evaluation

### NLP Expertise
- Text preprocessing pipelines
- Tokenization and lemmatization
- N-gram extraction
- TF-IDF feature engineering
- Topic modeling with LDA
- Sentiment polarity analysis

### Machine Learning
- Multi-model training and comparison
- Hyperparameter consideration
- Performance metrics analysis
- ROC curve analysis
- Confusion matrix interpretation

### Deep Learning & AI
- Hugging Face integration
- Pre-trained transformer models
- Transfer learning application
- Batch inference optimization
- Zero-shot classification

### Statistical Analysis
- Hypothesis testing (t-tests, chi-square)
- Correlation analysis
- Distribution analysis
- Statistical significance interpretation

### Data Visualization
- Interactive Plotly visualizations
- Static matplotlib/seaborn charts
- Word clouds
- Multi-subplot layouts
- Professional styling

### Software Engineering
- Clean, modular code
- Comprehensive documentation
- Error handling
- Reproducibility (random seeds)
- Production-ready structure

## 📈 Output Files

### Interactive HTML Visualizations (6)
- Sentiment distributions
- Word count analysis
- Polarity vs subjectivity
- Model comparisons
- ROC curves
- Transformer confidence
- Topic analysis

### Static PNG Images (6)
- Statistical features
- Correlation heatmap
- Word clouds
- Bigram analysis
- Trigram analysis
- Confusion matrix

### Reports & Data (3)
- Comprehensive markdown report
- Sample processed data CSV
- Model comparison CSV

## 🔧 Configuration

### Model Parameters
```python
# TF-IDF Settings
max_features = 5000
min_df = 5
max_df = 0.8
ngram_range = (1, 2)

# LDA Settings
n_topics = 5
max_iter = 20

# Train-Test Split
test_size = 0.2
random_state = 42
```

### Transformer Model
```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sample_size = 1000  # For transformer analysis
batch_size = 32
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: [IMDB Movie Reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) from Hugging Face
- **Transformer Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face
- **Libraries**: scikit-learn, Hugging Face Transformers, NLTK, Plotly, and all amazing open-source contributors

## 📧 Contact

For questions or feedback, please open an issue in the repository.

## 🌟 Star This Repository

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ for the data science community**
