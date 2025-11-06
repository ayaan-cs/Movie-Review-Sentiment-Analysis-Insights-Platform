"""
IMDB Movie Review Sentiment Analysis - Streamlit Dashboard
==========================================================
Interactive web dashboard for sentiment analysis predictions and results visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import sys

# Check Python interpreter
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
current_python = sys.executable

# Try importing textblob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError as e:
    TEXTBLOB_AVAILABLE = False
    st.error(f"""
    **TextBlob not found!** 
    
    You're using Python: `{current_python}`
    Expected venv Python: `{venv_path}`
    
    **Solution:** Run from terminal with venv activated:
    
    ```bash
    cd /Users/yaani/IdeaProjects/Movie-Review-Sentiment-Analysis-Insights-Platform
    source venv/bin/activate
    streamlit run app.py
    ```
    
    Or install in current environment:
    ```bash
    pip install textblob
    ```
    
    Error: {e}
    """)
    st.stop()

# Try importing nltk
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError as e:
    NLTK_AVAILABLE = False
    st.error(f"**NLTK not found!** Please install: `pip install nltk`\nError: {e}")
    st.stop()

# Try importing plotly, with helpful error message if it fails
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    PLOTLY_AVAILABLE = False
    st.error(f"""
    **Plotly not found!** Please install it:
    
    ```bash
    pip install plotly
    ```
    
    Or if using a virtual environment:
    ```bash
    source venv/bin/activate
    pip install plotly
    ```
    
    Error: {e}
    """)
    st.stop()

# Try importing transformers, with fallback
try:
    from transformers import pipeline
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing functions (same as main script)
def clean_text(text):
    """Advanced text preprocessing"""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def advanced_preprocess(text):
    """Tokenize, remove stopwords, and lemmatize - optimized for short texts"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Expand list of sentiment-bearing words to preserve
    sentiment_words = {
        'good', 'bad', 'great', 'terrible', 'excellent', 'awful', 'amazing', 'horrible',
        'fantastic', 'poor', 'wonderful', 'worst', 'best', 'love', 'hate', 'like', 'dislike',
        'awesome', 'awful', 'brilliant', 'boring', 'interesting', 'funny', 'sad', 'happy',
        'enjoyed', 'disappointed', 'recommend', 'not', 'no', 'nor', 'but', 'very', 'really',
        'quite', 'extremely', 'absolutely'
    }
    
    tokens = word_tokenize(text)
    # For short texts (less than 10 words), be more conservative with stopword removal
    is_short = len(tokens) < 10
    
    if is_short:
        # Keep all words for short texts, just lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 1]
    else:
        # For longer texts, remove stopwords but keep sentiment words
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                  if (word not in stop_words or word in sentiment_words) and len(word) > 2]
    
    return ' '.join(tokens)

@st.cache_resource
def load_ml_model():
    """Load the trained ML model and vectorizer"""
    try:
        model = joblib.load('models/best_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_transformer_model():
    """Load the transformer model"""
    if not TRANSFORMER_AVAILABLE:
        return None
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading transformer model: {e}")
        return None

@st.cache_data
def load_analysis_results():
    """Load saved analysis results"""
    results = {}
    try:
        # Load model comparison
        if os.path.exists('analysis_results/model_comparison.csv'):
            results['model_comparison'] = pd.read_csv('analysis_results/model_comparison.csv', index_col=0)
        
        # Load sample data
        if os.path.exists('analysis_results/sample_processed_data.csv'):
            results['sample_data'] = pd.read_csv('analysis_results/sample_processed_data.csv')
        
        return results
    except Exception as e:
        st.warning(f"Could not load some results: {e}")
        return results

def predict_sentiment_ml(text, model, vectorizer):
    """Predict sentiment using ML model"""
    if model is None or vectorizer is None:
        return None, None
    
    # Preprocess text
    cleaned = clean_text(text)
    processed = advanced_preprocess(cleaned)
    
    # If preprocessing resulted in empty or very short text, use cleaned version
    if len(processed.strip()) == 0 or len(processed.split()) < 1:
        processed = cleaned
    
    # Vectorize
    text_vectorized = vectorizer.transform([processed])
    
    # Check if vectorization resulted in empty vector (all zeros)
    if text_vectorized.sum() == 0:
        # If empty, use original text with minimal preprocessing (just lowercase, no special chars)
        processed = cleaned
        text_vectorized = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # FIX: Model labels are inverted (0=positive, 1=negative in trained model)
    # So we need to invert the mapping
    sentiment = "Positive" if prediction == 0 else "Negative"
    # Use the probability for the predicted class
    confidence = probability[prediction] * 100
    
    return sentiment, confidence

def predict_sentiment_transformer(text, transformer_model):
    """Predict sentiment using transformer model"""
    if transformer_model is None:
        return None, None
    
    # Truncate to max length
    text = text[:512]
    
    # Predict
    result = transformer_model(text)[0]
    
    label = result['label']
    score = result['score'] * 100
    
    # Handle potential label inversion
    if label == 'POSITIVE':
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    
    return sentiment, score

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🎬 IMDB Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["📊 Predict Sentiment", "📈 Analysis Results", "📋 About"]
    )
    
    # Load models
    ml_model, vectorizer = load_ml_model()
    transformer_model = load_transformer_model()
    
    if page == "📊 Predict Sentiment":
        st.header("Real-time Sentiment Prediction")
        
        # Info box about the models
        with st.expander("ℹ️ How It Works", expanded=False):
            st.markdown("""
            **This tool uses two different AI models to analyze sentiment:**
            
            - **🤖 Traditional ML Model (Logistic Regression)**: Trained on 40,000 IMDB movie reviews using TF-IDF features. 
              Best for longer, detailed reviews (50+ words). Accuracy: 88.45%
            
            - **🤗 Transformer Model (DistilBERT)**: A pre-trained deep learning model from Hugging Face, fine-tuned for sentiment analysis.
              Works well for both short and long texts. Accuracy: ~84%
            
            Both models will analyze your text and provide a sentiment prediction (Positive or Negative) with confidence scores.
            """)
        
        st.markdown("Enter a movie review below to analyze its sentiment using our trained models.")
        
        # Text input
        review_text = st.text_area(
            "Enter your movie review:",
            height=200,
            placeholder="Example: This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout...",
            help="Type or paste a movie review. The longer and more detailed the review, the more accurate the predictions will be."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if predict_button and review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                # ML Model Prediction
                if ml_model is not None and vectorizer is not None:
                    ml_sentiment, ml_confidence = predict_sentiment_ml(review_text, ml_model, vectorizer)
                else:
                    ml_sentiment, ml_confidence = None, None
                
                # Transformer Model Prediction
                if transformer_model is not None:
                    trans_sentiment, trans_confidence = predict_sentiment_transformer(review_text, transformer_model)
                else:
                    trans_sentiment, trans_confidence = None, None
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                cols = st.columns(2)
                
                # ML Model Results
                with cols[0]:
                    st.markdown("### 🤖 Traditional ML Model")
                    st.caption("Logistic Regression • Trained on 40K IMDB reviews")
                    if ml_sentiment:
                        sentiment_emoji = "😊" if ml_sentiment == "Positive" else "😞"
                        st.markdown(f"**Sentiment:** {sentiment_emoji} {ml_sentiment}")
                        st.progress(ml_confidence / 100)
                        st.metric("Confidence", f"{ml_confidence:.2f}%")
                        
                        # Confidence interpretation
                        if ml_confidence >= 90:
                            st.success("Very confident prediction")
                        elif ml_confidence >= 75:
                            st.info("Confident prediction")
                        else:
                            st.warning("Lower confidence - review may be ambiguous")
                        
                        # Warning for short texts
                        if len(review_text.split()) < 5:
                            st.info("ℹ️ Short texts may have lower accuracy. The model was trained on longer movie reviews (avg 232 words).")
                    else:
                        st.warning("ML model not available. Please run `python imdb_sentiment_analysis.py` first.")
                
                # Transformer Model Results
                with cols[1]:
                    st.markdown("### 🤗 Transformer Model (DistilBERT)")
                    st.caption("Pre-trained Hugging Face model • Works well for all text lengths")
                    if trans_sentiment:
                        sentiment_emoji = "😊" if trans_sentiment == "Positive" else "😞"
                        st.markdown(f"**Sentiment:** {sentiment_emoji} {trans_sentiment}")
                        st.progress(trans_confidence / 100)
                        st.metric("Confidence", f"{trans_confidence:.2f}%")
                        
                        # Confidence interpretation
                        if trans_confidence >= 90:
                            st.success("Very confident prediction")
                        elif trans_confidence >= 75:
                            st.info("Confident prediction")
                        else:
                            st.warning("Lower confidence - review may be ambiguous")
                    else:
                        st.warning("Transformer model not available.")
                
                # Agreement
                if ml_sentiment and trans_sentiment:
                    if ml_sentiment == trans_sentiment:
                        st.success("✅ Both models agree!")
                    else:
                        st.warning("⚠️ Models disagree. Consider reviewing the text.")
                
                # Additional analysis
                st.markdown("---")
                st.subheader("Text Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    word_count = len(review_text.split())
                    st.metric("Word Count", word_count)
                
                with col2:
                    char_count = len(review_text)
                    st.metric("Character Count", char_count)
                
                with col3:
                    polarity = TextBlob(review_text).sentiment.polarity
                    st.metric("Polarity", f"{polarity:.3f}")
                
                with col4:
                    subjectivity = TextBlob(review_text).sentiment.subjectivity
                    st.metric("Subjectivity", f"{subjectivity:.3f}")
                
                # Polarity visualization
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number",
                        value = polarity,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment Polarity"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, 0], 'color': "lightgray"},
                                {'range': [0, 1], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Polarity: {polarity:.3f} (Plotly not available for visualization)")
        
        elif predict_button:
            st.error("Please enter a review to analyze.")
        
        # Example reviews - displayed in boxes underneath
        st.markdown("---")
        st.subheader("💡 Example Reviews")
        
        example_reviews = [
            ("Positive", "This movie was absolutely fantastic! The acting was superb, the storyline was engaging, and the cinematography was breathtaking. I would definitely watch it again and recommend it to all my friends."),
            ("Negative", "What a waste of time! The plot was confusing, the acting was terrible, and the special effects looked cheap. I couldn't wait for it to end. Do not recommend."),
            ("Positive", "An amazing film with great character development and an unexpected twist at the end. The director did an excellent job bringing this story to life."),
            ("Negative", "Boring and predictable. The dialogue was weak and the characters were one-dimensional. I expected so much more from this film.")
        ]
        
        # Display examples in two columns
        col1, col2 = st.columns(2)
        
        positive_examples = [(sent, rev) for sent, rev in example_reviews if sent == "Positive"]
        negative_examples = [(sent, rev) for sent, rev in example_reviews if sent == "Negative"]
        
        with col1:
            st.markdown("### 😊 Positive Examples")
            for idx, (sentiment, review) in enumerate(positive_examples, 1):
                st.markdown(f"""
                <div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>Positive Example {idx}</strong><br>
                    <p style="margin-top: 10px; color: #333;">{review}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 😞 Negative Examples")
            for idx, (sentiment, review) in enumerate(negative_examples, 1):
                st.markdown(f"""
                <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>Negative Example {idx}</strong><br>
                    <p style="margin-top: 10px; color: #333;">{review}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "📈 Analysis Results":
        st.header("Analysis Results Dashboard")
        st.markdown("View comprehensive results from the sentiment analysis project.")
        
        # Dataset information
        with st.expander("📊 Dataset Information", expanded=False):
            st.markdown("""
            **IMDB Movie Reviews Dataset**
            - **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews)
            - **Size**: 40,000 movie reviews
            - **Balance**: 50% positive, 50% negative (perfectly balanced)
            - **Average Review Length**: 232 words
            - **Review Length Range**: 4 to 2,470 words
            
            The dataset was used to train and evaluate multiple machine learning models for sentiment classification.
            """)
        
        results = load_analysis_results()
        
        # Model Comparison
        if 'model_comparison' in results:
            st.subheader("Model Performance Comparison")
            df = results['model_comparison']
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            best_model = df['F1 Score'].idxmax()
            
            with col1:
                st.metric("Best Model", best_model)
            with col2:
                st.metric("Best Accuracy", f"{df.loc[best_model, 'Accuracy']:.4f}")
            with col3:
                st.metric("Best F1 Score", f"{df.loc[best_model, 'F1 Score']:.4f}")
            with col4:
                st.metric("Best Precision", f"{df.loc[best_model, 'Precision']:.4f}")
            with col5:
                st.metric("Best ROC AUC", f"{df.loc[best_model, 'ROC AUC']:.4f}")
            
            # Visualization
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=df.index,
                        y=df[metric],
                        text=df[metric].round(4),
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly not available. Showing data table only.")
            
            # Table
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.warning("Model comparison data not found. Please run the main analysis script first.")
        
        # Sample Data
        if 'sample_data' in results:
            st.subheader("Sample Processed Data")
            st.caption("A sample of 100 reviews from the processed dataset showing original text, sentiment labels, and extracted features.")
            with st.expander("View Sample Reviews", expanded=False):
                sample_df = results['sample_data'].head(100)
                st.dataframe(sample_df, use_container_width=True)
                
                # Show statistics
                if 'sentiment' in sample_df.columns:
                    st.markdown("**Sample Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(sample_df))
                    with col2:
                        positive_count = (sample_df['sentiment'] == 'positive').sum() if 'sentiment' in sample_df.columns else 0
                        st.metric("Positive", positive_count)
                    with col3:
                        negative_count = (sample_df['sentiment'] == 'negative').sum() if 'sentiment' in sample_df.columns else 0
                        st.metric("Negative", negative_count)
        
        # Visualizations
        st.subheader("Saved Visualizations")
        st.caption("Interactive HTML visualizations generated from the analysis. Click to open in your browser.")
        
        viz_files = [
            ("01_sentiment_distribution.html", "Sentiment Distribution", "Shows the balance between positive and negative reviews"),
            ("02_word_count_by_sentiment.html", "Word Count Analysis", "Compares review lengths by sentiment"),
            ("03_polarity_vs_subjectivity.html", "Polarity vs Subjectivity", "Scatter plot of sentiment scores"),
            ("10_model_comparison.html", "Model Performance Comparison", "Interactive comparison of all ML models"),
            ("11_roc_curves.html", "ROC Curves", "Receiver Operating Characteristic curves for model evaluation"),
            ("14_topic_distribution.html", "Topic Distribution", "LDA topic modeling results"),
        ]
        
        cols = st.columns(2)
        for idx, (viz_file, title, description) in enumerate(viz_files):
            col = cols[idx % 2]
            with col:
                if os.path.exists(f'analysis_results/{viz_file}'):
                    st.markdown(f"""
                    **{title}**  
                    <small>{description}</small>  
                    📄 `{viz_file}`
                    """, unsafe_allow_html=True)
                    st.markdown("---")
    
    elif page == "📋 About":
        st.header("About This Project")
        
        # Project Overview
        st.markdown("""
        ## 🎬 IMDB Movie Review Sentiment Analysis Platform
        
        A comprehensive sentiment analysis project that analyzes **40,000 IMDB movie reviews** using 
        both traditional machine learning and modern transformer models.
        """)
        
        # Dataset Information
        st.subheader("📊 Dataset")
        st.markdown("""
        - **Source**: [IMDB Movie Reviews Dataset](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) from Hugging Face
        - **Size**: 40,000 reviews (perfectly balanced: 20,000 positive, 20,000 negative)
        - **Average Length**: 232 words per review
        - **Range**: 4 to 2,470 words
        - **License**: Public dataset for research and educational purposes
        """)
        
        # Technologies
        st.subheader("🛠️ Technologies Used")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Machine Learning:**
            - Logistic Regression
            - Naive Bayes
            - Random Forest
            - Gradient Boosting
            
            **NLP Libraries:**
            - NLTK (tokenization, lemmatization)
            - TextBlob (sentiment polarity)
            - TF-IDF Vectorization
            """)
        with col2:
            st.markdown("""
            **Deep Learning:**
            - Hugging Face Transformers
            - DistilBERT (pre-trained model)
            
            **Visualization:**
            - Plotly (interactive charts)
            - Matplotlib & Seaborn
            - WordCloud
            """)
        
        # Model Performance
        st.subheader("📈 Model Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Best Accuracy", "88.45%", "Logistic Regression")
        with perf_col2:
            st.metric("F1 Score", "88.35%", "Best performing metric")
        with perf_col3:
            st.metric("ROC AUC", "95.52%", "Excellent discrimination")
        
        st.markdown("""
        | Model | Accuracy | F1 Score | ROC AUC |
        |-------|----------|----------|---------|
        | Logistic Regression | 88.45% | 88.35% | 95.52% |
        | Naive Bayes | 85.21% | 85.02% | 93.14% |
        | Random Forest | 84.31% | 84.20% | 92.40% |
        | Gradient Boosting | 80.69% | 79.49% | 89.30% |
        | DistilBERT (Transformer) | 83.60% | 83.50% | ~90% |
        """)
        
        # Key Features
        st.subheader("✨ Key Features")
        st.markdown("""
        1. **Real-time Predictions**: Analyze sentiment of any movie review instantly
        2. **Dual Model Comparison**: Compare traditional ML vs transformer predictions
        3. **Comprehensive Analysis**: 15+ interactive visualizations
        4. **Topic Modeling**: Discover 5 distinct themes in movie reviews using LDA
        5. **Statistical Testing**: Hypothesis tests and correlation analysis
        6. **Professional Visualizations**: Interactive HTML charts and static images
        """)
        
        # Getting Started
        st.subheader("🚀 Getting Started")
        st.markdown("""
        **For Users:**
        1. Go to **"Predict Sentiment"** to analyze your own movie reviews
        2. View **"Analysis Results"** to see model performance and statistics
        3. Check out the example reviews to see how it works
        
        **For Developers:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Run analysis: `python imdb_sentiment_analysis.py` (generates models and results)
        3. Launch app: `streamlit run app.py`
        4. Deploy: Follow `STREAMLIT_DEPLOYMENT.md` for deployment instructions
        """)
        
        # Project Structure
        st.subheader("📁 Project Structure")
        st.code("""
Movie-Review-Sentiment-Analysis-Insights-Platform/
├── imdb_sentiment_analysis.py  # Main analysis script
├── app.py                       # Streamlit dashboard (this app)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── QUICK_START.md              # Quick start guide
├── STREAMLIT_DEPLOYMENT.md     # Deployment guide
├── analysis_results/           # Generated outputs (after running script)
│   ├── *.html                  # Interactive visualizations
│   ├── *.png                   # Static images
│   └── FINAL_REPORT.md         # Comprehensive report
└── models/                     # Trained models (after running script)
    ├── best_model.pkl
    └── tfidf_vectorizer.pkl
        """, language="text")
        
        # Deployment
        st.subheader("🌐 Deployment")
        st.markdown("""
        This app can be deployed to:
        - **Streamlit Cloud** (recommended, free): [streamlit.io](https://streamlit.io)
        - **Heroku**: Follow deployment guide
        - **Railway**: Easy Python app deployment
        - **Any cloud platform** that supports Python
        
        See `STREAMLIT_DEPLOYMENT.md` for detailed instructions.
        """)
        
        # Credits
        st.subheader("🙏 Credits & Resources")
        st.markdown("""
        - **Dataset**: [IMDB Movie Reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) by ajaykarthick
        - **Transformer Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face
        - **Libraries**: scikit-learn, transformers, NLTK, Plotly, and the open-source community
        
        ---
        
        **Built with ❤️ for data science and NLP enthusiasts**
        
        *This project demonstrates enterprise-level data science capabilities including large-scale data processing, 
        advanced NLP, machine learning, deep learning, and professional visualization.*
        """)
        
        # Show file structure
        with st.expander("📁 Project Files"):
            if os.path.exists('analysis_results'):
                files = os.listdir('analysis_results')
                st.write("**Analysis Results:**")
                for f in sorted(files)[:10]:
                    st.write(f"- {f}")
                if len(files) > 10:
                    st.write(f"... and {len(files) - 10} more files")
            
            if os.path.exists('models'):
                models = os.listdir('models')
                st.write("**Saved Models:**")
                for m in models:
                    st.write(f"- {m}")

if __name__ == "__main__":
    main()

