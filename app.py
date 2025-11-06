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

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'quick_mode' not in st.session_state:
    st.session_state.quick_mode = False
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.0

# Enhanced Custom CSS with dark mode support
def get_css(dark_mode=False):
    """Generate CSS based on theme"""
    if dark_mode:
        return """
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #4ECDC4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #1e1e1e;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                border: 1px solid #333;
            }
            .stButton>button {
                width: 100%;
                background-color: #4ECDC4;
            }
            .review-box {
                background-color: #2d2d2d;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #4ECDC4;
            }
        </style>
        """
    else:
        return """
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
            .review-box {
                background-color: #ffffff;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #4ECDC4;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .positive-box {
                border-left-color: #4ECDC4;
                background: linear-gradient(90deg, #e8f5e9 0%, #ffffff 100%);
            }
            .negative-box {
                border-left-color: #FF6B6B;
                background: linear-gradient(90deg, #ffebee 0%, #ffffff 100%);
            }
        </style>
        """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

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

def add_to_history(text, ml_sentiment, ml_confidence, trans_sentiment, trans_confidence):
    """Add prediction to session history"""
    history_entry = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text': text[:100] + '...' if len(text) > 100 else text,
        'full_text': text,
        'ml_sentiment': ml_sentiment,
        'ml_confidence': ml_confidence,
        'trans_sentiment': trans_sentiment,
        'trans_confidence': trans_confidence,
        'agreement': ml_sentiment == trans_sentiment if ml_sentiment and trans_sentiment else None
    }
    st.session_state.prediction_history.insert(0, history_entry)
    # Keep only last 10 predictions
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[:10]

def create_sentiment_gauge(polarity, confidence):
    """Create a speedometer-style gauge for sentiment"""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Map polarity to 0-100 scale for gauge
    gauge_value = ((polarity + 1) / 2) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Gauge", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#4ECDC4" if polarity > 0 else "#FF6B6B"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': "#FF6B6B"},
                {'range': [33, 67], 'color': "#FFE66D"},
                {'range': [67, 100], 'color': "#4ECDC4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def get_educational_tooltip(term):
    """Return educational tooltip text for various terms"""
    tooltips = {
        'polarity': "Polarity ranges from -1 (very negative) to +1 (very positive). It measures the emotional tone of the text.",
        'subjectivity': "Subjectivity ranges from 0 (objective/factual) to 1 (subjective/opinionated). Higher values indicate more personal opinions.",
        'confidence': "Confidence score (0-100%) indicates how certain the model is about its prediction. Higher confidence = more reliable prediction.",
        'f1_score': "F1 Score balances precision and recall. It's the harmonic mean, giving equal weight to both metrics. Higher is better (max 1.0).",
        'roc_auc': "ROC AUC (Area Under Curve) measures the model's ability to distinguish between classes. 0.5 = random, 1.0 = perfect."
    }
    return tooltips.get(term.lower(), f"Information about {term}")

def get_did_you_know_fact():
    """Return a random 'Did You Know?' fact about sentiment analysis"""
    facts = [
        "Sentiment analysis is used by 80% of Fortune 500 companies to understand customer feedback!",
        "The first sentiment analysis algorithm was developed in the 1950s, but modern AI has made it 10x more accurate.",
        "Movie reviews are one of the most common datasets for training sentiment models because they're clearly labeled.",
        "Transformer models like BERT can understand context - they know 'not good' is negative, not positive!",
        "The average person writes about 100 words per review, but models work best with 50-500 words.",
        "Sentiment analysis helps companies respond to negative reviews 3x faster, improving customer satisfaction.",
        "Did you know? The word 'good' appears in both positive AND negative reviews - context matters!"
    ]
    import random
    return random.choice(facts)

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🎬 IMDB Movie Review Sentiment Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar with settings and history
    st.sidebar.title("Navigation")
    
    # Settings toggle
    with st.sidebar.expander("Settings", expanded=False):
        st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        st.session_state.quick_mode = st.checkbox("Quick Mode", value=st.session_state.quick_mode, 
                                                   help="Skip detailed visualizations for faster predictions")
        st.session_state.confidence_threshold = st.slider("Confidence Threshold", 0.0, 100.0, 
                                                          st.session_state.confidence_threshold, 5.0,
                                                          help="Only show predictions above this confidence level")
        if st.button("Reset Settings"):
            st.session_state.dark_mode = False
            st.session_state.quick_mode = False
            st.session_state.confidence_threshold = 0.0
            st.rerun()
    
    # Update CSS based on dark mode
    st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Prediction History Sidebar
    if st.session_state.prediction_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Recent Predictions")
        for idx, pred in enumerate(st.session_state.prediction_history[:5]):
            with st.sidebar.expander(f"{pred['timestamp']} - {pred['ml_sentiment'] or 'N/A'}", expanded=False):
                st.write(f"**Text:** {pred['text']}")
                if pred['ml_sentiment']:
                    st.write(f"**ML:** {pred['ml_sentiment']} ({pred['ml_confidence']:.1f}%)")
                if pred['trans_sentiment']:
                    st.write(f"**Transformer:** {pred['trans_sentiment']} ({pred['trans_confidence']:.1f}%)")
                if st.button(f"View Details", key=f"view_{idx}"):
                    st.session_state.selected_history = pred
                    st.rerun()
        
        if st.sidebar.button("Clear History"):
            st.session_state.prediction_history = []
            st.sidebar.success("History cleared!")
            st.rerun()
    
    # Navigation
    page = st.sidebar.radio(
        "Choose a page",
        ["Predict Sentiment", "Batch Predict", "Analysis Results", "Tutorial", "Settings", "About"]
    )
    
    # Load models
    with st.spinner("Loading models..."):
        ml_model, vectorizer = load_ml_model()
        transformer_model = load_transformer_model()
    
    if page == "Predict Sentiment":
        st.header("Real-time Sentiment Prediction")
        
        # Educational fact
        with st.expander("Did You Know?", expanded=False):
            st.info(get_did_you_know_fact())
        
        # Info box about the models
        with st.expander("How It Works", expanded=False):
            st.markdown("""
            **This tool uses two different AI models to analyze sentiment:**
            
            - **Traditional ML Model (Logistic Regression)**: Trained on 40,000 IMDB movie reviews using TF-IDF features. 
              Best for longer, detailed reviews (50+ words). Accuracy: 88.45%
            
            - **Transformer Model (DistilBERT)**: A pre-trained deep learning model from Hugging Face, fine-tuned for sentiment analysis.
              Works well for both short and long texts. Accuracy: ~84%
            
            Both models will analyze your text and provide a sentiment prediction (Positive or Negative) with confidence scores.
            """)
        
        st.markdown("Enter a movie review below to analyze its sentiment using our trained models.")
        
        # Text templates dropdown
        text_templates = {
            "Select a template...": "",
            "Positive Review Template": "This movie was absolutely fantastic! The acting was superb, the storyline was engaging, and the cinematography was breathtaking. I would definitely watch it again and recommend it to all my friends.",
            "Negative Review Template": "What a waste of time! The plot was confusing, the acting was terrible, and the special effects looked cheap. I couldn't wait for it to end. Do not recommend.",
            "Neutral/Mixed Review": "The movie had its moments. Some scenes were well done, but others fell flat. Overall, it's okay but nothing special."
        }
        
        selected_template = st.selectbox("Quick Templates", list(text_templates.keys()))
        template_text = text_templates.get(selected_template, "")
        
        # Text input with character counter
        review_text = st.text_area(
            "Enter your movie review:",
            value=template_text if template_text else "",
            height=200,
            placeholder="Example: This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout...",
            help="Type or paste a movie review. The longer and more detailed the review, the more accurate the predictions will be.",
            key="review_input_main"
        )
        
        # Character and word counter with warnings
        col_info1, col_info2, col_info3 = st.columns([2, 2, 1])
        with col_info1:
            char_count = len(review_text)
            word_count = len(review_text.split()) if review_text else 0
            st.caption(f"Characters: {char_count} | Words: {word_count}")
        
        with col_info2:
            if word_count > 0:
                if word_count < 10:
                    st.warning("Text is very short (<10 words). Accuracy may be lower.")
                elif word_count > 500:
                    st.warning("Text is very long (>500 words). Processing may take longer.")
                elif word_count >= 10 and word_count <= 500:
                    st.success("Text length is optimal for analysis.")
        
        with col_info3:
            if st.button("Clear", use_container_width=True):
                st.session_state.review_input_main = ""
                st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            predict_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        with col2:
            comparison_mode = st.checkbox("Comparison Mode", help="Show detailed side-by-side model analysis")
        with col3:
            show_preprocessing = st.checkbox("Show Preprocessing", help="Preview how text is processed before prediction")
        
        # Show preprocessing preview if enabled
        if show_preprocessing and review_text.strip():
            with st.expander("Text Preprocessing Preview", expanded=True):
                cleaned = clean_text(review_text)
                processed = advanced_preprocess(cleaned)
                col_pre1, col_pre2, col_pre3 = st.columns(3)
                with col_pre1:
                    st.markdown("**Original Text:**")
                    st.text(review_text[:200] + "..." if len(review_text) > 200 else review_text)
                with col_pre2:
                    st.markdown("**After Cleaning:**")
                    st.text(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
                with col_pre3:
                    st.markdown("**After Preprocessing:**")
                    st.text(processed[:200] + "..." if len(processed) > 200 else processed)
        
        if predict_button and review_text.strip():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading models...")
            progress_bar.progress(10)
            
            # ML Model Prediction
            status_text.text("Running ML model prediction...")
            progress_bar.progress(30)
            if ml_model is not None and vectorizer is not None:
                ml_sentiment, ml_confidence = predict_sentiment_ml(review_text, ml_model, vectorizer)
            else:
                ml_sentiment, ml_confidence = None, None
            
            # Transformer Model Prediction
            status_text.text("Running transformer model prediction...")
            progress_bar.progress(60)
            if transformer_model is not None:
                trans_sentiment, trans_confidence = predict_sentiment_transformer(review_text, transformer_model)
            else:
                trans_sentiment, trans_confidence = None, None
            
            # Check confidence threshold
            if st.session_state.confidence_threshold > 0:
                if ml_confidence and ml_confidence < st.session_state.confidence_threshold:
                    ml_sentiment, ml_confidence = None, None
                if trans_confidence and trans_confidence < st.session_state.confidence_threshold:
                    trans_sentiment, trans_confidence = None, None
            
            status_text.text("Generating visualizations...")
            progress_bar.progress(80)
            
            # Add to history
            add_to_history(review_text, ml_sentiment, ml_confidence, trans_sentiment, trans_confidence)
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            cols = st.columns(2)
                
            # ML Model Results
            with cols[0]:
                st.markdown("### Traditional ML Model")
                st.caption("Logistic Regression • Trained on 40K IMDB reviews")
                if ml_sentiment:
                    st.markdown(f"**Sentiment:** {ml_sentiment}")
                    st.progress(ml_confidence / 100)
                    st.metric("Confidence", f"{ml_confidence:.2f}%", 
                             help=get_educational_tooltip('confidence'))
                    
                    # Confidence interpretation
                    if ml_confidence >= 90:
                        st.success("Very confident prediction")
                    elif ml_confidence >= 75:
                        st.info("Confident prediction")
                    else:
                        st.warning("Lower confidence - review may be ambiguous")
                    
                    # Warning for short texts
                    if len(review_text.split()) < 5:
                        st.info("Short texts may have lower accuracy. The model was trained on longer movie reviews (avg 232 words).")
                else:
                    st.warning("ML model not available. Please run `python imdb_sentiment_analysis.py` first.")
            
            # Transformer Model Results
            with cols[1]:
                st.markdown("### Transformer Model (DistilBERT)")
                st.caption("Pre-trained Hugging Face model • Works well for all text lengths")
                if trans_sentiment:
                    st.markdown(f"**Sentiment:** {trans_sentiment}")
                    st.progress(trans_confidence / 100)
                    st.metric("Confidence", f"{trans_confidence:.2f}%",
                             help=get_educational_tooltip('confidence'))
                    
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
                    st.success("Both models agree!")
                else:
                    st.warning("Models disagree. Consider reviewing the text.")
            
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
                st.metric("Polarity", f"{polarity:.3f}", 
                         help=get_educational_tooltip('polarity'))
            
            with col4:
                subjectivity = TextBlob(review_text).sentiment.subjectivity
                st.metric("Subjectivity", f"{subjectivity:.3f}",
                         help=get_educational_tooltip('subjectivity'))
            
            # Enhanced visualizations
            if not st.session_state.quick_mode:
                if PLOTLY_AVAILABLE:
                    # Sentiment Gauge (speedometer style)
                    polarity = TextBlob(review_text).sentiment.polarity
                    avg_confidence = (ml_confidence + trans_confidence) / 2 if ml_confidence and trans_confidence else (ml_confidence or trans_confidence or 0)
                    gauge_fig = create_sentiment_gauge(polarity, avg_confidence)
                    if gauge_fig:
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Model disagreement visualization
                    if ml_sentiment and trans_sentiment and ml_sentiment != trans_sentiment:
                        st.markdown("#### Model Disagreement Analysis")
                        disagree_fig = go.Figure()
                        disagree_fig.add_trace(go.Bar(
                            name='ML Model',
                            x=['Confidence'],
                            y=[ml_confidence],
                            marker_color='#4ECDC4',
                            text=f"{ml_sentiment} ({ml_confidence:.1f}%)",
                            textposition='auto'
                        ))
                        disagree_fig.add_trace(go.Bar(
                            name='Transformer',
                            x=['Confidence'],
                            y=[trans_confidence],
                            marker_color='#FF6B6B',
                            text=f"{trans_sentiment} ({trans_confidence:.1f}%)",
                            textposition='auto'
                        ))
                        disagree_fig.update_layout(
                            title='Confidence Comparison (Models Disagree)',
                            yaxis_title='Confidence (%)',
                            height=300,
                            barmode='group'
                        )
                        st.plotly_chart(disagree_fig, use_container_width=True)
                else:
                    st.info(f"Polarity: {polarity:.3f} (Plotly not available for visualization)")
            
            # Comparison mode details
            if comparison_mode and ml_sentiment and trans_sentiment:
                st.markdown("---")
                st.subheader("Detailed Model Comparison")
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("### ML Model Details")
                    st.write(f"**Model Type:** Logistic Regression")
                    st.write(f"**Training Data:** 40,000 IMDB reviews")
                    st.write(f"**Best For:** Longer, detailed reviews (50+ words)")
                    st.write(f"**Prediction:** {ml_sentiment} with {ml_confidence:.2f}% confidence")
                    st.write(f"**Strengths:** Fast, interpretable, excellent on longer texts")
                
                with comp_col2:
                    st.markdown("### Transformer Model Details")
                    st.write(f"**Model Type:** DistilBERT (Hugging Face)")
                    st.write(f"**Training Data:** Pre-trained on large corpus, fine-tuned for sentiment")
                    st.write(f"**Best For:** All text lengths, understands context")
                    st.write(f"**Prediction:** {trans_sentiment} with {trans_confidence:.2f}% confidence")
                    st.write(f"**Strengths:** Context-aware, works well on short texts")
                
                # Agreement statistics
                if len(st.session_state.prediction_history) > 1:
                    recent = st.session_state.prediction_history[:5]
                    agreements = sum(1 for p in recent if p.get('agreement') == True)
                    st.info(f"Recent Agreement Rate: {agreements}/{len(recent)} predictions ({agreements/len(recent)*100:.1f}%)")
        
        elif predict_button:
            st.error("Please enter a review to analyze.")
        
        # Example reviews - displayed in boxes underneath
        st.markdown("---")
        st.subheader("Example Reviews")
        
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
            st.markdown("### Positive Examples")
            for idx, (sentiment, review) in enumerate(positive_examples, 1):
                st.markdown(f"""
                <div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>Positive Example {idx}</strong><br>
                    <p style="margin-top: 10px; color: #333;">{review}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Negative Examples")
            for idx, (sentiment, review) in enumerate(negative_examples, 1):
                st.markdown(f"""
                <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>Negative Example {idx}</strong><br>
                    <p style="margin-top: 10px; color: #333;">{review}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "Batch Predict":
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple reviews to analyze them all at once.")
        
        # CSV upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should have a column named 'review' or 'text' containing the movie reviews"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"File loaded successfully! Found {len(df)} rows.")
                
                # Find review column
                review_col = None
                for col in df.columns:
                    if col.lower() in ['review', 'text', 'reviews', 'comment', 'comments']:
                        review_col = col
                        break
                
                if review_col is None:
                    st.error("Could not find a 'review' or 'text' column in the CSV. Please ensure your CSV has a column with reviews.")
                    st.dataframe(df.head())
                else:
                    st.info(f"Using column: **{review_col}**")
                    
                    # Show preview
                    with st.expander("Preview Data", expanded=False):
                        st.dataframe(df.head(10))
                    
                    # Process button
                    if st.button("Process All Reviews", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        total = len(df)
                        
                        for idx, row in df.iterrows():
                            review = str(row[review_col])
                            status_text.text(f"Processing review {idx + 1} of {total}...")
                            progress_bar.progress((idx + 1) / total)
                            
                            # Predict
                            ml_sent, ml_conf = predict_sentiment_ml(review, ml_model, vectorizer) if ml_model else (None, None)
                            trans_sent, trans_conf = predict_sentiment_transformer(review, transformer_model) if transformer_model else (None, None)
                            
                            # Calculate agreement
                            agreement = "Yes" if (ml_sent and trans_sent and ml_sent == trans_sent) else "No"
                            
                            results.append({
                                'Review': review[:100] + '...' if len(review) > 100 else review,
                                'Full_Review': review,
                                'ML_Sentiment': ml_sent or "N/A",
                                'ML_Confidence': f"{ml_conf:.2f}%" if ml_conf else "N/A",
                                'Transformer_Sentiment': trans_sent or "N/A",
                                'Transformer_Confidence': f"{trans_conf:.2f}%" if trans_conf else "N/A",
                                'Agreement': agreement,
                                'Word_Count': len(review.split()),
                                'Polarity': f"{TextBlob(review).sentiment.polarity:.3f}" if TEXTBLOB_AVAILABLE else "N/A"
                            })
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"Processed {total} reviews!")
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Statistics
                        st.markdown("---")
                        st.subheader("Batch Statistics")
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            ml_pos = (results_df['ML_Sentiment'] == 'Positive').sum()
                            ml_neg = (results_df['ML_Sentiment'] == 'Negative').sum()
                            st.metric("ML Positive", ml_pos)
                            st.metric("ML Negative", ml_neg)
                        
                        with stat_col2:
                            trans_pos = (results_df['Transformer_Sentiment'] == 'Positive').sum()
                            trans_neg = (results_df['Transformer_Sentiment'] == 'Negative').sum()
                            st.metric("Transformer Positive", trans_pos)
                            st.metric("Transformer Negative", trans_neg)
                        
                        with stat_col3:
                            agreements = (results_df['Agreement'] == 'Yes').sum()
                            disagreements = (results_df['Agreement'] == 'No').sum()
                            agreement_rate = (agreements / total * 100) if total > 0 else 0
                            st.metric("Agreements", agreements)
                            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
                        
                        with stat_col4:
                            # Average confidence
                            ml_confs = [float(r['ML_Confidence'].replace('%', '')) for r in results if r['ML_Confidence'] != 'N/A']
                            trans_confs = [float(r['Transformer_Confidence'].replace('%', '')) for r in results if r['Transformer_Confidence'] != 'N/A']
                            avg_ml = np.mean(ml_confs) if ml_confs else 0
                            avg_trans = np.mean(trans_confs) if trans_confs else 0
                            st.metric("Avg ML Confidence", f"{avg_ml:.1f}%")
                            st.metric("Avg Transformer Confidence", f"{avg_trans:.1f}%")
                        
                        # Visualization
                        if PLOTLY_AVAILABLE:
                            # Sentiment distribution
                            fig = make_subplots(rows=1, cols=2, subplot_titles=('ML Model Distribution', 'Transformer Distribution'))
                            
                            ml_counts = results_df['ML_Sentiment'].value_counts()
                            fig.add_trace(go.Bar(x=ml_counts.index, y=ml_counts.values, name='ML', marker_color='#4ECDC4'), row=1, col=1)
                            
                            trans_counts = results_df['Transformer_Sentiment'].value_counts()
                            fig.add_trace(go.Bar(x=trans_counts.index, y=trans_counts.values, name='Transformer', marker_color='#FF6B6B'), row=1, col=2)
                            
                            fig.update_layout(height=400, title_text="Sentiment Distribution", showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("---")
                        st.subheader("Results Table")
                        st.markdown("**Sortable and filterable results. Use the search box to filter.**")
                        
                        # Filter options
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            filter_sentiment = st.selectbox("Filter by ML Sentiment", ["All", "Positive", "Negative", "N/A"])
                        with filter_col2:
                            filter_agreement = st.selectbox("Filter by Agreement", ["All", "Yes", "No"])
                        
                        # Apply filters
                        filtered_df = results_df.copy()
                        if filter_sentiment != "All":
                            filtered_df = filtered_df[filtered_df['ML_Sentiment'] == filter_sentiment]
                        if filter_agreement != "All":
                            filtered_df = filtered_df[filtered_df['Agreement'] == filter_agreement]
                        
                        st.dataframe(filtered_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"sentiment_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.exception(e)
    
    elif page == "Analysis Results":
        st.header("Analysis Results Dashboard")
        st.markdown("View comprehensive results from the sentiment analysis project.")
        
        # Dataset information
        with st.expander("Dataset Information", expanded=False):
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
                    `{viz_file}`
                    """, unsafe_allow_html=True)
                    st.markdown("---")
    
    elif page == "Tutorial":
        st.header("Interactive Tutorial")
        st.markdown("Learn how to use this sentiment analysis platform step by step.")
        
        tutorial_step = st.radio(
            "Select a tutorial section:",
            ["Getting Started", "Single Prediction", "Batch Processing", "Understanding Results", "Advanced Features"],
            horizontal=True
        )
        
        if tutorial_step == "Getting Started":
            st.subheader("Welcome to the Sentiment Analysis Platform!")
            st.markdown("""
            This platform allows you to analyze the sentiment of movie reviews using two powerful AI models.
            
            **What you can do:**
            - Analyze individual reviews in real-time
            - Process multiple reviews at once (batch mode)
            - View comprehensive analysis results
            - Learn about sentiment analysis
            
            **Navigation:**
            - Use the sidebar to switch between pages
            - Settings are available in the sidebar expander
            - Your prediction history is saved during the session
            """)
            
            st.info("**Tip:** Start with a single prediction to see how it works!")
        
        elif tutorial_step == "Single Prediction":
            st.subheader("How to Make a Single Prediction")
            st.markdown("""
            **Step 1:** Go to the "Predict Sentiment" page
            
            **Step 2:** Enter your movie review in the text area
            - You can type directly
            - Or use the template dropdown for quick examples
            - Character and word count updates in real-time
            
            **Step 3:** Click "Analyze Sentiment"
            - Watch the progress bar
            - Results appear below
            
            **Step 4:** Review the results
            - See predictions from both models
            - Check confidence scores
            - View text analysis metrics
            - Explore visualizations (if not in Quick Mode)
            
            **Understanding the Results:**
            - **Sentiment:** Positive or Negative
            - **Confidence:** 0-100% (higher = more certain)
            - **Polarity:** -1 (negative) to +1 (positive)
            - **Subjectivity:** 0 (factual) to 1 (opinionated)
            """)
            
            st.success("Try it now! Go to Predict Sentiment and analyze a review.")
        
        elif tutorial_step == "Batch Processing":
            st.subheader("How to Process Multiple Reviews")
            st.markdown("""
            **Step 1:** Prepare your CSV file
            - Must have a column named 'review', 'text', 'reviews', 'comment', or 'comments'
            - Each row should contain one review
            
            **Step 2:** Go to "Batch Predict" page
            
            **Step 3:** Upload your CSV file
            - Click "Choose a CSV file"
            - Select your file
            - Preview will show automatically
            
            **Step 4:** Click "Process All Reviews"
            - Progress bar shows processing status
            - Estimated time depends on number of reviews
            
            **Step 5:** Review results
            - Statistics summary
            - Sortable/filterable table
            - Download results as CSV
            """)
            
            st.warning("**Note:** Large files (1000+ reviews) may take several minutes to process.")
        
        elif tutorial_step == "Understanding Results":
            st.subheader("Understanding Your Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Model Predictions:**
                - **ML Model:** Best for longer reviews (50+ words)
                - **Transformer:** Works well for all lengths
                - **Agreement:** When both models agree, prediction is more reliable
                
                **Confidence Levels:**
                - **90%+:** Very confident - highly reliable
                - **75-90%:** Confident - generally reliable
                - **<75%:** Lower confidence - review may be ambiguous
                """)
            
            with col2:
                st.markdown("""
                **Text Metrics:**
                - **Polarity:** Overall sentiment strength
                - **Subjectivity:** How opinionated vs factual
                - **Word Count:** Review length
                
                **When Models Disagree:**
                - Check confidence scores
                - Review may be neutral/mixed
                - Consider the context
                - Use comparison mode for details
                """)
            
            st.info("**Pro Tip:** Enable Comparison Mode to see detailed model analysis!")
        
        elif tutorial_step == "Advanced Features":
            st.subheader("Advanced Features")
            
            with st.expander("Settings", expanded=True):
                st.markdown("""
                **Dark Mode:** Toggle between light and dark themes
                
                **Quick Mode:** Skip detailed visualizations for faster predictions
                
                **Confidence Threshold:** Only show predictions above this threshold
                """)
            
            with st.expander("Prediction History", expanded=True):
                st.markdown("""
                - View your last 5-10 predictions in the sidebar
                - Click to view details
                - Clear history when needed
                """)
            
            with st.expander("Comparison Mode", expanded=True):
                st.markdown("""
                - Enable in Predict Sentiment page
                - See detailed model comparison
                - View agreement statistics
                """)
            
            with st.expander("Preprocessing Preview", expanded=True):
                st.markdown("""
                - See how text is processed before prediction
                - Understand cleaning and preprocessing steps
                - Helpful for debugging
                """)
    
    elif page == "Settings":
        st.header("Settings & Preferences")
        
        st.subheader("Display Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode,
                                                      help="Toggle between light and dark themes")
            st.session_state.quick_mode = st.checkbox("Quick Mode", value=st.session_state.quick_mode,
                                                      help="Skip detailed visualizations for faster predictions")
        
        with col2:
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                0.0, 100.0,
                st.session_state.confidence_threshold,
                5.0,
                help="Only show predictions above this confidence level"
            )
        
        st.markdown("---")
        st.subheader("Model Information")
        
        model_info_col1, model_info_col2 = st.columns(2)
        
        with model_info_col1:
            ml_status = "Loaded" if ml_model else "Not Available"
            st.markdown(f"""
            **Traditional ML Model**
            - **Type:** Logistic Regression
            - **Training Data:** 40,000 IMDB reviews
            - **Accuracy:** 88.45%
            - **Best For:** Longer reviews (50+ words)
            - **Status:** {ml_status}
            """)
        
        with model_info_col2:
            trans_status = "Loaded" if transformer_model else "Not Available"
            st.markdown(f"""
            **Transformer Model**
            - **Type:** DistilBERT (Hugging Face)
            - **Training:** Pre-trained + fine-tuned
            - **Accuracy:** ~84%
            - **Best For:** All text lengths
            - **Status:** {trans_status}
            """)
        
        st.markdown("---")
        st.subheader("Session Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Predictions Made", len(st.session_state.prediction_history))
        
        with info_col2:
            if st.session_state.prediction_history:
                recent = st.session_state.prediction_history[:5]
                agreements = sum(1 for p in recent if p.get('agreement') == True)
                st.metric("Recent Agreement Rate", f"{agreements/len(recent)*100:.1f}%" if recent else "N/A")
            else:
                st.metric("Recent Agreement Rate", "N/A")
        
        with info_col3:
            st.metric("Models Loaded", f"{'2' if ml_model and transformer_model else '1' if ml_model or transformer_model else '0'}/2")
        
        st.markdown("---")
        st.subheader("Reset Options")
        
        reset_col1, reset_col2, reset_col3 = st.columns(3)
        
        with reset_col1:
            if st.button("Reset All Settings", use_container_width=True):
                st.session_state.dark_mode = False
                st.session_state.quick_mode = False
                st.session_state.confidence_threshold = 0.0
                st.success("Settings reset!")
                st.rerun()
        
        with reset_col2:
            if st.button("Clear Prediction History", use_container_width=True):
                st.session_state.prediction_history = []
                st.success("History cleared!")
                st.rerun()
        
        with reset_col3:
            if st.button("Reload Models", use_container_width=True):
                st.cache_resource.clear()
                st.success("Models will reload on next prediction!")
                st.rerun()
    
    elif page == "About":
        st.header("About This Project")
        
        # Project Overview
        st.markdown("""
        ## IMDB Movie Review Sentiment Analysis Platform
        
        A comprehensive sentiment analysis project that analyzes **40,000 IMDB movie reviews** using 
        both traditional machine learning and modern transformer models.
        """)
        
        # Dataset Information
        st.subheader("Dataset")
        st.markdown("""
        - **Source**: [IMDB Movie Reviews Dataset](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) from Hugging Face
        - **Size**: 40,000 reviews (perfectly balanced: 20,000 positive, 20,000 negative)
        - **Average Length**: 232 words per review
        - **Range**: 4 to 2,470 words
        - **License**: Public dataset for research and educational purposes
        """)
        
        # Technologies
        st.subheader("Technologies Used")
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
        st.subheader("Model Performance")
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
        st.subheader("Key Features")
        st.markdown("""
        1. **Real-time Predictions**: Analyze sentiment of any movie review instantly
        2. **Dual Model Comparison**: Compare traditional ML vs transformer predictions
        3. **Comprehensive Analysis**: 15+ interactive visualizations
        4. **Topic Modeling**: Discover 5 distinct themes in movie reviews using LDA
        5. **Statistical Testing**: Hypothesis tests and correlation analysis
        6. **Professional Visualizations**: Interactive HTML charts and static images
        """)
        
        # Getting Started
        st.subheader("Getting Started")
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
        st.subheader("Project Structure")
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
        st.subheader("Deployment")
        st.markdown("""
        This app can be deployed to:
        - **Streamlit Cloud** (recommended, free): [streamlit.io](https://streamlit.io)
        - **Heroku**: Follow deployment guide
        - **Railway**: Easy Python app deployment
        - **Any cloud platform** that supports Python
        
        See `STREAMLIT_DEPLOYMENT.md` for detailed instructions.
        """)
        
        # Credits
        st.subheader("Credits & Resources")
        st.markdown("""
        - **Dataset**: [IMDB Movie Reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) by ajaykarthick
        - **Transformer Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face
        - **Libraries**: scikit-learn, transformers, NLTK, Plotly, and the open-source community
        
        ---
        
        **Built for data science and NLP enthusiasts**
        
        *This project demonstrates enterprise-level data science capabilities including large-scale data processing, 
        advanced NLP, machine learning, deep learning, and professional visualization.*
        """)
        
        # Show file structure
        with st.expander("Project Files"):
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

