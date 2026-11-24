
import streamlit as st
from transformers import pipeline
import pandas as pd

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="EmotionSense | AI Studio",
    page_icon="💗",
    layout="wide"
)

# -----------------------------------
# CUSTOM DESIGN
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #17151A;
    color: #F7EEF2;
}

.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F8EAF0 !important;
}

p, label {
    color: #D6C7CF;
}

[data-testid="stMetric"] {
    background: #29232B;
    border: 1px solid #493744;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #DDB5C8 !important;
}

[data-testid="stMetricValue"] {
    color: #F4B9D0 !important;
}

.stTextArea textarea {
    background-color: #262229;
    color: #FFF5F8;
    border: 1px solid #76566A;
    border-radius: 12px;
}

.stTextArea textarea:focus {
    border-color: #E5A5C0;
}

.stButton > button {
    background-color: #DFA2BB;
    color: #281B24;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #F0BED1;
    color: #281B24;
    border: none;
}

.result-card {
    background: linear-gradient(135deg, #302631, #26212A);
    border: 1px solid #65465B;
    border-radius: 17px;
    padding: 24px;
    min-height: 145px;
}

.card-label {
    color: #D6A8BD;
    font-size: 13px;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.card-value {
    color: #F5C4D7;
    font-size: 29px;
    font-weight: 700;
    margin-top: 12px;
}

.card-subtitle {
    color: #C8B7C1;
    font-size: 13px;
    margin-top: 7px;
}

.section-heading {
    color: #F2C5D7;
    font-size: 21px;
    font-weight: 650;
    margin-top: 25px;
}

.top-line {
    border-top: 1px solid #453640;
    margin-top: 25px;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    color: #B99BAA;
    font-size: 13px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD AI MODELS
# -----------------------------------

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )


# -----------------------------------
# HEADER
# -----------------------------------

st.markdown("""
<div style="display:flex; align-items:center; gap:14px;">
    <div style="font-size:43px;">♡</div>
    <div>
        <div style="font-size:34px; font-weight:750;
                    color:#F4C1D4;">
            EmotionSense
        </div>
        <div style="color:#BDAAB5; font-size:13px;
                    letter-spacing:2px;">
            AI EMOTION INTELLIGENCE STUDIO
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<h2 style='font-size:27px;'>Understand the feeling behind the words.</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "Discover emotions and sentiment through deep learning-powered text analysis."
)

st.markdown("<div class='top-line'></div>", unsafe_allow_html=True)


# -----------------------------------
# TEXT INPUT
# -----------------------------------

st.markdown(
    "<div class='section-heading'>✦ Your text, your emotions</div>",
    unsafe_allow_html=True
)

user_text = st.text_area(
    "Enter a sentence or paragraph:",
    placeholder="Write something here... How are you feeling today?",
    height=150,
    key="user_text"
)

analyze = st.button("✦  Analyze my text", type="primary")


# -----------------------------------
# ANALYZE TEXT
# -----------------------------------

if analyze:

    if not user_text.strip():

        st.warning("Please enter some text before analyzing.")

    else:

        try:
            with st.spinner("Understanding your words..."):

                emotion_model = load_emotion_model()
                sentiment_model = load_sentiment_model()

                # Emotion prediction
                emotion_results = emotion_model(
                    user_text,
                    truncation=True
                )

                if emotion_results and isinstance(
                    emotion_results[0], list
                ):
                    emotion_results = emotion_results[0]

                emotion_results = sorted(
                    emotion_results,
                    key=lambda x: x["score"],
                    reverse=True
                )

                # Sentiment prediction
                sentiment_results = sentiment_model(
                    user_text,
                    truncation=True
                )

                if sentiment_results and isinstance(
                    sentiment_results[0], list
                ):
                    sentiment_results = sentiment_results[0]

                sentiment_results = sorted(
                    sentiment_results,
                    key=lambda x: x["score"],
                    reverse=True
                )

                st.session_state["analysis"] = {
                    "text": user_text,
                    "emotions": emotion_results,
                    "sentiments": sentiment_results
                }

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    emotions = analysis["emotions"]
    sentiments = analysis["sentiments"]

    best_emotion = emotions[0]
    best_sentiment = sentiments[0]

    emotion_name = best_emotion["label"].upper()
    sentiment_name = best_sentiment["label"].upper()

    emotion_score = best_emotion["score"] * 100
    sentiment_score = best_sentiment["score"] * 100

    st.markdown("<div class='top-line'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-heading'>Your analysis results</div>",
        unsafe_allow_html=True
    )

    # Result cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="card-label">DETECTED EMOTION</div>
            <div class="card-value">{emotion_name}</div>
            <div class="card-subtitle">
                {emotion_score:.2f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="card-label">OVERALL SENTIMENT</div>
            <div class="card-value">{sentiment_name}</div>
            <div class="card-subtitle">
                {sentiment_score:.2f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Emotion chart
    st.markdown(
        "<div class='section-heading'>♡ Emotion breakdown</div>",
        unsafe_allow_html=True
    )

    emotion_data = pd.DataFrame({
        "Emotion": [
            item["label"].capitalize()
            for item in emotions
        ],
        "Confidence (%)": [
            item["score"] * 100
            for item in emotions
        ]
    })

    st.bar_chart(
        emotion_data.set_index("Emotion"),
        y="Confidence (%)"
    )

    # Sentiment chart
    st.markdown(
        "<div class='section-heading'>✦ Sentiment breakdown</div>",
        unsafe_allow_html=True
    )

    sentiment_data = pd.DataFrame({
        "Sentiment": [
            item["label"].capitalize()
            for item in sentiments
        ],
        "Confidence (%)": [
            item["score"] * 100
            for item in sentiments
        ]
    })

    st.bar_chart(
        sentiment_data.set_index("Sentiment"),
        y="Confidence (%)"
    )

    # Download report
    report = f"""
EMOTIONSENSE AI - ANALYSIS REPORT

Input Text:
{analysis["text"]}

DETECTED EMOTION:
{emotion_name} ({emotion_score:.2f}%)

EMOTION BREAKDOWN:
"""

    for item in emotions:
        report += (
            f'{item["label"]}: '
            f'{item["score"] * 100:.2f}%\n'
        )

    report += f"""
OVERALL SENTIMENT:
{sentiment_name} ({sentiment_score:.2f}%)

SENTIMENT BREAKDOWN:
"""

    for item in sentiments:
        report += (
            f'{item["label"]}: '
            f'{item["score"] * 100:.2f}%\n'
        )

    st.markdown("<div class='top-line'></div>", unsafe_allow_html=True)

    st.download_button(
        label="↓  Download analysis report",
        data=report,
        file_name="EmotionSense_Report.txt",
        mime="text/plain"
    )


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("""
<div class="top-line"></div>

<div class="footer">
    EmotionSense AI &nbsp; ♡ &nbsp; Built with curiosity and a little pink.
    <br><br>
    Deep Learning B.Tech Project
</div>
""", unsafe_allow_html=True)
