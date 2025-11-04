from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download resources (first run only)
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("clickbait_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Setup FastAPI app
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Input schema
class Headline(BaseModel):
    headline: str

@app.get("/")
def root():
    return {"message": "Clickbait Detector API running"}

@app.post("/predict")
def predict(data: Headline):
    clean = preprocess(data.headline)
    vector = tfidf.transform([clean])
    pred = model.predict(vector)[0]
    return {"headline": data.headline,
            "prediction": "Clickbait" if pred == 1 else "Not Clickbait"}
