import streamlit as st
import torch
import torch.nn as nn
import joblib
import zipfile
import os
from transformers import BertTokenizer

# === Streamlit UI Config ===
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Extract tokenizer.zip contents ===
TOKENIZER_ZIP = "tokenizer_LSTM.zip"
TOKENIZER_FOLDER = "tokenizer"

if not os.path.exists(TOKENIZER_FOLDER):
    with zipfile.ZipFile(TOKENIZER_ZIP, 'r') as zip_ref:
        zip_ref.extractall(TOKENIZER_FOLDER)

# === Define LSTM model ===
class FakeNewsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FakeNewsLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.dropout(lstm_out[:, -1, :])
        return torch.sigmoid(self.fc(out))

# === Cache resource functions ===
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(TOKENIZER_FOLDER)

@st.cache_resource
def load_model():
    model = FakeNewsLSTM(30522, 64, 1).to(device)
    model.load_state_dict(torch.load("LSTM.pth", map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder_LSTM.joblib")

# === Load cached resources ===
try:
    tokenizer = load_tokenizer()
    model = load_model()
    label_encoder = load_label_encoder()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# === Custom CSS ===
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #FFF8E1 !important;  /* Soft light beige */
        }

        .fake {
            animation: pulse 1.5s infinite;
            color: white;
            background-color: #C62828;
            padding: 1em;
            border-radius: 8px;
            font-weight: bold;
        }

        .real {
            color: white;
            background-color: #2E7D32;
            padding: 1em;
            border-radius: 8px;
            font-weight: bold;
            animation: fadein 1s ease-in-out;
        }

        @keyframes pulse {
            0% {box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.7);}
            70% {box-shadow: 0 0 0 10px rgba(198, 40, 40, 0);}
            100% {box-shadow: 0 0 0 0 rgba(198, 40, 40, 0);}
        }

        @keyframes fadein {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        .bar-fill {
            height: 100%;
            background-color: white;
            border-radius: 10px;
        }

        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# === App Layout ===
st.title("📰 Fake News Detection App")
st.markdown("Enter a news sentence or paragraph to classify whether it's **Fake** or **Real**.")

text_input = st.text_area("📝 News Text Input")

if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter some text for prediction.")
    else:
        # Tokenize input
        inputs = tokenizer(
            text_input,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=200
        )
        input_ids = inputs['input_ids'].to(device)

        # Predict
        with torch.no_grad():
            output = model(input_ids)
            prediction = output.item()

        predicted_class = 1 if prediction > 0.5 else 0
        class_label = str(label_encoder.inverse_transform([predicted_class])[0])

        # === Display Result ===
        if predicted_class == 0:
            st.markdown(f"<div class='fake'>🚨 Prediction: FAKE NEWS!!</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='real'>✅ Prediction: REAL NEWS ✔️</div>", unsafe_allow_html=True)
