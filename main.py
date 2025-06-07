import streamlit as st
import torch
import torch.nn as nn
import joblib
import zipfile
import os
import requests
from transformers import BertTokenizer, BertModel


# === Streamlit UI Config ===
st.set_page_config(page_title="Fake News Detector", page_icon="\U0001F4F0")

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Extract tokenizer.zip contents ===
TOKENIZER_ZIP = "tokenizer.zip"
TOKENIZER_FOLDER = "tokenizer_dir/tokenizer_dir"


if not os.path.exists(TOKENIZER_FOLDER):
    with zipfile.ZipFile(TOKENIZER_ZIP, 'r') as zip_ref:
        zip_ref.extractall(TOKENIZER_FOLDER)

# === Define LSTM model ===
class FakeNewsLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(FakeNewsLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT for faster inference
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        output = self.dropout(lstm_out[:, -1, :])
        return torch.sigmoid(self.fc(output))

# === Load Resources ===
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(TOKENIZER_FOLDER)

@st.cache_resource
def load_model():
    # Download model weights from Hugging Face repo
    url = "https://huggingface.co/nanostar2416/fake_news_lstm/resolve/main/final_fakenews_model_weights.pth"
    model_path = "final_fakenews_model_weights.pth"

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(url).content)

    # Initialize your model architecture
    model = FakeNewsLSTM(hidden_dim=64, output_dim=1).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# === Load ===
try:
    tokenizer = load_tokenizer()
    model = load_model()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# === Custom CSS ===
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            background-color: #FFF8E1 !important;
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
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# === App Layout ===
st.title("\U0001F4F0 Fake News Detection App")
st.markdown("Enter a news sentence or paragraph to classify whether it's **Fake** or **Real**.")

text_input = st.text_area("\U0001F4DD News Text Input")

if st.button("\U0001F50D Predict"):
    if not text_input.strip():
        st.warning("Please enter some text for prediction.")
    else:
        inputs = tokenizer(
            text_input,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=5
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            prediction = output.item()

        predicted_class = 1 if prediction > 0.5 else 0

        if predicted_class == 0:
            st.markdown(f"<div class='fake'>\u26A0\uFE0F Prediction: FAKE NEWS!!</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='real'>\u2705 Prediction: REAL NEWS \u2714\uFE0F</div>", unsafe_allow_html=True)
