import streamlit as st
import torch
import os
import time
import queue
import threading
import re 
from transformers import BertTokenizer, BertForSequenceClassification
import pytchat
import gdown
def download_model():
    os.makedirs("models", exist_ok=True)
    model_path = "models/lyubomirt-toxicity-detector.pth"

    if not os.path.exists(model_path):
        with st.spinner("Downloading toxicity model (first run only)..."):
            gdown.download(
                id="1zJ__D4hWfXSM-dIkHsQ19SgdRWBNhXnV",
                output=model_path,
                quiet=False
            )
    return model_path


# --- Utility Function to Extract Video ID from URL ---

def extract_video_id(url_or_id):
    """
    Extracts the 11-character YouTube video ID from various URL formats or returns the ID if provided.
    """
    if not url_or_id:
        return None
        
    # 1. Check if it's already a valid 11-character ID (YouTube IDs are 11 chars)
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    # 2. Regex pattern to match various YouTube URLs (watch, youtu.be, embed, shorts, live)
    pattern = re.compile(
        r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e|embed|live)\/|.*[?&]v=|shorts\/)|youtu\.be\/|y2u\.be\/)([^"&?\/\s]{11})',
        re.I
    )
    
    match = pattern.search(url_or_id)
    if match:
        return match.group(1)
        
    return None

# --- Model and Tokenizer Loading ---

class ToxicClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(ToxicClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

@st.cache_resource
def load_model_and_tokenizer():
    MODEL_PATH = download_model() 
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult']
    n_classes = len(labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = ToxicClassifier(n_classes)
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found! Please make sure '{MODEL_PATH}' is in the same directory.")
            return None, None, None, None

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model, tokenizer, device, labels
    except Exception as e:
        st.error(f"Failed to load BERT model or tokenizer. Error: {e}")
        return None, None, None, None

model, tokenizer, device, labels = load_model_and_tokenizer()

# --- Prediction Function (Unchanged) ---

def predict_toxicity(text, model, tokenizer, device):
    if not text.strip():
        return None
        
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    result = {}
    for label, prob in zip(labels, probabilities):
        result[label] = {
            'probability': float(prob),
            'is_toxic': bool(prob > 0.5)
        }
    
    return result

# --- YouTube Chat Listener (Runs in Background Thread) ---

def youtube_chat_listener(video_id, chat_queue, stop_event):
    """
    Connects to YouTube Live Chat via pytchat and continuously reads messages.
    """
    try:
        # *** CRITICAL FIX APPLIED: interruptable=False bypasses Python's conflicting signal handler setup ***
        chat = pytchat.create(video_id=video_id, interruptable=False)
        chat_queue.put({"author": "System", "message": f"Attempting to connect to live chat for video ID: {video_id}..."})

        # Main polling loop
        while chat.is_alive() and not stop_event.is_set():
            for c in chat.get().sync_items():
                chat_queue.put({"author": c.author.name, "message": c.message})
            
            # Pause briefly to manage API request rate and CPU usage
            time.sleep(1) 

        if not chat.is_alive():
            chat_queue.put({"author": "System Alert", "message": "The live stream chat has ended or the video ID is invalid."})

    except Exception as e:
        # Catch and report any exceptions
        chat_queue.put({"author": "System Error", "message": f"Listener critical error. Try a different video. Details: {e}"})
    finally:
        chat_queue.put({"author": "System", "message": "Listener thread terminated."})


# --- Streamlit App UI ---

st.set_page_config(page_title="Real-Time YouTube Toxicity Detector", layout="wide")
st.title("🔴 Real-Time Toxicity Detection for YouTube Live Chat")
st.write("Enter a full **YouTube URL** or just the **Video ID** of a currently live stream.")

LABELS_TO_DISPLAY = ['toxic', 'threat', 'insult']
MAX_HISTORY = 10 
RERUN_INTERVAL = 1 # seconds

if model is None:
    st.stop()

# Initialize session state for stream control and chat history
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'chat_queue' not in st.session_state:
    st.session_state.chat_queue = queue.Queue()
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
# User input for the URL or video ID
url_input = st.text_input(
    "Enter YouTube URL or Video ID:", 
    placeholder="e.g., https://www.youtube.com/watch?v=g5_8p0Fh9uI or g5_8p0Fh9uI", 
    disabled=st.session_state.is_streaming
)

col1, col2 = st.columns([1, 4])
with col1:
    if st.session_state.is_streaming:
        stop_button = st.button("Stop Analysis", key="stop_stream")
        if stop_button:
            st.session_state.stop_event.set()
            st.session_state.is_streaming = False
            if st.session_state.thread and st.session_state.thread.is_alive():
                 st.session_state.thread.join(timeout=2) 
            st.session_state.thread = None
            st.session_state.stop_event.clear()
            st.info("Analysis stopped.")
            st.rerun()
    else:
        start_button = st.button("Start Analysis", key="start_stream", disabled=not url_input)
        if start_button and url_input:
            video_id = extract_video_id(url_input)
            
            if not video_id:
                st.error("Invalid YouTube URL or Video ID. Please check the input.")
            else:
                st.session_state.is_streaming = True
                st.session_state.video_id = video_id
                st.session_state.chat_history = []
                st.session_state.chat_queue = queue.Queue()
                st.session_state.stop_event = threading.Event()

                # Start the DAEMON thread
                st.session_state.thread = threading.Thread(
                    target=youtube_chat_listener, 
                    args=(st.session_state.video_id, st.session_state.chat_queue, st.session_state.stop_event),
                    daemon=True 
                )
                st.session_state.thread.start()
                st.rerun()

if st.session_state.is_streaming:
    st.subheader(f"Analyzing Live Chat for Video ID: **{st.session_state.video_id}**")
    
    chat_placeholder = st.empty()
    
    # --- Real-Time Processing Loop (Pulls from Queue) ---
    
    while not st.session_state.chat_queue.empty():
        try:
            item = st.session_state.chat_queue.get_nowait()
            author = item['author']
            message = item['message']
            
            # Handle System/Error messages
            if author.startswith("System"):
                st.session_state.chat_history.append(
                    f"<div style='color:orange; font-style:italic; margin-bottom: 5px;'>[{author}]: {message}</div>"
                )
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
                if author == "System Error":
                    st.session_state.is_streaming = False 
                    st.session_state.stop_event.set()
                continue
                
            # 1. Get Prediction
            predictions = predict_toxicity(message, model, tokenizer, device)

            # 2. Format the display
            if predictions:
                is_any_toxic = any(predictions[label]['is_toxic'] for label in LABELS_TO_DISPLAY)
                
                toxic_summary = []
                for label in LABELS_TO_DISPLAY:
                    prob = predictions[label]['probability']
                    if predictions[label]['is_toxic']:
                        toxic_summary.append(f"<span style='color:red;'>{label.title()} ({prob:.0%})</span>")
                
                if is_any_toxic:
                    display_html = (
                        f"<div style='border-left: 5px solid red; padding-left: 10px; margin-bottom: 5px; background-color: #120000;'>"
                        f"**🚨 {author}:** {message} <br/>"
                        f"**Toxicity Found:** {', '.join(toxic_summary)}"
                        f"</div>"
                    )
                else:
                    display_html = (
                        f"<div style='border-left: 5px solid green; padding-left: 10px; margin-bottom: 5px;'>"
                        f"**✅ {author}:** {message} <br/>"
                        f"<span style='color:green;'>Not Toxic in key categories.</span>"
                        f"</div>"
                    )

                st.session_state.chat_history.append(display_html)
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
            
        except queue.Empty:
            break
        except Exception as e:
            st.session_state.chat_history.append(f"<div style='color:red;'>**Stream Processing Error**: {e}</div>")
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
            break

    # Update the Streamlit placeholder
    chat_placeholder.markdown("<br>".join(st.session_state.chat_history), unsafe_allow_html=True)
    
    # Rerun to check for new messages in the queue (maintains the real-time loop)
    time.sleep(RERUN_INTERVAL)
    st.rerun()
