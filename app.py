# dunkingscam_streamlit_app.py
# Single-file Streamlit app that behaves like a 2-page app (Home + Chat)
# - Attempts to keep colors and layout similar to the HTML/Tailwind you provided
# - Loads Vintern (image OCR) and PhoBERT (text classifier) once using st.cache_resource
# - If user sends text -> only PhoBERT is used
# - If user uploads image -> Vintern OCR runs then PhoBERT classifies extracted messages
#
# NOTES:
# 1) Models are heavy. Set `FAST_UI_ONLY = True` to skip loading models for UI testing.
# 2) Install requirements: pip install streamlit transformers torch torchvision pillow
# 3) Run: streamlit run dunkingscam_streamlit_app.py

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import io
import re
import textwrap

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="DunkingScam AI", layout="wide")

# Toggle for quick UI testing without loading heavy models
FAST_UI_ONLY = False

# ----------------------- STYLES (mimic Tailwind colors) -----------------------
CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
:root{
  --primary-blue:#1e40af;
  --secondary-blue:#3b82f6;
  --light-blue:#dbeafe;
  --dark-blue:#1e3a8a;
  --accent-blue:#60a5fa;
}
body { background: var(--light-blue); }
.header-card{ background: linear-gradient(90deg,var(--primary-blue),var(--dark-blue)); color: white; padding: 14px; border-radius: 8px; }
.hero{ background: linear-gradient(135deg,var(--primary-blue),var(--dark-blue)); color:white; padding: 48px; border-radius:12px; }
.btn-primary{ background: white; color: var(--primary-blue); padding:12px 20px; border-radius:10px; font-weight:600; }
.stat-card{ background: #ffffffcc; padding:20px; border-radius:14px; }
.chat-bubble-user{ background:#1e40af; color:white; padding:12px 14px; border-radius:14px; display:inline-block; }
.chat-bubble-ai{ background:#f3f4f6; color:#111827; padding:12px 14px; border-radius:14px; display:inline-block; }
.chat-area{ background:white; border-radius:10px; padding:12px; height:520px; overflow:auto; }
.input-area{ background:white; padding:12px; border-top:1px solid #e5e7eb; }
.small-muted{ color:#64748b; font-size:0.9rem; }
.code-like{ background:#0f172a; color:#a78bfa; padding:8px; border-radius:8px; }
.typing-dots > div{ display:inline-block; width:7px; height:7px; background:#475569; border-radius:50%; margin-right:4px; animation: pop 1s infinite ease-in-out; }
.typing-dots > div:nth-child(2){ animation-delay: 0.15s; }
.typing-dots > div:nth-child(3){ animation-delay: 0.3s; }
@keyframes pop { 0%,100%{ transform: translateY(0); opacity:0.4 } 50%{ transform: translateY(-6px); opacity:1 } }

/* small responsive tweaks */
@media (max-width: 640px){ .hero{ padding:24px } .chat-area{ height:360px } }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ----------------------- MODEL LOADING -----------------------
@st.cache_resource
def load_phobert(phobert_path: str = "DuyKien016/phobert-scam-detector"):
    if FAST_UI_ONLY:
        return None, None, "cpu"
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(phobert_path).eval().to(device)
        return tokenizer, model, device
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_vintern():
    if FAST_UI_ONLY:
        return None, None, "cpu"
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        vintern_model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        vintern_tokenizer = AutoTokenizer.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            trust_remote_code=True
        )
        device = next(vintern_model.parameters()).device
        return vintern_model, vintern_tokenizer, device
    except Exception as e:
        return None, None, str(e)

# Load models (lazy; will only load when first called)
with st.spinner("Checking model availability..."):
    phobert_tokenizer, phobert_model, phobert_device = load_phobert()
    vintern_model, vintern_tokenizer, vintern_device = load_vintern()

# ----------------------- UTILITIES -----------------------
import torch
import torchvision.transforms as T

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    return image


def vintern_ocr_extract(img_pil: Image.Image):
    """Return list of extracted message strings using Vintern model.
    If vintern_model not available, returns [] and an error string.
    """
    if vintern_model is None:
        return [], "Vintern not available (FAST_UI_ONLY or load error)."
    try:
        img = img_pil.convert("RGB")
        img = enhance_image_for_ocr(img)
        max_size = (448, 448)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, max_size, color=(245, 245, 245))
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(img).unsqueeze(0).to(vintern_device)

        prompt = textwrap.dedent("""
        <image>
        Đọc từng tin nhắn trong ảnh và xuất ra định dạng:

        Tin nhắn 1: [nội dung]
        Tin nhắn 2: [nội dung]
        Tin nhắn 3: [nội dung]

        Quy tắc:
        - Mỗi ô chat = 1 tin nhắn
        - Chỉ lấy nội dung văn bản
        - Bỏ thời gian, tên người, emoji
        - Đọc từ trên xuống dưới

        Bắt đầu:
        """)

        # vintern API in user's snippet: vintern_model.chat(...)
        # We'll attempt the same call; wrap in try/except since model implementations vary.
        response, *_ = vintern_model.chat(
            tokenizer=vintern_tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=dict(
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                early_stopping=True
            ),
            history=None,
            return_history=True
        )
        # response is str containing lines like 'Tin nhắn 1: ...'
        messages = re.findall(r"Tin nhắn \d+: (.+?)(?=\nTin nhắn|\Z)", response, re.S)
        def quick_clean(msg):
            msg = re.sub(r"\s+", " ", msg.strip())
            msg = re.sub(r'^\d+[\.\)\-\s]+', '', msg)
            return msg.strip()
        cleaned = [quick_clean(m) for m in messages if m.strip()]
        return cleaned, None
    except Exception as e:
        return [], f"Error running Vintern: {e}"


def phobert_predict(texts):
    """Classify list of texts using PhoBERT. Returns list of dicts.
    If phobert not available, returns fallback predictions (Unknown).
    """
    if phobert_tokenizer is None or phobert_model is None:
        return [{"text": t, "prediction": "UNKNOWN", "confidence": "0%"} for t in texts]
    results = []
    for text in texts:
        encoded = phobert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        encoded = {k: v.to(phobert_model.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = phobert_model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            label = torch.argmax(probs).item()
        results.append({
            "text": text,
            "prediction": "LỪA ĐẢO" if label == 1 else "BÌNH THƯỜNG",
            "confidence": f"{(probs[label]*100).item():.2f}%"
        })
    return results

# ----------------------- PAGE NAV -----------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

def go_chat():
    st.session_state['page'] = 'chat'

# ----------------------- HOME PAGE -----------------------
if st.session_state['page'] == 'home':
    col1, col2 = st.columns([6,5])
    with col1:
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.markdown("""
        <h1 style='font-size:34px;margin:0 0 12px 0'>Bảo Vệ Bạn Khỏi Lừa Đảo Với<br /><span style='color:var(--accent-blue)'>Mô hình Trí tuệ nhân tạo</span></h1>
        <p style='font-size:18px;color:rgba(255,255,255,0.9);margin-bottom:16px'>Phân tích tin nhắn từ ảnh chụp màn hình để phát hiện lừa đảo tức thì.</p>
        """, unsafe_allow_html=True)
        if st.button("Bắt đầu ngay", key='start_btn'):
            go_chat()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.image("https://img.freepik.com/free-photo/team-developers-doing-brainstorming-optimizing-code_482257-112972.jpg?fit=crop&w=600&h=400")

    st.markdown("---")
    # Trust indicators
    t1, t2, t3 = st.columns(3)
    t1.markdown('<div class="stat-card"><h2 style="color:var(--primary-blue);font-size:28px;margin:0">99%</h2><p class="small-muted">Tỉ lệ chính xác</p></div>', unsafe_allow_html=True)
    t2.markdown('<div class="stat-card"><h2 style="color:var(--primary-blue);font-size:28px;margin:0">2K+</h2><p class="small-muted">Tin nhắn trong bộ dữ liệu huấn luyện</p></div>', unsafe_allow_html=True)
    t3.markdown('<div class="stat-card"><h2 style="color:var(--primary-blue);font-size:28px;margin:0">0.99</h2><p class="small-muted">F1-score</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    # Features (simple)
    st.header("Tính Năng Bảo Vệ Lừa Đảo Nâng Cao")
    f1, f2, f3, f4 = st.columns(4)
    f1.markdown("<div><i class='fas fa-brain' style='color:var(--primary-blue)'></i> AI phân loại thông minh</div>", unsafe_allow_html=True)
    f2.markdown("<div><i class='fas fa-envelope' style='color:var(--accent-blue)'></i> Phân tích nội dung tinh vi</div>", unsafe_allow_html=True)
    f3.markdown("<div><i class='fas fa-comment-dots' style='color:var(--primary-blue)'></i> OCR tích hợp</div>", unsafe_allow_html=True)
    f4.markdown("<div><i class='fas fa-database' style='color:var(--secondary-blue)'></i> Giao diện thân thiện</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.write("*Ghi chú*: Nếu muốn test giao diện nhanh, bật `FAST_UI_ONLY = True` trong file để skip load mô hình nặng.")

# ----------------------- CHAT PAGE -----------------------
elif st.session_state['page'] == 'chat':
    # Header
    st.markdown('<div class="header-card"><div style="display:flex;align-items:center;gap:12px"><div style="width:40px;height:40px;background:rgba(255,255,255,0.12);border-radius:50%;display:flex;align-items:center;justify-content:center"><i class="fas fa-lock" style="color:white"></i></div><div><h3 style="margin:0">DunkingScam AI</h3><div class="small-muted">Đoạn hội thoại an toàn</div></div><div style="margin-left:auto;display:flex;align-items:center;gap:8px"><div style="width:10px;height:10px;background:#10b981;border-radius:50%"></div><div class="small-muted">Online</div></div></div></div>', unsafe_allow_html=True)
    st.write("")

    col_left, col_right = st.columns([7,3])
    with col_left:
        # Chat area
        st.markdown('<div class="chat-area" id="chatMessages">', unsafe_allow_html=True)
        # Show conversation from session_state
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        for msg in st.session_state['messages']:
            if msg['role'] == 'ai':
                st.markdown(f"<div style='margin-bottom:12px'><div class='chat-bubble-ai'>{msg['text']}</div><div class='small-muted' style='margin-top:6px'>{msg.get('meta','AI')}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom:12px; text-align:right'><div class='chat-bubble-user'>{msg['text']}</div><div class='small-muted' style='margin-top:6px'>{msg.get('meta','You')}</div></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input area
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        with st.form(key='message_form', clear_on_submit=True):
            uploaded_file = st.file_uploader("Tải ảnh (chỉ khi muốn trích văn bản từ ảnh)", type=['png','jpg','jpeg'], key='u1')
            text_input = st.text_area("Hoặc gõ tin nhắn...", height=80, key='t1')
            submit = st.form_submit_button("Gửi")

            if submit:
                # Add user message to chat
                user_msg = (uploaded_file.name if uploaded_file else text_input) or "(empty)"
                st.session_state['messages'].append({'role':'user','text': user_msg, 'meta':'You'})

                # Decide pipeline
                if uploaded_file and uploaded_file.type.startswith('image'):
                    # Run Vintern OCR then PhoBERT
                    try:
                        img = Image.open(uploaded_file).convert('RGB')
                    except Exception:
                        img = None
                    if img is None:
                        st.session_state['messages'].append({'role':'ai','text':'Không thể mở ảnh.','meta':'AI'})
                    else:
                        with st.spinner('Đang trích xuất tin nhắn từ ảnh (Vintern)...'):
                            extracted, err = vintern_ocr_extract(img)
                        if err:
                            st.session_state['messages'].append({'role':'ai','text':f'Vintern lỗi: {err}','meta':'AI'})
                        elif not extracted:
                            st.session_state['messages'].append({'role':'ai','text':'Không trích được tin nhắn từ ảnh.','meta':'AI'})
                        else:
                            # classify each message
                            with st.spinner('Phân loại bằng PhoBERT...'):
                                preds = phobert_predict(extracted)
                            # show results
                            result_txt = ""
                            for i,p in enumerate(preds,1):
                                result_txt += f"Tin nhắn {i}: {p['prediction']} ({p['confidence']})\n{p['text']}\n\n"
                            st.session_state['messages'].append({'role':'ai','text':result_txt,'meta':'AI'})
                else:
                    # Text-only -> use PhoBERT
                    text = text_input.strip()
                    if not text:
                        st.session_state['messages'].append({'role':'ai','text':'Bạn chưa nhập nội dung.','meta':'AI'})
                    else:
                        with st.spinner('Phân loại bằng PhoBERT...'):
                            preds = phobert_predict([text])
                        p = preds[0]
                        st.session_state['messages'].append({'role':'ai','text':f"Kết quả: {p['prediction']} ({p['confidence']})\n{p['text']}", 'meta':'AI'})

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('''
        <div style='background:white;padding:12px;border-radius:10px'>
        <h4 style='margin:4px 0'>Hướng dẫn nhanh</h4>
        <ul style='color:#475569'>
          <li>Gửi văn bản để kiểm tra nhanh (dùng PhoBERT).</li>
          <li>Tải ảnh chụp màn hình chứa nhiều tin nhắn để trích văn bản (dùng Vintern + PhoBERT).</li>
          <li>Model nặng, nên lần đầu chạy sẽ mất thời gian để tải trọng số.</li>
        </ul>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        if st.button('Quay về trang chủ'):
            st.session_state['page'] = 'home'

# ----------------------- END -----------------------

