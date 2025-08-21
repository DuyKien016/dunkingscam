# dunkingscam_streamlit_app_fixed_v2.py
# Single-file Streamlit app (homepage + chat) - cleaned and fixed
# - FAST_UI_ONLY disabled: the app will attempt to load PhoBERT and Vintern models.
# - Robust safe_rerun wrapper, single definitions, persisted uploaded file bytes,
#   friendly error handling and clear structure.
# NOTES:
# 1) Models are heavy. Make sure you have transformers/torch and enough RAM/GPU.
# 2) Install requirements: pip install streamlit transformers torch torchvision pillow
# 3) Run: streamlit run dunkingscam_streamlit_app_fixed_v2.py

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import io
import re
import textwrap
import time
import torch
import torchvision.transforms as T

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="DunkingScam AI", layout="wide", initial_sidebar_state="collapsed")
# You said "bỏ fast ui luôn" => load models for real
FAST_UI_ONLY = False

# ----------------------- SAFE HELPERS -----------------------

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        # fallback to stop if rerun fails in some environments
        st.stop()

# small auto-scroll helper
def add_auto_scroll_js():
    st.markdown(
        """
        <script>
        function scrollToBottom() {
            var chatContainer = document.querySelector('.chat-messages');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        setTimeout(scrollToBottom, 100);
        </script>
        """,
        unsafe_allow_html=True,
    )

# ----------------------- MODEL LOADING -----------------------
@st.cache_resource
def load_phobert(phobert_path: str = "DuyKien016/phobert-scam-detector"):
    if FAST_UI_ONLY:
        return None, None, "cpu"
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(phobert_path)
        model.to(device).eval()
        return tokenizer, model, device
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_vintern(vintern_path: str = "5CD-AI/Vintern-1B-v3_5"):
    if FAST_UI_ONLY:
        return None, None, "cpu"
    try:
        from transformers import AutoModel, AutoTokenizer
        vintern_model = AutoModel.from_pretrained(
            vintern_path, trust_remote_code=True, low_cpu_mem_usage=True
        ).eval()
        vintern_tokenizer = AutoTokenizer.from_pretrained(vintern_path, trust_remote_code=True)
        device = next(vintern_model.parameters()).device
        return vintern_model, vintern_tokenizer, device
    except Exception as e:
        return None, None, str(e)

with st.spinner("Đang tải kiểm tra mô hình (nếu máy không đủ mạnh có thể báo lỗi)..."):
    phobert_tokenizer, phobert_model, phobert_device = load_phobert()
    vintern_model, vintern_tokenizer, vintern_device = load_vintern()

# ----------------------- UTILITIES -----------------------
# Pillow resampling compat
try:
    RESAMPLING = Image.Resampling.LANCZOS
except Exception:
    RESAMPLING = Image.LANCZOS


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.6)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    return image


def vintern_ocr_extract(img_pil: Image.Image):
    """Return (list_of_messages, error_or_None)"""
    if vintern_model is None:
        return [], "Vintern model not loaded (FAST_UI_ONLY or load error)."
    try:
        img = img_pil.convert("RGB")
        img = enhance_image_for_ocr(img)
        max_size = (448, 448)
        img.thumbnail(max_size, RESAMPLING)
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

        # Use model.chat if available
        try:
            response, *_ = vintern_model.chat(
                tokenizer=vintern_tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=dict(max_new_tokens=512, do_sample=False, num_beams=1, early_stopping=True),
                history=None,
                return_history=True,
            )
        except Exception as e:
            return [], f"Vintern inference error: {e}"

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
    if phobert_tokenizer is None or phobert_model is None:
        return [{"text": t, "prediction": "UNKNOWN", "confidence": "0%"} for t in texts]

    results = []
    for text in texts:
        try:
            encoded = phobert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            encoded = {k: v.to(phobert_device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = phobert_model(**encoded).logits
                probs = torch.softmax(logits, dim=1).squeeze()
                label = torch.argmax(probs).item()
            results.append({
                "text": text,
                "prediction": "LỪA ĐẢO" if label == 1 else "BÌNH THƯỜNG",
                "confidence": f"{(probs[label]*100).item():.2f}%"
            })
        except Exception as e:
            results.append({"text": text, "prediction": "ERROR", "confidence": str(e)})
    return results

# ----------------------- SESSION STATE SETUP -----------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'uploaded_file_bytes' not in st.session_state:
    st.session_state['uploaded_file_bytes'] = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None

# helpers to save uploaded file
def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return
    try:
        st.session_state['uploaded_file_bytes'] = uploaded_file.read()
        st.session_state['uploaded_file_name'] = uploaded_file.name
    except Exception as e:
        st.error(f"Không thể lưu file: {e}")

# nav helpers
def go_chat():
    st.session_state['page'] = 'chat'
    safe_rerun()

def go_home():
    st.session_state['page'] = 'home'
    safe_rerun()

# ----------------------- RENDERERS -----------------------
# Homepage renderer (kept compact but pretty)
def render_homepage():
    CSS_HOME = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    :root{--primary-blue:#1e40af;--accent-blue:#60a5fa;--light-blue:#dbeafe}
    .hero{background:linear-gradient(135deg,var(--primary-blue),#12275a);padding:48px;border-radius:12px;color:white}
    .feature{background:white;padding:18px;border-radius:12px}
    </style>
    """
    st.markdown(CSS_HOME, unsafe_allow_html=True)

    nav_col1, nav_col2 = st.columns([3,1])
    with nav_col1:
        st.markdown("<div style='display:flex;gap:10px;align-items:center'><i class='fas fa-shield-alt' style='color:var(--primary-blue);font-size:22px'></i><span style='font-weight:700;font-size:18px'>DunkingScam</span></div>", unsafe_allow_html=True)
    with nav_col2:
        if st.button("Đăng nhập", key="top_login"):
            st.info("Tính năng đăng nhập chưa triển khai.")

    st.write("")
    c1, c2 = st.columns([6,5])
    with c1:
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.markdown("<h1 style='margin:0 0 8px 0'>Bảo vệ bạn khỏi lừa đảo với <span style='color:var(--accent-blue)'>AI</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:rgba(255,255,255,0.9)'>Tải ảnh chụp màn hình hoặc dán nội dung — hệ thống trích xuất và phân loại ngay lập tức.</p>", unsafe_allow_html=True)
        if st.button("🚀 Bắt đầu ngay", key="home_start"):
            go_chat()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.image("https://img.freepik.com/free-photo/team-developers-doing-brainstorming-optimizing-code_482257-112972.jpg?fit=crop&w=800&h=500", caption="DunkingScam AI", use_column_width=True)

    st.markdown("---")
    t1, t2, t3 = st.columns(3)
    t1.metric("Độ chính xác", "99%")
    t2.metric("Tin nhắn trong dữ liệu", "2k+")
    t3.metric("F1-score", "0.99")

    st.markdown("### Tính năng")
    fcols = st.columns(4)
    features = [
        ("AI phân loại", "PhoBERT phân tích ngôn ngữ tiếng Việt"),
        ("OCR tích hợp", "Vintern trích xuất văn bản từ ảnh"),
        ("Giao diện thân thiện", "Dễ dùng, trực quan"),
        ("Khuyến nghị an toàn", "Hướng dẫn xử lý khi phát hiện lừa đảo")
    ]
    for col, (title, desc) in zip(fcols, features):
        with col:
            st.markdown(f"<div class='feature'><strong>{title}</strong><div style='color:#6b7280;margin-top:6px'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Mẹo an toàn nhanh")
    st.write("- Không chia sẻ OTP\n- Kiểm tra link trước khi nhấp\n- Xác thực qua kênh chính thức")

# Chat renderer
def render_chat_page():
    # header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:18px'>
      <div style='width:48px;height:48px;border-radius:50%;background:linear-gradient(90deg,#1e40af,#1e3a8a);display:flex;align-items:center;justify-content:center;color:white;font-weight:700'>AI</div>
      <div>
        <h3 style='margin:0'>DunkingScam AI</h3>
        <div style='color:#6b7280;font-size:13px'>Phân tích tin nhắn - an toàn & nhanh</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([7,3])

    # left: chat area
    with col_left:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-messages" style="height:520px;overflow-y:auto;padding:18px;background:white;border-radius:12px">', unsafe_allow_html=True)

        for msg in st.session_state['messages']:
            role = msg.get('role')
            text = msg.get('text','')
            meta = msg.get('meta','')
            tme = msg.get('time','')
            if role == 'ai':
                st.markdown(f"<div style='margin-bottom:12px'><div style='display:inline-block;background:#f3f4f6;color:#0f172a;padding:12px 14px;border-radius:14px;max-width:85%'>{text.replace(chr(10),'<br>')}</div><div style='font-size:12px;color:#6b7280;margin-top:6px'>{meta} • {tme}</div></div>", unsafe_allow_html=True)
            elif role == 'user':
                st.markdown(f"<div style='margin-bottom:12px;text-align:right'><div style='display:inline-block;background:#1e40af;color:white;padding:12px 14px;border-radius:14px;max-width:85%'>{text}</div><div style='font-size:12px;color:#6b7280;margin-top:6px'>{meta} • {tme}</div></div>", unsafe_allow_html=True)
            elif role == 'typing':
                st.markdown("<div style='margin-bottom:12px'><em style='color:#6b7280'>AI đang suy nghĩ...</em></div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # input
        st.markdown('<div style="background:white;padding:14px;border-radius:12px;margin-top:12px">', unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            uploaded_file = st.file_uploader("📸 Tải ảnh (png/jpg)", type=['png','jpg','jpeg'], key='uploader')
            text_input = st.text_area("💬 Hoặc gõ tin nhắn để kiểm tra", height=90, key='txt')
            submit = st.form_submit_button("📤 Gửi", use_container_width=True)

            if submit:
                now = time.strftime('%H:%M')
                if uploaded_file is not None:
                    save_uploaded_file(uploaded_file)
                    user_text = f"📸 {uploaded_file.name}"
                elif text_input and text_input.strip():
                    user_text = text_input.strip()
                else:
                    user_text = '(tin nhắn trống)'

                st.session_state['messages'].append({'role':'user','text':user_text,'meta':'Bạn','time':now})
                st.session_state['messages'].append({'role':'typing','text':'','meta':'AI','time':now})
                safe_rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # right: sidebar
    with col_right:
        st.markdown('<div style="background:white;padding:16px;border-radius:12px">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin:0 0 8px 0">🔍 Hướng dẫn</h4>', unsafe_allow_html=True)
        st.markdown('<ul style="color:#6b7280"><li>Gõ hoặc dán nội dung để kiểm tra nhanh.</li><li>Tải ảnh chụp màn hình để phân tích nhiều tin nhắn.</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write('')
        if st.button('🏠 Quay về trang chủ'):
            go_home()
        if st.button('🗑️ Xóa cuộc trò chuyện'):
            st.session_state['messages'] = []
            st.session_state['uploaded_file_bytes'] = None
            st.session_state['uploaded_file_name'] = None
            safe_rerun()

    # processing: if last message is typing
    if st.session_state['messages'] and st.session_state['messages'][-1].get('role') == 'typing':
        st.session_state['messages'].pop()  # remove typing
        user_msg = st.session_state['messages'][-1]
        now = time.strftime('%H:%M')
        # small delay to show indicator
        time.sleep(0.8)

        try:
            if user_msg['text'].startswith('📸'):
                if not st.session_state.get('uploaded_file_bytes'):
                    ai_response = '⚠️ Không tìm thấy ảnh trong session. Tải lại và thử lại.'
                else:
                    try:
                        img = Image.open(io.BytesIO(st.session_state['uploaded_file_bytes']))
                    except Exception as e:
                        ai_response = f'❌ Không thể mở ảnh: {e}'
                    else:
                        msgs, err = vintern_ocr_extract(img)
                        if err:
                            ai_response = f'⚠️ Vintern OCR không khả dụng: {err}'
                        elif not msgs:
                            ai_response = '⚠️ Không trích xuất được tin nhắn từ ảnh.'
                        else:
                            preds = phobert_predict(msgs)
                            lines = ['🔍 **KẾT QUẢ PHÂN TÍCH ẢNH**\n']
                            for i, p in enumerate(preds, 1):
                                lines.append(f"- Tin nhắn {i}: {p['prediction']} ({p['confidence']})\n  > {p['text']}")
                            ai_response = '\n'.join(lines)

            elif user_msg['text'] == '(tin nhắn trống)':
                ai_response = '⚠️ Bạn chưa nhập nội dung.'
            else:
                preds = phobert_predict([user_msg['text']])
                p = preds[0]
                if p['prediction'] == 'UNKNOWN':
                    ai_response = '⚠️ PhoBERT chưa được tải hoặc không khả dụng.'
                elif p['prediction'] == 'LỪA ĐẢO':
                    ai_response = (f"🚨 **CẢNH BÁO LỪA ĐẢO**\n\n**Kết quả:** {p['prediction']}\n**Độ tin cậy:** {p['confidence']}\n\n- Không cung cấp thông tin cá nhân\n- Không chuyển tiền\n- Xác minh bằng kênh chính thức\n\n**Nội dung:**\n\"{p['text']}\"")
                else:
                    ai_response = (f"✅ **Tin nhắn an toàn**\n\n**Kết quả:** {p['prediction']}\n**Độ tin cậy:** {p['confidence']}\n\n**Nội dung:**\n\"{p['text']}\"\n\n💡 Hãy luôn thận trọng với yêu cầu chuyển tiền.")
        except Exception as e:
            ai_response = f'❌ Lỗi xử lý: {e}'

        st.session_state['messages'].append({'role':'ai','text':ai_response,'meta':'AI Assistant','time':now})
        try:
            add_auto_scroll_js()
        except Exception:
            pass
        safe_rerun()

# ----------------------- MAIN -----------------------
if st.session_state.get('page','home') == 'home':
    render_homepage()
elif st.session_state.get('page') == 'chat':
    render_chat_page()

# EOF
