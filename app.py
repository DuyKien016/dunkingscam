# dunkingscam_streamlit_app_fixed.py
# Single-file Streamlit app (fixed)
# - Fixes: properly persist uploaded file across reruns, actually process uploaded image with Vintern OCR (if available),
#   robust error handling, clearer FAST_UI_ONLY flow, and small bug fixes (PIL resampling compatibility).
# - NOTES:
# 1) Models are heavy. Set `FAST_UI_ONLY = True` to skip loading models for UI testing.
# 2) Install requirements: pip install streamlit transformers torch torchvision pillow
# 3) Run: streamlit run dunkingscam_streamlit_app_fixed.py

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import io
import re
import textwrap
import time
import base64

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="DunkingScam AI", layout="wide", initial_sidebar_state="collapsed")

# Toggle for quick UI testing without loading heavy models
FAST_UI_ONLY = False  # <-- switch to False when you want to load actual models

# ----------------------- ENHANCED STYLES -----------------------
# (omitted here for brevity in the chat document) - keep same CSS as your original file
CSS = """
<!-- CSS TRUNCATED FOR READABILITY - paste your original CSS here if you want full styling -->
"""
st.markdown("<style>/* tiny reset to avoid streamlit header overlap */ header {display:none}</style>", unsafe_allow_html=True)

# ----------------------- MODEL LOADING -----------------------
@st.cache_resource
def load_phobert(phobert_path: str = "DuyKien016/phobert-scam-detector"):
    if FAST_UI_ONLY:
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(phobert_path)
        model.to(device).eval()
        return tokenizer, model
    except Exception as e:
        return None, None

@st.cache_resource
def load_vintern():
    if FAST_UI_ONLY:
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        vintern_model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
        vintern_tokenizer = AutoTokenizer.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            trust_remote_code=True
        )
        return vintern_model, vintern_tokenizer
    except Exception as e:
        return None, None

# Load models (or placeholders)
with st.spinner("Kiểm tra mô hình (nếu bật FAST_UI_ONLY = False thì có thể mất thời gian)..."):
    phobert_tokenizer, phobert_model = load_phobert()
    vintern_model, vintern_tokenizer = load_vintern()

# ----------------------- UTILITIES -----------------------
import torch
import torchvision.transforms as T

# Backwards-compatible resampling attribute for PIL
try:
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING_LANCZOS = Image.LANCZOS


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    return image


def vintern_ocr_extract(img_pil: Image.Image):
    """Run Vintern model to extract list of messages from an image.
    Returns (messages_list, error_message_or_None)
    """
    if vintern_model is None:
        return [], "Vintern model not available (FAST_UI_ONLY or load error)."
    try:
        img = img_pil.convert("RGB")
        img = enhance_image_for_ocr(img)
        max_size = (448, 448)
        img.thumbnail(max_size, RESAMPLING_LANCZOS)
        img = ImageOps.pad(img, max_size, color=(245, 245, 245))

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(img).unsqueeze(0)

        # Build prompt in Vietnamese similar to your original
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

        # Some Vintern variants implement a chat() helper; try calling it and fall back if not.
        try:
            response, *_ = vintern_model.chat(
                tokenizer=vintern_tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=dict(max_new_tokens=512, do_sample=False, num_beams=1, early_stopping=True),
                history=None,
                return_history=True
            )
        except Exception:
            # If vintern doesn't have chat(), try forward pass (this may fail depending on model code)
            return [], "Vintern model does not support chat() in this environment."

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
        # Return UNKNOWN results so UI still works in FAST_UI_ONLY mode
        return [{"text": t, "prediction": "UNKNOWN", "confidence": "0%"} for t in texts]

    results = []
    for text in texts:
        encoded = phobert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        encoded = {k: v.to(next(phobert_model.parameters()).device) for k, v in encoded.items()}
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

# ----------------------- JAVASCRIPT FOR AUTO SCROLL -----------------------

def add_auto_scroll_js():
    st.markdown("""
    <script>
    function scrollToBottom() {
        var chatContainer = document.querySelector('.chat-messages');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    setTimeout(scrollToBottom, 100);
    </script>
    """, unsafe_allow_html=True)

# ----------------------- PAGE NAV -----------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# store uploaded file bytes so they survive reruns and form clearing
if 'uploaded_file_bytes' not in st.session_state:
    st.session_state['uploaded_file_bytes'] = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None


def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return
    data = uploaded_file.read()
    st.session_state['uploaded_file_bytes'] = data
    st.session_state['uploaded_file_name'] = uploaded_file.name


def go_chat():
    st.session_state['page'] = 'chat'
    st.experimental_rerun()


def go_home():
    st.session_state['page'] = 'home'
    st.experimental_rerun()

# ----------------------- HOME PAGE -----------------------
if st.session_state['page'] == 'home':
    st.markdown("<h1 style='text-align:center'>DunkingScam AI (Demo)</h1>", unsafe_allow_html=True)
    st.write("Trang demo. Nhấn nút để qua giao diện chat.")
    if st.button("🚀 Bắt đầu ngay"):
        go_chat()

# ----------------------- CHAT PAGE -----------------------
elif st.session_state['page'] == 'chat':
    st.markdown("<h2>DunkingScam Chat</h2>", unsafe_allow_html=True)
    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-messages" id="chatMessages">', unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state['messages']):
            if msg['role'] == 'ai':
                st.markdown(f"""
                <div class="message">
                    <div class="message-bubble bubble-ai">{msg['text'].replace(chr(10), '<br>')}</div>
                    <div class="message-meta">{msg.get('meta', 'AI')} • {msg.get('time', 'vừa xong')}</div>
                </div>
                """, unsafe_allow_html=True)
            elif msg['role'] == 'user':
                st.markdown(f"""
                <div class="message message-user">
                    <div class="message-bubble bubble-user">{msg['text']}</div>
                    <div class="message-meta">{msg.get('meta', 'Bạn')} • {msg.get('time', 'vừa xong')}</div>
                </div>
                """, unsafe_allow_html=True)
            elif msg['role'] == 'typing':
                st.markdown("""
                <div class="typing-indicator">
                    <span style="color: #6b7280; font-size: 14px;">AI đang suy nghĩ</span>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.form(key='chat_form', clear_on_submit=True):
            uploaded_file = st.file_uploader("📸 Tải ảnh chụp màn hình tin nhắn", type=['png', 'jpg', 'jpeg'], key='file_input')
            text_input = st.text_area("💬 Hoặc gõ tin nhắn để kiểm tra...", height=80, key='text_input', placeholder="Nhập nội dung tin nhắn bạn muốn kiểm tra...")
            submit = st.form_submit_button("📤 Gửi", use_container_width=True)

            if submit:
                current_time = time.strftime("%H:%M")

                if uploaded_file is not None:
                    # persist uploaded file across reruns
                    save_uploaded_file(uploaded_file)
                    user_msg = f"📸 {uploaded_file.name}"
                elif text_input.strip():
                    user_msg = text_input.strip()
                else:
                    user_msg = "(tin nhắn trống)"

                st.session_state['messages'].append({'role': 'user', 'text': user_msg, 'meta': 'Bạn', 'time': current_time})
                st.session_state['messages'].append({'role': 'typing', 'text': '', 'meta': 'AI', 'time': current_time})

                # show typing indicator immediately
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='chat-sidebar'><h4>Hướng dẫn</h4><ul><li>Gõ hoặc tải ảnh để kiểm tra.</li></ul></div>", unsafe_allow_html=True)
        if st.button("🏠 Quay về trang chủ"):
            go_home()
        if st.button("🗑️ Xóa cuộc trò chuyện"):
            st.session_state['messages'] = []
            st.session_state['uploaded_file_bytes'] = None
            st.session_state['uploaded_file_name'] = None
            st.experimental_rerun()

    # ---------------- PROCESSING: trigger when last message was typing ----------------
    if st.session_state['messages'] and st.session_state['messages'][-1]['role'] == 'typing':
        # remove typing indicator
        st.session_state['messages'].pop()
        user_message = st.session_state['messages'][-1]
        current_time = time.strftime("%H:%M")

        # small delay to simulate processing
        time.sleep(0.8)

        try:
            if user_message['text'].startswith('📸'):
                # handle uploaded image
                if st.session_state['uploaded_file_bytes'] is None:
                    ai_response = "⚠️ Không tìm thấy ảnh. Vui lòng tải lại ảnh và nhấn Gửi."
                else:
                    img_bytes = st.session_state['uploaded_file_bytes']
                    img_name = st.session_state.get('uploaded_file_name', 'image')
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                    except Exception as e:
                        ai_response = f"❌ Không thể mở ảnh: {e}"
                    else:
                        messages_extracted, err = vintern_ocr_extract(img)
                        if err:
                            ai_response = f"⚠️ Vintern OCR không khả dụng: {err}\n\nBạn vẫn có thể dán văn bản trực tiếp để kiểm tra."
                        elif not messages_extracted:
                            ai_response = "⚠️ Không trích xuất được tin nhắn nào từ ảnh. Vui lòng thử ảnh khác hoặc dán trực tiếp văn bản." 
                        else:
                            # classify each extracted message
                            preds = phobert_predict(messages_extracted)
                            lines = ["🔍 **Kết quả phân tích ảnh**\n"]
                            for idx, p in enumerate(preds, start=1):
                                lines.append(f"- Tin nhắn {idx}: {p['prediction']} ({p['confidence']})\n  > {p['text']}")

                            ai_response = "\n".join(lines)
            elif user_message['text'] == "(tin nhắn trống)":
                ai_response = "⚠️ Bạn chưa nhập nội dung tin nhắn. Vui lòng nhập văn bản hoặc tải ảnh để phân tích."
            else:
                # plain text classification
                text = user_message['text']
                predictions = phobert_predict([text])
                pred = predictions[0]
                if pred['prediction'] == 'LỪA ĐẢO':
                    ai_response = (
                        f"🚨 **CẢNH BÁO LỪA ĐẢO**\n\n"
                        f"**Kết quả phân tích:** {pred['prediction']}\n"
                        f"**Độ tin cậy:** {pred['confidence']}\n\n"
                        "**Khuyến nghị:**\n- Không cung cấp thông tin cá nhân\n- Không chuyển tiền\n- Xác minh thông qua kênh chính thức\n\n"
                        f"**Nội dung đã phân tích:**\n\"{pred['text']}\""
                    )
                elif pred['prediction'] == 'UNKNOWN':
                    ai_response = (
                        "⚠️ Mô hình Phobert chưa được tải hoặc không khả dụng.\n"
                        "Bạn có thể bật FAST_UI_ONLY=False và cài đặt mô hình, hoặc dán trực tiếp văn bản để kiểm tra theo logic tĩnh."
                    )
                else:
                    ai_response = (
                        f"✅ **Tin nhắn an toàn**\n\n"
                        f"**Kết quả phân tích:** {pred['prediction']}\n"
                        f"**Độ tin cậy:** {pred['confidence']}\n\n"
                        f"**Nội dung đã phân tích:**\n\"{pred['text']}\"\n\n"
                        "💡 Lưu ý: Dù được đánh giá an toàn, hãy thận trọng với yêu cầu chuyển tiền hoặc thông tin nhạy cảm."
                    )
        except Exception as e:
            ai_response = f"❌ Lỗi xử lý: {e}"

        # append AI response and scroll
        st.session_state['messages'].append({'role': 'ai', 'text': ai_response, 'meta': 'AI Assistant', 'time': current_time})
        add_auto_scroll_js()
        st.experimental_rerun()

# EOF
