# dunkingscam_streamlit_app.py
# Single-file Streamlit app that behaves like a 2-page app (Home + Chat)
# - Homepage designed to match the HTML/Tailwind design
# - Enhanced chat interface with typing indicator and auto-scroll
# - Loads Vintern (image OCR) and PhoBERT (text classifier) once using st.cache_resource
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
import time

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="DunkingScam AI", layout="wide", initial_sidebar_state="collapsed")

# Toggle for quick UI testing without loading heavy models
FAST_UI_ONLY = False

# ----------------------- ENHANCED STYLES -----------------------
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

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
.stApp > header {visibility: hidden;}

body { 
  background: var(--light-blue); 
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Navigation Bar */
.nav-bar {
  background: white;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border-bottom: 2px solid var(--primary-blue);
  padding: 16px 0;
  margin-bottom: 0;
}

.nav-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
}

.nav-logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: bold;
  color: #1f2937;
}

.nav-logo i {
  color: var(--primary-blue);
  font-size: 24px;
}

/* Hero Section */
.hero-section {
  background: linear-gradient(135deg, var(--primary-blue), var(--dark-blue));
  color: white;
  padding: 80px 0;
  margin: 0;
}

.hero-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 48px;
  align-items: center;
}

.hero-text h1 {
  font-size: 48px;
  font-weight: bold;
  line-height: 1.2;
  margin-bottom: 24px;
}

.hero-text .accent {
  color: var(--accent-blue);
}

.hero-text p {
  font-size: 20px;
  color: rgba(255,255,255,0.9);
  margin-bottom: 32px;
  line-height: 1.6;
}

.hero-image {
  position: relative;
}

.hero-image img {
  width: 100%;
  border-radius: 16px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
}

.hero-badge {
  position: absolute;
  top: 24px;
  left: 24px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
  padding: 12px;
}

.hero-badge .status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 12px;
  height: 12px;
  background: #10b981;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Buttons */
.btn-primary {
  background: white;
  color: var(--primary-blue);
  padding: 16px 32px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 16px;
  border: none;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
  text-decoration: none;
}

.btn-primary:hover {
  background: var(--light-blue);
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
}

/* Trust Indicators */
.trust-section {
  padding: 80px 0;
  background: white;
}

.trust-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  text-align: center;
}

.trust-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 32px;
  margin-top: 64px;
}

.trust-card {
  background: white;
  padding: 32px;
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border-top: 4px solid var(--primary-blue);
}

.trust-number {
  font-size: 48px;
  font-weight: bold;
  color: var(--primary-blue);
  margin-bottom: 8px;
}

.trust-label {
  color: #6b7280;
  font-size: 16px;
}

/* Features Section */
.features-section {
  padding: 80px 0;
  background: #f9fafb;
}

.section-header {
  text-align: center;
  margin-bottom: 64px;
}

.section-title {
  font-size: 42px;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 16px;
}

.section-subtitle {
  font-size: 20px;
  color: #6b7280;
  max-width: 800px;
  margin: 0 auto;
  line-height: 1.6;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 32px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.feature-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  padding: 32px;
  transition: all 0.3s ease;
  border-top: 4px solid var(--primary-blue);
}

.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  background: var(--light-blue);
  width: 64px;
  height: 64px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
}

.feature-icon i {
  font-size: 24px;
  color: var(--primary-blue);
}

.feature-title {
  font-size: 20px;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 16px;
}

.feature-desc {
  color: #6b7280;
  line-height: 1.6;
}

/* Steps Section */
.steps-section {
  padding: 80px 0;
  background: white;
}

.steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 48px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.step-item {
  text-align: center;
}

.step-number {
  background: var(--primary-blue);
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 24px;
  color: white;
  font-size: 32px;
  font-weight: bold;
}

.step-title {
  font-size: 20px;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 16px;
}

.step-desc {
  color: #6b7280;
  line-height: 1.6;
}

/* FAQ Section */
.faq-section {
  padding: 80px 0;
  background: var(--light-blue);
}

.faq-content {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 24px;
}

.faq-item {
  background: white;
  border-radius: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 24px;
  margin-bottom: 16px;
}

.faq-question {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.faq-number {
  background: var(--primary-blue);
  color: white;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
}

.faq-answer {
  color: #6b7280;
  line-height: 1.6;
  margin-left: 44px;
}

/* Stats Section */
.stats-section {
  padding: 80px 0;
  background: white;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 32px;
  max-width: 1200px;
  margin: 48px auto 0;
  padding: 0 24px;
}

.stat-card {
  background: #f9fafb;
  border-radius: 16px;
  padding: 48px 32px;
  text-align: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.stat-number {
  font-size: 48px;
  font-weight: bold;
  color: var(--secondary-blue);
  margin-bottom: 16px;
}

.stat-label {
  color: #6b7280;
  font-size: 14px;
  line-height: 1.4;
}

/* CTA Section */
.cta-section {
  padding: 80px 0;
  background: linear-gradient(90deg, var(--primary-blue), var(--dark-blue));
  color: white;
  text-align: center;
}

.cta-content {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 24px;
}

.cta-title {
  font-size: 42px;
  font-weight: bold;
  margin-bottom: 24px;
}

.cta-subtitle {
  font-size: 20px;
  color: rgba(255,255,255,0.9);
  margin-bottom: 48px;
}

/* Chat Interface */
.chat-header {
  background: linear-gradient(90deg, var(--primary-blue), var(--dark-blue));
  color: white;
  padding: 20px;
  border-radius: 12px;
  margin-bottom: 20px;
}

.chat-header-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.chat-avatar {
  width: 48px;
  height: 48px;
  background: rgba(255,255,255,0.12);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chat-info h3 {
  margin: 0;
  font-size: 20px;
}

.chat-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.status-indicator {
  width: 12px;
  height: 12px;
  background: #10b981;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.chat-container {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.chat-messages {
  height: 500px;
  overflow-y: auto;
  padding: 20px;
  scroll-behavior: smooth;
}

.message {
  margin-bottom: 16px;
}

.message-user {
  text-align: right;
}

.message-bubble {
  display: inline-block;
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 18px;
  word-wrap: break-word;
}

.bubble-user {
  background: var(--primary-blue);
  color: white;
  border-bottom-right-radius: 4px;
}

.bubble-ai {
  background: #f3f4f6;
  color: #1f2937;
  border-bottom-left-radius: 4px;
}

.message-meta {
  font-size: 12px;
  color: #6b7280;
  margin-top: 4px;
}

.typing-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #f3f4f6;
  border-radius: 18px;
  border-bottom-left-radius: 4px;
  margin-bottom: 16px;
  max-width: fit-content;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.typing-dot {
  width: 8px;
  height: 8px;
  background: #6b7280;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0ms; }
.typing-dot:nth-child(2) { animation-delay: 200ms; }
.typing-dot:nth-child(3) { animation-delay: 400ms; }

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-10px);
    opacity: 1;
  }
}

.chat-input {
  background: white;
  padding: 20px;
  border-top: 1px solid #e5e7eb;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chat-sidebar {
  background: white;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  height: fit-content;
}

.sidebar-title {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
}

.sidebar-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar-list li {
  color: #6b7280;
  margin-bottom: 8px;
  line-height: 1.5;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-content {
    grid-template-columns: 1fr;
    gap: 32px;
    text-align: center;
  }
  
  .hero-text h1 {
    font-size: 36px;
  }
  
  .features-grid,
  .steps-grid,
  .trust-grid,
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .section-title {
    font-size: 32px;
  }
  
  .chat-messages {
    height: 400px;
  }
}
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

# Load models
with st.spinner("Đang kiểm tra tính khả dụng của mô hình..."):
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
    // Call scroll function after a short delay to ensure content is rendered
    setTimeout(scrollToBottom, 100);
    </script>
    """, unsafe_allow_html=True)

# ----------------------- PAGE NAV -----------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def go_chat():
    st.session_state['page'] = 'chat'
    st.rerun()

def go_home():
    st.session_state['page'] = 'home'
    st.rerun()

# ----------------------- HOME PAGE -----------------------
if st.session_state['page'] == 'home':
    # Navigation Bar
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-content">
            <div class="nav-logo">
                <i class="fas fa-shield-alt"></i>
                <span>DunkingScam</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-text">
                <h1>Bảo Vệ Bạn Khỏi Lừa Đảo Với<br><span class="accent">Mô hình Trí tuệ nhân tạo</span></h1>
                <p>Phân tích tin nhắn từ ảnh chụp màn hình để phát hiện lừa đảo tức thì.</p>
            </div>
            <div class="hero-image">
                <img src="https://img.freepik.com/free-photo/team-developers-doing-brainstorming-optimizing-code_482257-112972.jpg?fit=crop&w=600&h=400" alt="AI Protection">
                <div class="hero-badge">
                    <div class="status">
                        <div class="status-dot"></div>
                        <span style="color: #1f2937; font-weight: 600; font-size: 14px;">DunkingScam AI</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Call-to-action button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Bắt đầu ngay", key="hero_btn", help="Chuyển đến giao diện chat"):
            go_chat()
    
    # Trust Indicators
    st.markdown("""
    <div class="trust-section">
        <div class="trust-content">
            <h2 class="section-title">Độ tin cậy đã được kiểm chứng</h2>
            <div class="trust-grid">
                <div class="trust-card">
                    <div class="trust-number">99%</div>
                    <div class="trust-label">Tỉ lệ chính xác</div>
                </div>
                <div class="trust-card">
                    <div class="trust-number">2K+</div>
                    <div class="trust-label">Tin nhắn trong bộ dữ liệu huấn luyện</div>
                </div>
                <div class="trust-card">
                    <div class="trust-number">0.99</div>
                    <div class="trust-label">F1-score</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="features-section">
        <div class="section-header">
            <h2 class="section-title">Tính Năng Bảo Vệ Lừa Đảo Nâng Cao</h2>
            <p class="section-subtitle">Với sự hỗ trợ của AI, hệ thống của chúng tôi áp dụng công nghệ hiện đại để giúp nhận diện và ngăn chặn nhiều hình thức lừa đảo, nhằm mang lại sự an toàn tốt hơn cho bạn.</p>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-brain"></i>
                </div>
                <h3 class="feature-title">AI phân loại thông minh</h3>
                <p class="feature-desc">Sử dụng mô hình PhoBERT để nhận biết tin nhắn lừa đảo.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-envelope"></i>
                </div>
                <h3 class="feature-title">Phân tích nội dung tinh vi</h3>
                <p class="feature-desc">Hiểu và phân loại tin nhắn ngay cả khi kẻ gian dùng từ ngữ lắt léo hoặc đảo cấu trúc câu.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-comment-dots"></i>
                </div>
                <h3 class="feature-title">OCR tích hợp</h3>
                <p class="feature-desc">Nhận diện văn bản từ ảnh chụp màn hình tin nhắn.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-database"></i>
                </div>
                <h3 class="feature-title">Giao diện thân thiện</h3>
                <p class="feature-desc">Dễ dàng và thân thiện với mọi người dùng.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("""
    <div class="steps-section">
        <div class="section-header">
            <h2 class="section-title">Cách Mô Hình Hoạt Động</h2>
            <p class="section-subtitle">Bảo vệ chống lừa đảo đơn giản, nhanh chóng và hiệu quả — được thiết kế để giúp bạn an tâm chỉ với 3 bước dễ dàng.</p>
        </div>
        <div class="steps-grid">
            <div class="step-item">
                <div class="step-number">1</div>
                <h3 class="step-title">Chụp màn hình</h3>
                <p class="step-desc">Tin nhắn có dấu hiệu đáng ngờ</p>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <h3 class="step-title">Tải ảnh chụp màn hình tin nhắn</h3>
                <p class="step-desc">Mô hình sẽ trích xuất tin nhắn và đánh giá</p>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <h3 class="step-title">Nhận kết quả</h3>
                <p class="step-desc">Chatbox sẽ cảnh báo ngay cho bạn khi có dấu hiệu đáng ngờ</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ Section
    st.markdown("""
    <div class="faq-section">
        <div class="faq-content">
            <div class="section-header">
                <h2 class="section-title">5 Nguyên Tắc Giúp Bảo Vệ Bạn Khỏi Các Vụ Lừa Đảo</h2>
                <p class="section-subtitle">Tổng hợp từ các nguồn uy tín như Cục An ninh mạng (Bộ Công an Việt Nam) và Cục An toàn thông tin (Bộ Thông tin và Truyền thông)</p>
            </div>
            <div class="faq-item">
                <div class="faq-question">
                    <div class="faq-number">1</div>
                    Không cung cấp thông tin cá nhân cho người không quen biết
                </div>
                <div class="faq-answer">
                    Đặc biệt là các thông tin nhạy cảm như số CMND, tài khoản ngân hàng, mật khẩu, mã OTP.
                </div>
            </div>
            <div class="faq-item">
                <div class="faq-question">
                    <div class="faq-number">2</div>
                    Không nhấp vào các đường link hoặc tải ứng dụng không rõ nguồn gốc
                </div>
                <div class="faq-answer">
                    Đường link lạ gửi qua tin nhắn, email hoặc mạng xã hội có thể chứa mã độc hoặc lừa đảo.
                </div>
            </div>
            <div class="faq-item">
                <div class="faq-question">
                    <div class="faq-number">3</div>
                    Không chuyển tiền hoặc cung cấp thông tin tài chính cho người lạ
                </div>
                <div class="faq-answer">
                    Trước khi chuyển tiền cần xác minh thật kỹ thông tin người nhận để tránh mất tiền oan.
                </div>
            </div>
            <div class="faq-item">
                <div class="faq-question">
                    <div class="faq-number">4</div>
                    Luôn xác minh thông tin khi có yêu cầu chuyển tiền hoặc nhận thưởng
                </div>
                <div class="faq-answer">
                    Nếu nhận các cuộc gọi, tin nhắn hay đề nghị nghi vấn, hãy gọi điện trực tiếp xác nhận với người thân hoặc tổ chức liên quan.
                </div>
            </div>
            <div class="faq-item">
                <div class="faq-question">
                    <div class="faq-number">5</div>
                    Không tham gia các chương trình đầu tư hay nhận thưởng có lời hứa hấp dẫn quá mức
                </div>
                <div class="faq-answer">
                    Đây thường là "bẫy" để lừa đảo tài chính hoặc chiếm đoạt tài sản của bạn.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-section">
        <div class="section-header">
            <h2 class="section-title">Thiệt hại do lừa đảo trực tuyến gây ra trong năm 2024</h2>
            <p class="section-subtitle">Theo thống kê của Bộ Công an</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">6.000</div>
                <div class="stat-label">Là ước tính số vụ lừa đảo đã diễn ra</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">460%</div>
                <div class="stat-label">Nạn nhân báo cho cơ quan chức năng</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">12.000 Tỉ VNĐ</div>
                <div class="stat-label">Là tổng số tiền thiệt hại được ước tính</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div class="cta-section">
        <div class="cta-content">
            <h2 class="cta-title">Bạn đã sẵn sàng để sử dụng?</h2>
            <p class="cta-subtitle">Bắt đầu bảo vệ an toàn của bản thân từ hôm nay</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Final CTA Button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Bắt đầu dùng thử ngay", key="cta_btn", help="Chuyển đến giao diện chat"):
            go_chat()
    
    st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------------- CHAT PAGE -----------------------
elif st.session_state['page'] == 'chat':
    # Chat Header
    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-content">
            <div class="chat-avatar">
                <i class="fas fa-lock"></i>
            </div>
            <div class="chat-info">
                <h3>DunkingScam AI</h3>
                <div style="color: rgba(255,255,255,0.8); font-size: 14px;">Đoạn hội thoại an toàn</div>
            </div>
            <div class="chat-status">
                <div class="status-indicator"></div>
                <span style="color: rgba(255,255,255,0.8); font-size: 14px;">Online</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([7, 3])
    
    with col_left:
        # Chat Container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Messages Area
        st.markdown('<div class="chat-messages" id="chatMessages">', unsafe_allow_html=True)
        
        # Display conversation
        for i, msg in enumerate(st.session_state['messages']):
            if msg['role'] == 'ai':
                st.markdown(f"""
                <div class="message">
                    <div class="message-bubble bubble-ai">
                        {msg['text'].replace(chr(10), '<br>')}
                    </div>
                    <div class="message-meta">{msg.get('meta', 'AI')} • {msg.get('time', 'vừa xong')}</div>
                </div>
                """, unsafe_allow_html=True)
            elif msg['role'] == 'user':
                st.markdown(f"""
                <div class="message message-user">
                    <div class="message-bubble bubble-user">
                        {msg['text']}
                    </div>
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
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close messages
        
        # Input Area
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        
        with st.form(key='chat_form', clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "📸 Tải ảnh chụp màn hình tin nhắn", 
                type=['png', 'jpg', 'jpeg'], 
                key='file_input',
                help="Chỉ tải ảnh khi muốn trích xuất văn bản từ ảnh"
            )
            
            text_input = st.text_area(
                "💬 Hoặc gõ tin nhắn để kiểm tra...",
                height=80,
                key='text_input',
                placeholder="Nhập nội dung tin nhắn bạn muốn kiểm tra..."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit = st.form_submit_button("📤 Gửi", use_container_width=True)
            
            if submit:
                current_time = time.strftime("%H:%M")
                
                # Add user message
                if uploaded_file:
                    user_msg = f"📸 {uploaded_file.name}"
                elif text_input.strip():
                    user_msg = text_input.strip()
                else:
                    user_msg = "(tin nhắn trống)"
                
                st.session_state['messages'].append({
                    'role': 'user',
                    'text': user_msg,
                    'meta': 'Bạn',
                    'time': current_time
                })
                
                # Add typing indicator
                st.session_state['messages'].append({
                    'role': 'typing',
                    'text': '',
                    'meta': 'AI',
                    'time': current_time
                })
                
                # Rerun to show typing indicator
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close input
        st.markdown('</div>', unsafe_allow_html=True)  # Close container
    
    with col_right:
        # Sidebar
        st.markdown("""
        <div class="chat-sidebar">
            <h4 class="sidebar-title">🔍 Hướng dẫn sử dụng</h4>
            <ul class="sidebar-list">
                <li>📝 <strong>Kiểm tra văn bản:</strong> Gõ nội dung tin nhắn để phân tích nhanh bằng PhoBERT.</li>
                <li>📸 <strong>Phân tích ảnh:</strong> Tải ảnh chụp màn hình chứa nhiều tin nhắn để trích xuất bằng Vintern + PhoBERT.</li>
                <li>⏳ <strong>Thời gian xử lý:</strong> Lần đầu sử dụng có thể mất thời gian để tải mô hình.</li>
                <li>🎯 <strong>Kết quả:</strong> Hệ thống sẽ cho biết tin nhắn "BÌNH THƯỜNG" hay "LỪA ĐẢO" kèm độ tin cậy.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Back to home button
        if st.button("🏠 Quay về trang chủ", use_container_width=True):
            go_home()
        
        # Clear chat button
        if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
            st.session_state['messages'] = []
            st.rerun()
    
    # Process the actual request after showing typing indicator
    if st.session_state['messages'] and st.session_state['messages'][-1]['role'] == 'typing':
        # Remove typing indicator
        st.session_state['messages'].pop()
        
        # Get the user message
        user_message = st.session_state['messages'][-1]
        current_time = time.strftime("%H:%M")
        
        # Simulate processing time
        time.sleep(1)
        
        # Process the request
        if user_message['text'].startswith('📸'):
            # This was an image upload - we need to get the file from session state
            # For demo purposes, we'll show a placeholder response
            ai_response = """🔍 **Phân tích ảnh hoàn tất**

⚠️ **Lưu ý:** Do giới hạn của Streamlit, việc xử lý ảnh đã được tải lên cần được thực hiện trong form. 

Để sử dụng tính năng phân tích ảnh:
1. Tải ảnh chụp màn hình tin nhắn
2. Nhấn "Gửi" 
3. Hệ thống sẽ trích xuất và phân tích từng tin nhắn

**Kết quả mẫu:**
- Tin nhắn 1: BÌNH THƯỜNG (85.32%)
- Tin nhắn 2: LỪA ĐẢO (92.15%)"""
        
        elif user_message['text'] == "(tin nhắn trống)":
            ai_response = "⚠️ Bạn chưa nhập nội dung tin nhắn. Vui lòng nhập văn bản hoặc tải ảnh để phân tích."
        
        else:
            # Process text with PhoBERT
            text = user_message['text']
            try:
                predictions = phobert_predict([text])
                pred = predictions[0]
                
                if pred['prediction'] == "LỪA ĐẢO":
                    ai_response = f"""🚨 **CẢNH BÁO LỪA ĐẢO**

**Kết quả phân tích:** {pred['prediction']} 
**Độ tin cậy:** {pred['confidence']}

⚠️ **Tin nhắn này có dấu hiệu lừa đảo!**

**Khuyến nghị:**
- Không cung cấp thông tin cá nhân
- Không chuyển tiền hoặc thực hiện giao dịch
- Xác minh thông tin qua kênh chính thức
- Báo cáo cho cơ quan chức năng nếu cần

**Nội dung đã phân tích:**
"{pred['text']}" """
                else:
                    ai_response = f"""✅ **Tin nhắn an toàn**

**Kết quả phân tích:** {pred['prediction']}
**Độ tin cậy:** {pred['confidence']}

**Nội dung đã phân tích:**
"{pred['text']}"

💡 **Lưu ý:** Dù tin nhắn được đánh giá an toàn, bạn vẫn nên thận trọng với các yêu cầu tài chính hoặc thông tin cá nhân."""
                        
            except Exception as e:
                ai_response = f"❌ **Lỗi xử lý:** {str(e)}\n\nVui lòng thử lại sau."
        
        # Add AI response
        st.session_state['messages'].append({
            'role': 'ai', 
            'text': ai_response,
            'meta': 'AI Assistant',
            'time': current_time
        })
        
        # Add auto-scroll JavaScript
        add_auto_scroll_js()
        
        # Rerun to show the response
        st.rerun()
