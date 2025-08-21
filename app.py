# ----------------------- HOME (REWRITTEN) -----------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.stop()

def render_homepage(go_chat_callback=None):
    # Minimal CSS (keeps look & feel but safer than huge inline HTML)
    CSS_HOME = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    :root{
      --primary-blue:#1e40af;
      --accent-blue:#60a5fa;
      --light-blue:#dbeafe;
      --card-radius:14px;
    }
    .hero-box{
      background: linear-gradient(135deg, var(--primary-blue), #12275a);
      color: white;
      padding: 48px;
      border-radius: var(--card-radius);
      box-shadow: 0 10px 30px rgba(2,6,23,0.15);
    }
    .muted { color: rgba(255,255,255,0.85); }
    .section-title { font-size: 28px; font-weight:700; color:#0f172a; }
    .feature-card { background:white; padding:20px; border-radius:12px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
    .trust-num { font-size:28px; font-weight:800; color:var(--primary-blue); }
    .cta-btn { background: white; color: var(--primary-blue); font-weight:700; padding:10px 18px; border-radius:10px; text-decoration:none; }
    </style>
    """
    st.markdown(CSS_HOME, unsafe_allow_html=True)

    # Top nav (simple)
    nav_col1, nav_col2 = st.columns([3,1])
    with nav_col1:
        st.markdown("<div style='display:flex;gap:10px;align-items:center'><i class='fas fa-shield-alt' style='color:var(--primary-blue);font-size:22px'></i><span style='font-weight:700;font-size:18px'>DunkingScam</span></div>", unsafe_allow_html=True)
    with nav_col2:
        if st.button("Đăng nhập", key="top_login"):
            st.info("Tính năng đăng nhập chưa triển khai (demo).")

    st.write("")  # spacer

    # Hero / Intro
    c1, c2 = st.columns([6,5])
    with c1:
        st.markdown("<div class='hero-box'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size:36px;margin:0 0 8px 0'>Bảo vệ bạn khỏi lừa đảo bằng <span style='color:var(--accent-blue)'>AI thông minh</span></h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted' style='margin:0 0 18px 0'>Tải ảnh chụp màn hình hoặc dán nội dung tin nhắn — hệ thống trích xuất và phân loại ngay lập tức.</p>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1,1])
        with btn_col1:
            if st.button("🚀 Bắt đầu dùng thử", key="hero_try"):
                if callable(go_chat_callback):
                    go_chat_callback()
                else:
                    st.session_state['page'] = 'chat'
                    safe_rerun()
        with btn_col2:
            st.markdown("<a class='cta-btn' href='#features'>Tìm hiểu thêm</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        # hero image preview
        st.image("https://img.freepik.com/free-photo/team-developers-doing-brainstorming-optimizing-code_482257-112972.jpg?fit=crop&w=800&h=500", caption="DunkingScam AI", use_column_width=True)

    st.write("")  # spacer

    # Trust indicators (3 numbers)
    st.markdown("---")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"<div style='text-align:center'><div class='trust-num'>99%</div><div>Độ chính xác</div></div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<div style='text-align:center'><div class='trust-num'>2k+</div><div>Tin nhắn trong dữ liệu</div></div>", unsafe_allow_html=True)
    with t3:
        st.markdown(f"<div style='text-align:center'><div class='trust-num'>0.99</div><div>F1-score (ước tính)</div></div>", unsafe_allow_html=True)

    st.write("")  # spacer

    # Features (grid)
    st.markdown("<a id='features'></a>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:16px;margin-bottom:8px'><h2 class='section-title' style='margin:0'>Tính năng</h2></div>", unsafe_allow_html=True)
    fcols = st.columns(4)
    features = [
        ("AI phân loại", "PhoBERT phân tích ngôn ngữ tiếng Việt và phát hiện mẫu lừa đảo"),
        ("OCR tích hợp", "Trích xuất văn bản từ ảnh chụp màn hình (Vintern)"),
        ("Giao diện thân thiện", "Dễ sử dụng cho mọi người, kể cả không chuyên"),
        ("Khuyến nghị an toàn", "Hướng dẫn cụ thể khi phát hiện rủi ro")
    ]
    for col, (title, desc) in zip(fcols, features):
        with col:
            st.markdown(f"<div class='feature-card'><h4 style='margin:0 0 6px 0'>{title}</h4><div style='color:#6b7280'>{desc}</div></div>", unsafe_allow_html=True)

    st.write("")  # spacer

    # How it works (3 steps)
    st.markdown("---")
    st.markdown("<h3 style='margin-bottom:6px'>Cách hoạt động (3 bước)</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("<div style='text-align:center'><div style='background:var(--primary-blue);color:white;width:64px;height:64px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:22px'>1</div><h4 style='margin-top:8px'>Chụp màn hình</h4><div style='color:#6b7280'>Chụp hoặc lưu tin nhắn đáng ngờ</div></div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div style='text-align:center'><div style='background:var(--primary-blue);color:white;width:64px;height:64px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:22px'>2</div><h4 style='margin-top:8px'>Tải ảnh lên</h4><div style='color:#6b7280'>Hệ thống trích xuất văn bản và phân tích</div></div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div style='text-align:center'><div style='background:var(--primary-blue);color:white;width:64px;height:64px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:22px'>3</div><h4 style='margin-top:8px'>Nhận cảnh báo</h4><div style='color:#6b7280'>Cảnh báo lừa đảo kèm hướng xử lý</div></div>", unsafe_allow_html=True)

    st.write("")  # spacer

    # FAQ (compact)
    st.markdown("---")
    st.markdown("<h3 style='margin-bottom:6px'>Mẹo an toàn nhanh</h3>", unsafe_allow_html=True)
    faqs = [
        ("Không chia sẻ OTP", "Đừng cung cấp mã OTP cho bất kỳ ai, kể cả người tự xưng là nhân viên ngân hàng."),
        ("Kiểm tra link trước khi nhấp", "Di chuột để xem URL thật, không click link lạ."),
        ("Xác thực qua kênh chính thức", "Gọi số tổng đài chính thức hoặc truy cập website chính thức.")
    ]
    for q,a in faqs:
        st.markdown(f"**{q}** — <span style='color:#6b7280'>{a}</span>", unsafe_allow_html=True)

    st.write("")  # spacer

    # Final CTA
    st.markdown("<hr>", unsafe_allow_html=True)
    cta_col1, cta_col2 = st.columns([3,1])
    with cta_col1:
        st.markdown("<h2 style='margin:0'>Sẵn sàng bảo vệ bản thân khỏi lừa đảo?</h2><div style='color:#6b7280;margin-top:6px'>Dùng thử công cụ phân tích tin nhắn bằng AI ngay bây giờ.</div>", unsafe_allow_html=True)
    with cta_col2:
        if st.button("🚀 Bắt đầu dùng thử", key="final_cta"):
            if callable(go_chat_callback):
                go_chat_callback()
            else:
                st.session_state['page'] = 'chat'
                safe_rerun()

# call the renderer when on home page
if st.session_state.get('page', 'home') == 'home':
    # if you already have go_chat function (in your file), pass it:
    try:
        render_homepage(go_chat_callback=go_chat)
    except Exception:
        # fallback: call without passing go_chat (it will set session_state itself)
        render_homepage()
# ----------------------- CHAT PAGE (REPLACEMENT) -----------------------
# Make sure FAST_UI_ONLY = False at top of file so models will be loaded.

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.stop()

# Ensure session keys exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'uploaded_file_bytes' not in st.session_state:
    st.session_state['uploaded_file_bytes'] = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None

def save_uploaded_file_to_state(uploaded_file):
    if uploaded_file is None:
        return
    try:
        data = uploaded_file.read()
        st.session_state['uploaded_file_bytes'] = data
        st.session_state['uploaded_file_name'] = uploaded_file.name
    except Exception as e:
        st.error(f"Không thể lưu file: {e}")

def render_chat_page(go_home_callback=None):
    # Header
    st.markdown("""
    <div class="chat-header" style="margin-bottom:18px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(90deg,#1e40af,#1e3a8a);display:flex;align-items:center;justify-content:center;color:white;font-weight:700">AI</div>
            <div>
                <h3 style="margin:0">DunkingScam AI</h3>
                <div style="color: #6b7280; font-size:13px">Phân tích tin nhắn - an toàn và nhanh</div>
            </div>
            <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 6px rgba(16,185,129,0.4)"></div>
                <div style="color:#6b7280;font-size:13px">Online</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 3])

    # LEFT: Chat area
    with col_left:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-messages" id="chatMessages" style="height:520px;overflow-y:auto;padding:18px;background:white;border-radius:12px;">', unsafe_allow_html=True)

        # Render messages
        for msg in st.session_state['messages']:
            role = msg.get('role')
            text = msg.get('text', '')
            meta = msg.get('meta', '')
            time_meta = msg.get('time', '')
            if role == 'ai':
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                  <div style="display:inline-block;background:#f3f4f6;color:#0f172a;padding:12px 14px;border-radius:14px;max-width:85%;">{text.replace(chr(10), '<br>')}</div>
                  <div style="font-size:12px;color:#6b7280;margin-top:6px">{meta} • {time_meta}</div>
                </div>
                """, unsafe_allow_html=True)
            elif role == 'user':
                st.markdown(f"""
                <div style="margin-bottom:12px;text-align:right;">
                  <div style="display:inline-block;background:#1e40af;color:white;padding:12px 14px;border-radius:14px;max-width:85%;">{text}</div>
                  <div style="font-size:12px;color:#6b7280;margin-top:6px">{meta} • {time_meta}</div>
                </div>
                """, unsafe_allow_html=True)
            elif role == 'typing':
                st.markdown("""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                  <div style="background:#f3f4f6;padding:10px;border-radius:14px;display:inline-flex;align-items:center;">
                    <div style="margin-right:8px;color:#6b7280">AI đang suy nghĩ</div>
                    <div style="display:flex;gap:4px">
                      <div style="width:8px;height:8px;border-radius:50%;background:#6b7280;animation:dot 1.2s infinite"></div>
                      <div style="width:8px;height:8px;border-radius:50%;background:#6b7280;animation:dot 1.2s .2s infinite"></div>
                      <div style="width:8px;height:8px;border-radius:50%;background:#6b7280;animation:dot 1.2s .4s infinite"></div>
                    </div>
                  </div>
                </div>
                <style>@keyframes dot{0%{opacity:0.3;transform:translateY(0)}50%{opacity:1;transform:translateY(-6px)}100%{opacity:0.3;transform:translateY(0)}}</style>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close messages

        # Input area
        st.markdown('<div style="background:white;padding:14px;border-radius:12px;margin-top:12px;">', unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "📸 Tải ảnh chụp màn hình (png/jpg)", 
                type=['png', 'jpg', 'jpeg'],
                key='chat_file_uploader',
                help="Tải ảnh để hệ thống trích xuất và phân tích nội dung."
            )
            text_input = st.text_area("💬 Hoặc gõ nội dung tin nhắn để kiểm tra", height=90, key='chat_text_input', placeholder="Dán nội dung hoặc gõ tin nhắn...")
            submit = st.form_submit_button("📤 Gửi", use_container_width=True)

            if submit:
                current_time = time.strftime("%H:%M")
                # Persist uploaded file bytes
                if uploaded_file is not None:
                    save_uploaded_file_to_state(uploaded_file)
                    user_msg = f"📸 {uploaded_file.name}"
                elif text_input and text_input.strip():
                    user_msg = text_input.strip()
                else:
                    user_msg = "(tin nhắn trống)"

                # Append user and typing indicator
                st.session_state['messages'].append({'role': 'user', 'text': user_msg, 'meta': 'Bạn', 'time': current_time})
                st.session_state['messages'].append({'role': 'typing', 'text': '', 'meta': 'AI', 'time': current_time})

                # Immediately rerun to show typing
                safe_rerun()

        st.markdown('</div>', unsafe_allow_html=True)  # close input area
        st.markdown('</div>', unsafe_allow_html=True)  # close container

    # RIGHT: Sidebar
    with col_right:
        st.markdown("""
        <div style="background:white;padding:16px;border-radius:12px;box-shadow:0 6px 18px rgba(2,6,23,0.05);">
          <h4 style="margin:0 0 8px 0">🔍 Hướng dẫn</h4>
          <ul style="padding-left:18px;margin:0;color:#6b7280">
            <li>Gõ hoặc dán nội dung để kiểm tra nhanh.</li>
            <li>Tải ảnh chụp màn hình để trích xuất và phân tích nhiều tin nhắn.</li>
            <li>Tránh tải ảnh chứa thông tin quá nhạy cảm trong môi trường demo.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("🏠 Quay về trang chủ", use_container_width=True):
            if callable(go_home_callback):
                go_home_callback()
            else:
                st.session_state['page'] = 'home'
                safe_rerun()

        if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
            st.session_state['messages'] = []
            st.session_state['uploaded_file_bytes'] = None
            st.session_state['uploaded_file_name'] = None
            safe_rerun()

    # Processing: if last message is typing, handle it
    if st.session_state['messages'] and st.session_state['messages'][-1].get('role') == 'typing':
        # Remove typing indicator
        st.session_state['messages'].pop()
        user_message = st.session_state['messages'][-1]
        current_time = time.strftime("%H:%M")

        # simulate a short processing delay to show indicator
        time.sleep(0.8)

        try:
            # IMAGE case
            if user_message['text'].startswith('📸'):
                if not st.session_state.get('uploaded_file_bytes'):
                    ai_response = "⚠️ Không tìm thấy ảnh trong session. Vui lòng tải lại ảnh và nhấn Gửi."
                else:
                    try:
                        img = Image.open(io.BytesIO(st.session_state['uploaded_file_bytes']))
                    except Exception as e:
                        ai_response = f"❌ Không thể mở ảnh: {e}"
                    else:
                        # Run Vintern OCR (if available)
                        messages_extracted, err = vintern_ocr_extract(img)
                        if err:
                            ai_response = f"⚠️ Vintern OCR không khả dụng: {err}\n\nBạn có thể dán trực tiếp văn bản để kiểm tra."
                        elif not messages_extracted:
                            ai_response = "⚠️ Không trích xuất được tin nhắn từ ảnh. Vui lòng thử ảnh khác hoặc dán văn bản."
                        else:
                            preds = phobert_predict(messages_extracted)
                            lines = ["🔍 **KẾT QUẢ PHÂN TÍCH ẢNH**\n"]
                            for i, p in enumerate(preds, start=1):
                                lines.append(f"- Tin nhắn {i}: {p['prediction']} ({p['confidence']})\n  > {p['text']}")
                            ai_response = "\n".join(lines)

            # EMPTY message
            elif user_message['text'] == "(tin nhắn trống)":
                ai_response = "⚠️ Bạn chưa nhập nội dung. Vui lòng nhập văn bản hoặc tải ảnh để phân tích."

            # TEXT case
            else:
                text = user_message['text']
                preds = phobert_predict([text])
                pred = preds[0]
                if pred['prediction'] == "UNKNOWN":
                    ai_response = "⚠️ Mô hình PhoBERT chưa được tải hoặc không khả dụng. Kiểm tra cấu hình mô hình."
                elif pred['prediction'] == "LỪA ĐẢO":
                    ai_response = (
                        f"🚨 **CẢNH BÁO LỪA ĐẢO**\n\n"
                        f"**Kết quả:** {pred['prediction']}\n"
                        f"**Độ tin cậy:** {pred['confidence']}\n\n"
                        "- Không cung cấp thông tin cá nhân\n- Không chuyển tiền\n- Xác minh bằng kênh chính thức\n\n"
                        f"**Nội dung:**\n\"{pred['text']}\""
                    )
                else:
                    ai_response = (
                        f"✅ **Tin nhắn an toàn**\n\n"
                        f"**Kết quả:** {pred['prediction']}\n"
                        f"**Độ tin cậy:** {pred['confidence']}\n\n"
                        f"**Nội dung:**\n\"{pred['text']}\"\n\n💡 Dù vậy, luôn thận trọng với yêu cầu chuyển tiền hoặc thông tin nhạy cảm."
                    )

        except Exception as e:
            ai_response = f"❌ Lỗi xử lý: {e}"

        # Append AI response and auto-scroll JS
        st.session_state['messages'].append({'role': 'ai', 'text': ai_response, 'meta': 'AI Assistant', 'time': current_time})
        # auto-scroll JS (keeps existing function add_auto_scroll_js)
        try:
            add_auto_scroll_js()
        except Exception:
            pass

        safe_rerun()

# Call the chat renderer when on chat page
if st.session_state.get('page') == 'chat':
    render_chat_page(go_home_callback=go_home)
