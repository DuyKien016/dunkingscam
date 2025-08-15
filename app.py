import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as T
from flask import Flask, request, jsonify, render_template

# ===== INIT APP =====
app = Flask(__name__)

# ===== LOAD VINTERN 1 LẦN =====
print("Loading VinTern model...")
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
print("VinTern loaded!")

# ===== LOAD PHOBERT 1 LẦN =====
print("Loading PhoBERT model...")
phobert_path = "DuyKien016/phobert-scam-detector"
phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)
phobert_model = AutoModelForSequenceClassification.from_pretrained(phobert_path).eval()
phobert_model = phobert_model.to("cuda" if torch.cuda.is_available() else "cpu")
print("PhoBERT loaded!")

# ===== HÀM XỬ LÝ ẢNH =====
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    max_size = (448, 448)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img = ImageOps.pad(img, max_size, color=(245, 245, 245))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(img).unsqueeze(0).to(vintern_model.device)
    return pixel_values

# ===== HÀM TRÍCH XUẤT TIN NHẮN VINTERN =====
def extract_messages(pixel_values):
    prompt = """<image>
Đọc từng tin nhắn trong ảnh và xuất ra định dạng:

Tin nhắn 1: [nội dung]
Tin nhắn 2: [nội dung]
Tin nhắn 3: [nội dung]

Quy tắc:
- Mỗi ô chat = 1 tin nhắn
- Chỉ lấy nội dung văn bản
- Bỏ thời gian, tên người, emoji
- Đọc từ trên xuống dưới

Bắt đầu:"""
    response, *_ = vintern_model.chat(
        tokenizer=vintern_tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=dict(max_new_tokens=1024, do_sample=False, num_beams=1, early_stopping=True),
        history=None,
        return_history=True
    )
    messages = re.findall(r"Tin nhắn \d+: (.+?)(?=\nTin nhắn|\Z)", response, re.S)
    def quick_clean(msg):
        msg = re.sub(r"\s+", " ", msg.strip())
        msg = re.sub(r'^\d+[\.\)\-\s]+', '', msg)
        return msg.strip()
    return [quick_clean(msg) for msg in messages if msg.strip()]

# ===== HÀM DỰ ĐOÁN PHOBERT =====
def predict_phobert(texts):
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
            "confidence": f"{probs[label]*100:.2f}%"
        })
    return results

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_endpoint():
    if "text" in request.json:
        texts = request.json["text"]
        if isinstance(texts, str): texts = [texts]
        phobert_results = predict_phobert(texts)
        return jsonify({"messages": phobert_results})

    elif "image" in request.files:
        file = request.files["image"]
        file_path = "/tmp/temp_image.png"
        file.save(file_path)
        pixel_values = process_image(file_path)
        messages = extract_messages(pixel_values)
        phobert_results = predict_phobert(messages)
        return jsonify({"messages": phobert_results})

    else:
        return jsonify({"error": "No valid input provided"}), 400

# ===== RUN SERVER =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
