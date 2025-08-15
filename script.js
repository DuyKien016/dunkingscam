class ChatApp {
  constructor() {
    this.chatMessages = document.getElementById("chatMessages");
    this.messageForm = document.getElementById("messageForm");
    this.messageInput = document.getElementById("messageInput");
    this.sendButton = document.getElementById("sendButton");
    this.fileInput = document.getElementById("fileInput");
    this.attachButton = document.getElementById("attachButton");
    this.typingIndicator = document.getElementById("typingIndicator");
    this.imagePreview = document.getElementById("imagePreview");
    this.previewImage = document.getElementById("previewImage");
    this.removeImageButton = document.getElementById("removeImage");

    this.currentImageFile = null;
    this.BACKEND_URL = "/process"; // Flask route

    this.init();
  }

  init() {
    this.bindEvents();
    this.adjustTextareaHeight();
  }

  bindEvents() {
    this.messageForm.addEventListener("submit", (e) => this.handleSubmit(e));
    this.messageInput.addEventListener("input", () => this.adjustTextareaHeight());
    this.attachButton.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => this.handleFileSelect(e));
    this.removeImageButton.addEventListener("click", () => this.removeImage());
    this.messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSubmit(e);
      }
    });
  }

  adjustTextareaHeight() {
    const textarea = this.messageInput;
    textarea.style.height = "auto";
    const maxHeight = 128;
    textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + "px";

    const hasText = textarea.value.trim().length > 0;
    const hasImage = this.currentImageFile !== null;
    this.sendButton.disabled = !hasText && !hasImage;

    if (hasText || hasImage) {
      this.sendButton.classList.remove("opacity-50", "cursor-not-allowed");
    } else {
      this.sendButton.classList.add("opacity-50", "cursor-not-allowed");
    }
  }

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      this.currentImageFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        this.previewImage.src = e.target.result;
        this.imagePreview.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
      this.adjustTextareaHeight();
    }
  }

  removeImage() {
    this.currentImageFile = null;
    this.imagePreview.classList.add("hidden");
    this.fileInput.value = "";
    this.adjustTextareaHeight();
  }

  async handleSubmit(e) {
    e.preventDefault();
    const messageText = this.messageInput.value.trim();
    const hasImage = this.currentImageFile !== null;
    if (!messageText && !hasImage) return;

    const userMessage = {
      type: "user",
      text: messageText,
      image: hasImage ? this.currentImageFile : null,
      timestamp: new Date(),
    };
    this.addMessage(userMessage);
    this.showTypingIndicator();

    try {
      const formData = new FormData();
      if (messageText) formData.append("text", messageText);
      if (hasImage) formData.append("image", this.currentImageFile);

      const res = await fetch(this.BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      this.hideTypingIndicator();

      if (!res.ok) {
        const errData = await res.json();
        this.addMessage({
          type: "bot",
          text: `❌ Lỗi: ${errData.error || "Không thể xử lý"}`,
          timestamp: new Date(),
        });
        return;
      }

      const data = await res.json();
      let botReply = "";
      if (data.messages && data.messages.length > 0) {
        botReply = data.messages
          .map((r) => `[${r.prediction}] (${r.confidence}) → ${r.text}`)
          .join("\n");
      } else {
        botReply = "Không có kết quả.";
      }

      this.addMessage({
        type: "bot",
        text: botReply,
        timestamp: new Date(),
      });

      this.messageInput.value = "";
      this.removeImage();
      this.adjustTextareaHeight();
    } catch (err) {
      this.hideTypingIndicator();
      this.addMessage({
        type: "bot",
        text: `❌ Lỗi kết nối server: ${err.message}`,
        timestamp: new Date(),
      });
    }
  }

  addMessage(message) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `flex ${
      message.type === "user" ? "justify-end" : "justify-start"
    } mb-4 message-appear`;

    const isUser = message.type === "user";
    const bubbleClass = isUser
      ? "bg-gradient-to-r from-secondary-blue to-primary-blue text-white rounded-lg rounded-tr-none"
      : "bg-gray-100 text-gray-800 rounded-lg rounded-tl-none";

    let imageContent = "";
    if (message.image) {
      const imageUrl = URL.createObjectURL(message.image);
      imageContent = `<div class="image-message mb-2"><img src="${imageUrl}" alt="Uploaded image" class="rounded-lg shadow-sm"></div>`;
    }

    const textContent = message.text ? `<p class="text-sm">${this.escapeHtml(message.text)}</p>` : "";

    messageDiv.innerHTML = `
      <div class="max-w-xs lg:max-w-md">
        <div class="${bubbleClass} px-4 py-2 shadow-sm">
          ${imageContent}
          ${textContent}
        </div>
        <div class="flex items-center mt-1 text-xs text-gray-500 ${isUser ? "justify-end" : ""}">
          ${!isUser ? '<div class="w-4 h-4 bg-secondary-blue rounded-full mr-2"></div>' : ""}
          <span>${isUser ? "You" : "AI Assistant"} • ${this.formatTime(message.timestamp)}</span>
          ${isUser ? '<div class="w-4 h-4 bg-secondary-blue rounded-full ml-2"></div>' : ""}
        </div>
      </div>
    `;

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();
  }

  showTypingIndicator() {
    this.typingIndicator.classList.remove("hidden");
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.typingIndicator.classList.add("hidden");
  }

  scrollToBottom() {
    setTimeout(() => {
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }, 100);
  }

  formatTime(date) {
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    if (diffInMinutes < 1) return "now";
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 24 * 60) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
}

document.addEventListener("DOMContentLoaded", () => new ChatApp());

window.addEventListener("load", () => {
  document.body.style.opacity = "0";
  document.body.style.transition = "opacity 0.3s ease";
  requestAnimationFrame(() => { document.body.style.opacity = "1"; });
});
