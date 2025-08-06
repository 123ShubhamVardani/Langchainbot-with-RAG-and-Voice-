
# 🧠 LangChain RAG Chatbot with Voice (Part 2)

A conversational AI assistant powered by:
- **LangChain** + **Groq** models (Auto/manual fallback)
- **RAG** (Retrieval-Augmented Generation) with FAISS
- **Web Search** using SerpAPI
- **Voice Input** & **Text-to-Speech** with Google APIs

---

## 🚀 Features

✅ Multi-model fallback using Groq LLMs  
✅ RAG-based document Q&A (PDF, DOCX, TXT, CSV)  
✅ Voice input via mic and audio playback  
✅ Web search integration using SerpAPI  
✅ Chat history logging and download support  
✅ Responsive UI with avatars and theme styling

---

## 🧰 Tech Stack

- `LangChain`, `Groq`, `OpenAI`, `FAISS`
- `Streamlit` for UI
- `gTTS`, `speech_recognition` for audio
- `SerpAPI` for external web search
- `Python 3.10+`

---

## 🛠️ Setup Instructions

### 1. 📦 Clone the Repo
```bash
git clone https://github.com/your-username/langchain-rag-voice-chatbot.git
cd langchain-rag-voice-chatbot
```

### 2. 🧪 Create Virtual Environment
```bash
uv venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. 📥 Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 4. 🔐 Setup Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_key
```
> 🔐 Never commit your actual `.env` file to GitHub. Use `.env.example` for sharing structure.

### 5. 🧠 Run the App
```bash
streamlit run app.py
```

---

## 📁 File Structure
```
├── app.py               # Main Streamlit app
├── main.py              # Entry script (optional)
├── requirements.txt     # Full dependency list
├── pyproject.toml       # Project metadata
├── .env.example         # API key placeholders
├── .gitignore           # Ignores env and chat logs
├── README.md            # You are here
```

---

## 🌐 Deployment
✅ The app is fully ready for [Streamlit Cloud](https://streamlit.io/cloud). Just push to GitHub and connect it to your Streamlit Cloud account.

---

## 👤 Author
**Shubham Vardani**  
🔗 [GitHub](https://github.com/ShubhamVardani) ・ [LinkedIn](https://linkedin.com/in/shubhamvardani) ・ [Email](mailto:shubhamvardani@gmail.com)

---

## 📌 Next Phase (Part 3)
- Voice Assistant interface
- QR-based public platform access
- Deployment to multiple platforms

Stay tuned for more! 🚀
