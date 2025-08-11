# 💬 LangChain Chatbot with RAG & Voice  

🚀 **An advanced AI-powered chatbot** built with [LangChain](https://www.langchain.com/), [Groq LLMs](https://groq.com/), Hugging Face embeddings, Google Gemini, and voice capabilities — featuring **Retrieval-Augmented Generation (RAG)**, **Web Search**, **voice input/output**, and **a sleek modern UI**.

[![Live Demo - Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?logo=streamlit)](https://jrf4c9x7689jay35f2bx4v.streamlit.app/)  
[![GitHub Stars](https://img.shields.io/github/stars/123ShubhamVardani/langchain-rag-voice-chatbot?style=social)](https://github.com/123ShubhamVardani/langchain-rag-voice-chatbot)

![Chatbot Demo](assets/demo_screenshot.png)

---

## ✨ Features  

✅ **Multiple AI Models with Auto/Manual Selection**  
- **Groq LLaMA 3.3 70B Versatile** (Fast & high quality)  
- **Groq LLaMA 3.1 8B Instant** (Ultra-fast)  
- **Google Gemini 1.5 Pro**  
- **Hugging Face Qwen/Qwen3-32B**  

✅ **Document Q&A with RAG**  
- Upload PDF, DOCX, TXT, or CSV files.  
- Extracts, chunks, and indexes text with **Hugging Face embeddings** for contextual answers.  

✅ **Web Search Integration**  
- Uses **SerpAPI** to fetch live information when enabled.  

✅ **Voice Input & Output**  
- 🎤 Speak your question.  
- 🔊 Hear AI's response with text-to-speech.  

✅ **Beautiful Chat UI**  
- Modern dark theme.  
- Custom avatars for user & bot.  
- Chat bubble animations.  

✅ **Chat Log Download**  
- Save the entire conversation for reference.  

---

## 🖥️ Tech Stack  

| Layer             | Technology |
|-------------------|------------|
| **Frontend/UI**   | Streamlit |
| **LLMs**          | Groq, Gemini, Hugging Face |
| **RAG**           | FAISS Vector Store + Hugging Face embeddings |
| **Web Search**    | SerpAPI |
| **Voice**         | SpeechRecognition + gTTS |
| **Chunking**      | RecursiveCharacterTextSplitter |

---

## 📸 Screenshots  

### Chatbot Interface  
![Chat UI](assets/demo_screenshot.png)

---

## ⚡ Quick Start  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/123ShubhamVardani/langchain-rag-voice-chatbot.git
cd langchain-rag-voice-chatbot
```

### 2️⃣ Create a virtual environment  
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Add API keys  
Create a `.env` file in the root directory:  
```env
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

### 5️⃣ Run locally  
```bash
streamlit run app.py
```

---

## ☁ Deploy on Streamlit Cloud  

**Live Demo:** [Click Here to Try 🚀](https://jrf4c9x7689jay35f2bx4v.streamlit.app/)  

1. Push your repo to GitHub.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) → **New app** → Connect your repo.  
3. Set environment variables in **Secrets** tab.  
4. Deploy 🚀  

---

## 📜 License  
This project is under a **Custom License**:  
- **No modifications or commercial use** allowed without the author's permission.  

---

## 👨‍💻 Author  
**Shubham Vardani**  
📧 [shub.vardani@gmail.com](mailto:shub.vardani@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/shubham-vardani-325428174) | [GitHub](https://github.com/123ShubhamVardani)

---

⭐ If you like this project, consider giving it a **star** on GitHub!
