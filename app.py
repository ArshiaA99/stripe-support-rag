import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag import retrieve_context
from llm import build_prompt, ask_llm

app = FastAPI(title="Stripe Support AI Assistant")

app.mount("/static", StaticFiles(directory="."), name="static")

class ChatRequest(BaseModel):
    message: str

MAX_MEMORY_TURNS = 6  # 6 messages = 3 user questions + 3 bot answers
chat_history = []

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    API Route that takes a user query, runs it through the RAG pipeline,
    and returns the LLM's response alongside source files.
    """
    global chat_history
    try:
        user_question = payload.message.strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        context = retrieve_context(user_question)
        
        prompt = build_prompt(user_question, context, chat_history)
        
        answer = ask_llm(prompt)
        
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": answer})

        chat_history = chat_history[-MAX_MEMORY_TURNS:]

        return {
            "answer": answer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stripe Support AI</title>
        <style>
            :root {
                --stripe-blurple: #635bff;
                --stripe-dark: #0a2540;
                --stripe-light: #f6f9fc;
                --stripe-border: #e6ebf1;
                --text-muted: #4f5b66;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            body {
                background-color: var(--stripe-light);
                color: var(--stripe-dark);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                padding: 20px;
            }

            .chat-container {
                width: 100%;
                max-width: 800px;
                height: 85vh;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 50px 100px -20px rgba(50, 50, 93, 0.12), 
                            0 30px 60px -30px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                border: 1px solid var(--stripe-border);
            }

            /* Header styling mimicking Stripe Nav/Dashboard */
            .chat-header {
                padding: 20px 24px;
                border-bottom: 1px solid var(--stripe-border);
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .chat-header img {
                height: 40px;
                width: auto;
                object-fit: contain;
                vertical-align: middle;
            }

            .chat-header h1 {
                font-size: 18px;
                font-weight: 600;
                color: var(--stripe-dark);
            }

            /* Chat Content Body */
            .chat-messages {
                flex: 1;
                padding: 24px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
                background: #fafafa;
            }

            .message-wrapper {
                display: flex;
                flex-direction: column;
                max-width: 80%;
            }

            .message-wrapper.user {
                align-self: flex-end;
            }

            .message-wrapper.bot {
                align-self: flex-start;
            }

            .message-bubble {
                padding: 12px 16px;
                border-radius: 12px;
                font-size: 15px;
                line-height: 1.5;
                word-wrap: break-word;
            }

            .user .message-bubble {
                background-color: var(--stripe-blurple);
                color: #ffffff;
                border-bottom-right-radius: 2px;
            }

            .bot .message-bubble {
                background-color: #ffffff;
                color: var(--stripe-dark);
                border-bottom-left-radius: 2px;
                border: 1px solid var(--stripe-border);
                box-shadow: 0 2px 5px rgba(0,0,0,0.03);
            }

            .sources {
                font-size: 11px;
                color: var(--text-muted);
                margin-top: 6px;
                padding: 4px 8px;
                background: #f1f3f5;
                border-radius: 4px;
                display: inline-block;
                width: fit-content;
            }

            /* Loading Indicator */
            .typing-indicator {
                display: flex;
                gap: 4px;
                padding: 8px;
            }
            .typing-indicator span {
                width: 8px;
                height: 8px;
                background: #a3acb9;
                border-radius: 50%;
                animation: bounce 1.3s infinite both;
            }
            .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }

            /* Footer Input Box */
            .chat-footer {
                padding: 16px 24px;
                background: #ffffff;
                border-top: 1px solid var(--stripe-border);
                display: flex;
                gap: 12px;
            }

            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                font-size: 15px;
                outline: none;
                transition: border-color 0.2s, box-shadow 0.2s;
            }

            .chat-input:focus {
                border-color: var(--stripe-blurple);
                box-shadow: 0 0 0 3px rgba(99, 91, 255, 0.15);
            }

            .send-btn {
                background-color: var(--stripe-blurple);
                color: white;
                border: none;
                padding: 0 24px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                cursor: pointer;
                transition: background-color 0.2s;
            }

            .send-btn:hover {
                background-color: #534bc7;
            }

            .send-btn:disabled {
                background-color: #b3b0eb;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>

    <div class="chat-container">
        <div class="chat-header">
            <img src="/static/logo.png" alt="Stripe Logo" onerror="this.style.display='none'">
            <h1>AI Support Assistant</h1>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message-wrapper bot">
                <div class="message-bubble">
                    Hello! I am your Stripe support assistant. Ask me anything based on our documentation.
                </div>
            </div>
        </div>

        <div class="chat-footer">
            <input type="text" id="userInput" class="chat-input" placeholder="Type your question here..." autocomplete="off">
            <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        // Allow entering text via 'Enter' key
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !sendBtn.disabled) {
                sendMessage();
            }
        });

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            // Clear input and disable elements during execution
            userInput.value = '';
            userInput.disabled = true;
            sendBtn.disabled = true;

            // Append User Message to UI
            appendMessage(text, 'user');

            // Append Loading Indicator
            const loadingId = appendLoadingIndicator();
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });

                const data = await response.json();
                
                // Remove Loading Indicator
                document.getElementById(loadingId).remove();

                if (response.ok) {
                    appendMessage(data.answer, 'bot');
                } else {
                    appendMessage("Error: " + (data.detail || "Unable to fetch answer."), 'bot');
                }

            } catch (error) {
                document.getElementById(loadingId).remove();
                appendMessage("Network error. Please make sure the backend is running.", 'bot');
            }

            // Re-enable elements
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function appendMessage(text, sender) {
            const wrapper = document.createElement('div');
            wrapper.classList.add('message-wrapper', sender);

            const bubble = document.createElement('div');
            bubble.classList.add('message-bubble');
            bubble.innerText = text;
            wrapper.appendChild(bubble);

            chatMessages.appendChild(wrapper);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function appendLoadingIndicator() {
            const id = 'loading-' + Date.now();
            const wrapper = document.createElement('div');
            wrapper.classList.add('message-wrapper', 'bot');
            wrapper.id = id;

            const bubble = document.createElement('div');
            bubble.classList.add('message-bubble');

            const indicator = document.createElement('div');
            indicator.classList.add('typing-indicator');
            indicator.innerHTML = '<span></span><span></span><span></span>';
            
            bubble.appendChild(indicator);
            wrapper.appendChild(bubble);
            chatMessages.appendChild(wrapper);
            return id;
        }
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)