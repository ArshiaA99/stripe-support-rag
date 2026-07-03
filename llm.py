from groq import Groq

def build_prompt(question, context, history_messages):
    history_block = ""
    if history_messages:
        history_block = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_messages])
    else:
        history_block = "No previous conversation history."

    prompt = f"""
You are a helpful customer support assistant. Analyze the user's current input and the chat history, then follow the matching rule strictly:

1. Greetings & Compliments: If the user is greeting you, acknowledging a good point, or giving a compliment (e.g., "Nice", "Good memory", "Cool", "Awesome"), accept it warmly and briefly, then ask how else you can help them. KEEP the conversation open.
2. Explicit Goodbyes & Finality: Trigger this ONLY if the user explicitly says goodbye, states they are completely finished, or rejects further help (e.g., "Goodbye", "No, thanks", "That is all for now"). Respond politely to close the chat (e.g., "Have a great day!") and DO NOT ask any follow-up questions.
3. Factual Queries: If the user is asking a question that requires factual info, act strictly as a retrieval-based assistant. Use ONLY the provided context below. If the answer is not explicitly contained within the context, say exactly: "I don't know based on the provided context." Do not extrapolate or use outside knowledge.

Knowledge Base: {context}

Recent Chat History:
{history_block}

Current User Question: {question}

Answer:
"""
    return prompt

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_llm(prompt):

  response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
          {
              "role": "user",
              "content": prompt
          }
      ]
  )

  return response.choices[0].message.content
