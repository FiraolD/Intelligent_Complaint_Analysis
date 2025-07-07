import os
import gradio as gr
from src.rag_pipeline import RAGPipeline


# Initialize RAG pipeline
rag = RAGPipeline()
print("[INFO] RAG Pipeline loaded successfully")


def respond(message, chat_history):
    """
    Process user message through RAG and return answer with sources
    """
    result = rag.generate_answer(message)

    answer = result["answer"]
    sources = result["retrieved_chunks"][:2]

    # Format sources into readable strings
    formatted_sources = "\n\n".join([f"{i+1}. {chunk[:500]}..." for i, chunk in enumerate(sources)])

    # Update chat history in tuples format
    chat_history.append((message, answer))

    return chat_history, formatted_sources, chat_history  # Match: chatbot, sources_box, state


# Define Gradio Chat Interface
with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:
    gr.Markdown("# 🧾 CrediTrust Financial - Intelligent Complaint Analysis")
    gr.Markdown("Ask any question about customer complaints and get evidence-backed responses.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="💬 Ask a Question", placeholder="e.g., 'What issues do customers face with BNPL services?'")
            submit_btn = gr.Button("❓ Ask / Submit")
            clear = gr.Button("🗑️ Clear")

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Source Documents Used")
            sources_box = gr.Textbox(label="Source Chunks", lines=15, max_lines=15, interactive=False)

    state = gr.State([])

    # Bind button click and enter key
    submit_btn.click(
        fn=respond,
        inputs=[msg, state],
        outputs=[chatbot, sources_box, state]
    )
    
    # Optional: Also allow pressing Enter
    msg.submit(respond,[msg, state],[chatbot, sources_box, state])
    clear.click(lambda: ([], ""), None, [chatbot, sources_box], queue=False)

if __name__ == "__main__":
    print("[INFO] Launching Gradio App...")
    demo.launch(share=True)