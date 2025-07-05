"""
evaluate_rag.py

Evaluates the RAG system by running test questions,
collects model outputs, source chunks, and manual ratings,
and saves results to Evaluation_table.md
"""

import json
from datetime import datetime
from rag_pipeline import RAGPipeline


def run_evaluation():
    # Initialize RAG pipeline
    print("[INFO] Initializing RAG pipeline with TinyLlama...")
    rag = RAGPipeline()

    # Define test questions
    test_questions = [
        "Why are people unhappy with BNPL services?",
        "Are there any fraud-related complaints in money transfers?",
        "What issues do customers face with savings accounts?",
        "Do customers complain about late fees on credit cards?",
        "How do users feel about poor customer service?",
        "Are there any complaints about hidden fees in personal loans?",
        "What are common issues with interest charges on credit cards?",
        "How do users perceive debt collection practices?"
    ]

    evaluation_results = []

    print("\n🚀 Starting evaluation...\n")
    for question in test_questions:
        print("-" * 80)
        print("🔍 Question:", question)

        result = rag.generate_answer(question)

        print("📄 Retrieved Chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"][:2]):
            print(f"{i+1}. {chunk[:200]}...")

        answer = result["answer"]
        print("🤖 Generated Answer:", answer)

        rating = input("Enter quality score (1–5): ")
        comment = input("Any observations or comments? ")

        evaluation_results.append({
            "Question": question,
            "Generated Answer": answer,
            "Retrieved Sources (1–2)": [f"{i+1}. {chunk[:200]}..." for i, chunk in enumerate(result["retrieved_chunks"][:2])],
            "Quality Score (1–5)": rating,
            "Comments/Analysis": comment
        })

    # Save results to file
    save_to_markdown(evaluation_results)

    print("\n✅ Evaluation completed!")
    print(f"📊 Results saved to Evaluation_table.md")


def save_to_markdown(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = "# 🧪 RAG System Evaluation Results\n"
    md_content += f"**Generated on:** {timestamp}\n\n"
    md_content += "| Question | Generated Answer | Retrieved Sources (1–2) | Quality Score (1–5) | Comments / Analysis |\n"
    md_content += "|----------|------------------|--------------------------|--------------------|----------------------|\n"

    for entry in results:
        question = entry['Question'].replace('|', '\\|')
        answer = entry['Generated Answer'].replace('|', '\\|')
        sources = "<br>".join(entry['Retrieved Sources (1–2)'])
        score = entry['Quality Score (1–5)']
        comment = entry['Comments/Analysis'].replace('|', '\\|')

        md_content += f"| {question} | {answer} | {sources} | {score} | {comment} |\n"

    with open("Evaluation_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    run_evaluation()