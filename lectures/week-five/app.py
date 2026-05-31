"""
app.py — Gradio web UI for the Insurellm RAG assistant.

This is the user-facing layer of the pipeline. It wires the retrieval +
answer logic from implementation/answer.py into a two-column chat interface:
  - left column : the conversation (chatbot + input box)
  - right column: the context chunks that were retrieved for the last answer

Run it (from the week-five folder so the `implementation` package imports):
    .venv/bin/python lectures/week-five/app.py
or:
    uv run --no-project --python .venv/bin/python lectures/week-five/app.py

Adapted from ed-donner/llm_engineering (week5/app.py).
"""

import gradio as gr
from dotenv import load_dotenv

# Reuse the RAG logic we already built — answer_question returns
# (answer_text, context_documents). This import requires app.py to sit
# alongside the implementation/ folder (i.e. in lectures/week-five/).

#from implementation.answer import answer_question

from pro_implementation.answer import answer_question

# Load OPENAI_API_KEY (etc.) from .env.
load_dotenv(override=True)


def format_context(context):
    """
    Render the retrieved documents as HTML for the right-hand panel.

    Each chunk is shown with its source file (in Insurellm orange) followed
    by the chunk text, so you can see exactly what grounded the answer.
    """
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def message_text(content) -> str:
    """
    Flatten a Gradio message's `content` down to a plain text string.

    Gradio 6 normalizes message content into a list of "content part" dicts,
    e.g. [{"type": "text", "text": "Who is Avery?"}], rather than a bare
    string. answer.py expects plain strings, so we collapse that list (or
    pass a string through unchanged) before handing it off.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def chat(history):
    """
    Generate the assistant's reply for the latest user turn.

    `history` is Gradio's message list (already includes the new user
    message as the last item). We split off that last message as the
    question, pass the earlier turns as conversation history, then append
    the assistant's answer and also return the formatted context panel.

    Each message's content is normalized to plain text (see message_text)
    so the downstream RAG code only ever sees strings.
    """
    last_message = message_text(history[-1]["content"])   # the question just asked
    # everything before it = chat history, with content flattened to strings
    prior = [{"role": m["role"], "content": message_text(m["content"])} for m in history[:-1]]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    """Build and launch the Gradio Blocks app."""

    def put_message_in_chatbot(message, history):
        """
        First step on submit: clear the textbox and immediately echo the
        user's message into the chatbot, so they see it before the (slower)
        model call runs in the chained .then() step below.
        """
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    # Gradio 6 moved `theme` out of the Blocks constructor; it's passed to launch() below.
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            # Left: the live conversation.
            with gr.Column(scale=1):
                # Gradio 6 uses the messages format by default and shows a copy
                # button automatically, so `type` and `show_copy_button` are gone.
                chatbot = gr.Chatbot(label="💬 Conversation", height=600)
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            # Right: the retrieved context behind the latest answer.
            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # Two-stage handler on submit:
        #   1) put_message_in_chatbot -> instantly show the user's message
        #   2) .then(chat ...)        -> run RAG and fill in the answer + context
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(theme=theme, inbrowser=True)  # open the local web UI in a browser


if __name__ == "__main__":
    main()
