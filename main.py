"""
RAG PDF Chatbot Web Interface

This is the main entry point for the RAG PDF Chatbot application. It sets up
a Gradio-based web interface that allows users to:
1. Upload and process PDF documents
2. Chat with an AI about the document's contents
3. View processing status and chat history

The interface is designed to be user-friendly and responsive, with proper
error handling and state management.
"""

import gradio as gr
from rag.rag_logic import RAGChatbot
from rag.settings import Settings
from rag.utils import logger

# Initialize the RAG chatbot system
try:
    rag_chatbot = RAGChatbot()
except Exception as e:
    logger.error(f"Failed to initialize RAGChatbot: {e}")
    raise SystemExit(f"Error initializing chatbot: {e}")

def validate_input(prompt: str) -> dict:
    """Validate user input and control send button state.
    
    Args:
        prompt (str): User's input text
        
    Returns:
        dict: Gradio update object for send button interactivity
    """
    return gr.update(interactive=bool(prompt and prompt.strip()))

# Custom CSS for better UI appearance
CSS = """
#component-0 { height: 100vh !important; max-width: 100vw !important; padding: 0 !important; margin: 0 !important; }
.gradio-container { height: 100vh !important; }
.gap { padding: 0 !important; }
#chatbot { height: 70vh !important; }
#input-box textarea { padding: 10px !important; }
"""

# Build the Gradio interface
with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="RAG PDF Chatbot") as app:
    # Initialize application state
    app_state = gr.State(value=rag_chatbot.get_initial_state()) 

    with gr.Row(elem_classes="container"):
        # Document Processing Section
        with gr.Column(scale=1):
            gr.Markdown("## Document Processing")
            file_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                scale=1
            )
            upload_button = gr.Button(
                "Process Document",
                variant="primary",
                scale=1
            )
            upload_status = gr.Textbox(
                label="Processing Status",
                interactive=False,
                lines=3,
                scale=1
            )
            processed_file_display = gr.Textbox(
                label="Currently Processed File",
                value="None",
                interactive=False,
                scale=1
            )

        # Chat Interface Section
        with gr.Column(scale=3):
            gr.Markdown("# AI Chat Assistant")
            chatbot_ui = gr.Chatbot(
                label="Chat History",
                elem_id="chatbot",
                container=True,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                textbox = gr.Textbox(
                    placeholder="Type your message here...",
                    elem_id="input-box",
                    autofocus=True,
                    container=False,
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button(
                    "Send",
                    interactive=False,
                    elem_id="submit_btn",
                    variant="primary",
                    scale=1
                )
                clear_btn = gr.ClearButton(
                    components=[textbox, chatbot_ui],
                    value="Clear Chat",
                    scale=1
                )

    # Event Handler Configuration
    textbox.input(
        validate_input,
        inputs=[textbox],
        outputs=[submit_btn]
    )

    upload_button.click(
        fn=rag_chatbot.process_uploaded_file,
        inputs=[file_input],
        outputs=[upload_status]
    ).then(
        fn=rag_chatbot.update_state_after_upload,
        inputs=[upload_status, app_state],
        outputs=[app_state]
    ).then(
        fn=lambda state: state.get('processed_file_name', "None"),
        inputs=[app_state],
        outputs=[processed_file_display]
    )

    submit_btn.click(
        fn=rag_chatbot.chat_interface_logic,
        inputs=[textbox, chatbot_ui, app_state],
        outputs=[textbox, chatbot_ui, app_state]
    )
    textbox.submit(
        fn=rag_chatbot.chat_interface_logic,
        inputs=[textbox, chatbot_ui, app_state],
        outputs=[textbox, chatbot_ui, app_state]
    )

if __name__ == "__main__":
    logger.info("Starting RAG PDF Chatbot application")
    try:
        app.launch(
            server_name=Settings.GRADIO_SERVER_NAME,
            share=Settings.GRADIO_SHARE,
            debug=Settings.GRADIO_DEBUG,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise