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

try:
    Settings.initialize_directories() # Explicit call for safety before RAGChatbot init
    rag_chatbot = RAGChatbot()
    logger.info("RAGChatbot initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAGChatbot: {e}", exc_info=True)
    # Provide a more user-friendly exit or fallback if running in a server environment
    # For local execution, SystemExit is okay.
    raise SystemExit(f"Error initializing chatbot: {e}. Check logs for details.")

def validate_input(prompt: str) -> dict:
    """Validate user input and control send button state.
    
    Args:
        prompt (str): User's input text
        
    Returns:
        dict: Gradio update object for send button interactivity
    """
    is_interactive = bool(prompt and prompt.strip())
    return gr.update(interactive=is_interactive)

def update_available_docs() -> dict:
    """Get list of available processed documents for the dropdown."""
    docs = rag_chatbot.get_available_documents()
    choices = ["Select a document..."] + docs if docs else ["Select a document..."]
    return gr.update(choices=choices, value="Select a document...")

def load_selected_doc(selected: str) -> str:
    """Load a previously processed document."""
    if not selected or selected == "Select a document...":
        return "Please select a document to load"
    return rag_chatbot.load_document(selected)

def handle_processing_start() -> list:
    """Disable buttons when processing starts."""
    return [
        gr.update(interactive=False),  # upload_button
        gr.update(interactive=False),  # load_button
        "Processing document..."  # status
    ]

def handle_processing_end(status_msg: str) -> list:
    """Re-enable buttons when processing ends."""
    return [
        gr.update(interactive=True),  # upload_button
        gr.update(interactive=True),  # load_button
        status_msg  # status
    ]

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="RAG PDF Chatbot") as app:
    # app_state stores the state of the RAG chatbot that needs to persist across interactions
    # It includes which file is loaded and its processed data.
    app_state = gr.State(value=rag_chatbot.get_initial_state()) 

    with gr.Column(elem_classes="main_content_area_container"): # Outer container for potential overall padding/margin
        with gr.Row(equal_height=False, elem_classes="main_columns"): # Main row for sidebar and chat area
            with gr.Column(scale=1, min_width=300): # Sidebar for document processing
                gr.Markdown("## Document Hub")
                with gr.Group():
                    file_input = gr.File(
                        label="Upload PDF Document",
                        file_types=[".pdf"],
                        # scale=1 # scale is not a direct param of gr.File, layout within column handles size
                    )
                    upload_button = gr.Button(
                        "Process Document",
                        variant="primary",
                        # scale=1
                    )
                with gr.Group():
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        lines=3, # Enough for a few lines of status
                        max_lines=5,
                        elem_id="upload_status_box"
                        # scale=1
                    )
                    processed_file_display = gr.Textbox(
                        label="Active Document",
                        value="None", # Initial value
                        interactive=False,
                        elem_id="processed_file_display_box"
                        # scale=1
                    )
                
                gr.Markdown("## Available Documents")
                doc_dropdown = gr.Dropdown(
                    choices=["Select a document..."],
                    value="Select a document...",
                    label="Select Document",
                    interactive=True
                )
                load_button = gr.Button(
                    "Load Document",
                    variant="secondary"
                )

            with gr.Column(scale=3): # Main chat area
                gr.Markdown("# AI Chat Assistant")
                chatbot_ui = gr.Chatbot(
                    label="Chat History",
                    elem_id="chatbot",
                    show_copy_button=True,
                    avatar_images=(None, "https://img.icons8.com/fluency/48/robot.png")
                )
                
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="Type your message here...",
                        elem_id="input-box",
                        autofocus=True,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button(
                        "Send",
                        interactive=False,
                        variant="primary",
                        scale=1
                    )
                    clear_btn = gr.ClearButton(
                        components=[textbox, chatbot_ui],
                        value="Clear Chat",
                        scale=1
                    )

    # --- Event Handler Configuration ---

    # Enable/disable send button based on textbox input
    textbox.input(
        fn=validate_input,
        inputs=[textbox],
        outputs=[submit_btn],
        queue=False # Usually UI updates like this don't need to be queued
    )

    # Update document dropdown on app load
    app.load(
        fn=update_available_docs,
        outputs=[doc_dropdown]
    )

    # Document Upload and Processing Logic
    upload_button.click(
        fn=handle_processing_start,  # Disable buttons first
        outputs=[upload_button, load_button, upload_status]
    ).then(
        fn=rag_chatbot.process_uploaded_file,
        inputs=[file_input],
        outputs=[upload_status]
    ).then(
        fn=handle_processing_end,  # Re-enable buttons with status
        inputs=[upload_status],
        outputs=[upload_button, load_button, upload_status]
    ).then(
        fn=rag_chatbot.update_state_after_upload_or_load,
        inputs=[upload_status, app_state],
        outputs=[app_state]
    ).then(
        fn=lambda current_app_state: current_app_state.get('processed_file_name', "None"),
        inputs=[app_state],
        outputs=[processed_file_display]
    ).then(
        fn=update_available_docs,
        outputs=[doc_dropdown]
    ).then(
        fn=lambda: (None, ""),
        outputs=[chatbot_ui, textbox]
    )
    
    # Document Loading Logic
    load_button.click(
        fn=handle_processing_start,  # Disable buttons first
        outputs=[upload_button, load_button, upload_status]
    ).then(
        fn=load_selected_doc,
        inputs=[doc_dropdown],
        outputs=[upload_status]
    ).then(
        fn=handle_processing_end,  # Re-enable buttons with status
        inputs=[upload_status],
        outputs=[upload_button, load_button, upload_status]
    ).then(
        fn=rag_chatbot.update_state_after_upload_or_load,
        inputs=[upload_status, app_state],
        outputs=[app_state]
    ).then(
        fn=lambda current_app_state: current_app_state.get('processed_file_name', "None"),
        inputs=[app_state],
        outputs=[processed_file_display]
    ).then(
        fn=lambda: (None, ""),
        outputs=[chatbot_ui, textbox]
    )

    # Helper function to clear textbox after message is sent (common UX)
    def clear_textbox_on_send():
        return ""

    # Chat Logic for Send button
    submit_btn.click(
        fn=rag_chatbot.chat_interface_logic,
        inputs=[textbox, chatbot_ui, app_state],
        outputs=[textbox, chatbot_ui, app_state] # Textbox, Chatbot, and State are updated
        # To clear textbox after send:
        # outputs=[gr.Textbox(value=""), chatbot_ui, app_state]
        # or use .then(clear_textbox_on_send, outputs=[textbox])
    ).then(
        fn=clear_textbox_on_send, # Clears the textbox
        inputs=None,
        outputs=[textbox],
        queue=False
    )

    # Chat Logic for Enter key in Textbox
    textbox.submit(
        fn=rag_chatbot.chat_interface_logic,
        inputs=[textbox, chatbot_ui, app_state],
        outputs=[textbox, chatbot_ui, app_state]
    ).then(
        fn=clear_textbox_on_send, # Clears the textbox
        inputs=None,
        outputs=[textbox],
        queue=False
    )

if __name__ == "__main__":
    logger.info(f"Starting RAG PDF Chatbot application with Gradio server name: {Settings.GRADIO_SERVER_NAME}")
    try:
        app.launch(
            server_name=Settings.GRADIO_SERVER_NAME,
            share=Settings.GRADIO_SHARE,
            debug=Settings.GRADIO_DEBUG,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio application: {e}", exc_info=True)
        raise