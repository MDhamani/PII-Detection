import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the pre-trained model and tokenizer
model_path = "deberta3base_1024"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, model_max_length=1024)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create a Named Entity Recognition (NER) pipeline using the loaded model and tokenizer
# The pipeline uses GPU if available, otherwise it defaults to CPU
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # Combine tokens into entities
    device=0 if torch.cuda.is_available() else -1
)

# Define the target labels for PII detection
# These labels represent the types of entities to be identified as PII
target_labels = {
    'EMAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 
    'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME'
}

# Function to detect PII entities in the input text
def detect_pii(text):
    # Use the NER pipeline to extract entities from the text
    results = ner_pipeline(text)
    
    # Filter entities to include only those matching the target labels
    filtered_entities = [{
        'text': entity['word'],  # Extracted text
        'type': entity['entity_group'],  # Entity type
        'start': entity['start'],  # Start position in the text
        'end': entity['end'],  # End position in the text
        'score': float(entity['score'])  # Confidence score (converted to Python float)
    } for entity in results if entity['entity_group'] in target_labels]
    
    return filtered_entities

# Function to merge overlapping or adjacent entities of the same type
def merge_entities(entities):
    if not entities:
        return []
    
    # Sort entities by their start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    merged = []
    current = None

    for entity in sorted_entities:
        # Check if the current entity overlaps or is adjacent to the previous one
        if current and entity['type'] == current['type'] and entity['start'] <= current['end']:
            # Extend the current entity's end position and update its text
            current['end'] = max(current['end'], entity['end'])
            current['text'] = current['text'] if entity['start'] == current['end'] else current['text'] + entity['text'][current['end']-entity['start']:]
            # Average the confidence scores
            current['score'] = (current['score'] + entity['score']) / 2
        else:
            # Add the current entity to the merged list and start a new one
            if current:
                merged.append(current)
            current = entity.copy()
    
    # Add the last entity to the merged list
    if current:
        merged.append(current)
    
    return merged

# Function to highlight PII entities in the input text
def highlight_pii(text, entities):
    # Merge overlapping or adjacent entities
    entities = merge_entities(entities)
    
    # Sort entities by start position in reverse order to avoid index shifting
    entities.sort(key=lambda x: x['start'], reverse=True)
    
    html_parts = []
    last_end = len(text)
    
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_type = entity['type']
        entity_text = text[start:end]
        
        # Add text after this entity to the next one
        if end < last_end:
            html_parts.insert(0, text[end:last_end])
        
        # Add the highlighted entity with a tooltip showing its type
        highlight_html = f'<span class="pii-highlight" title="{entity_type}">{entity_text}</span>'
        html_parts.insert(0, highlight_html)
        
        last_end = start
    
    # Add any text before the first entity
    if last_end > 0:
        html_parts.insert(0, text[0:last_end])
    
    return ''.join(html_parts)

# Function to analyze the input text and return highlighted PII entities
def analyze(text):
    if not text.strip():
        # Return an error message if the input text is empty
        return "<p style='color:red;'>Please enter some text to analyze.</p>"
    
    # Detect PII entities in the text
    pii = detect_pii(text)
    # Highlight the detected entities in the text
    highlighted_text = highlight_pii(text, pii)
    
    # Return the highlighted text wrapped in HTML
    return f"""
    <h4>Highlighted Text:</h4>
    <div class="highlighted-text">{highlighted_text}</div>
    """

# Define custom CSS for styling the Gradio interface
custom_css = """
.highlighted-text {
    padding: 1em;
    border: 1px solid #ccc;
    background-color: #2a2a2a;  /* Dark background to match your UI */
    color: #e0e0e0;  /* Light text color for better readability */
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: monospace;
    line-height: 1.5;
}

.pii-highlight {
    background-color: yellow;  /* Use same yellow highlight for all PII */
    color: black;
    font-weight: bold;
    padding: 0 3px;
    border-radius: 3px;
    cursor: help;
}
"""

# Define the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    # Add a title and description
    gr.Markdown("<h1>PII Detection Tool</h1>")
    gr.Markdown("<p>Enter text to detect personally identifiable information (PII). The tool will highlight detected PII entities.</p>")
    
    # Create a row with a text input box and a button
    with gr.Row():
        text_input = gr.Textbox(
            label="Input Text", 
            lines=10, 
            placeholder="Enter your text here...", 
            elem_id="textInput"
        )
    
    detect_btn = gr.Button("Detect PII", elem_id="detectButton")
    output = gr.HTML(elem_id="highlightedText")

    # Link the button to the analyze function
    detect_btn.click(fn=analyze, inputs=text_input, outputs=output)

# Main entry point for the application
if __name__ == "__main__":
    print("Starting PII Detection App...")
    demo.launch()  # Launch the Gradio app
    print("PII Detection App is running!")