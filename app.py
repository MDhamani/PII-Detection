import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and tokenizer
model_path = "deberta3base_1024"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, model_max_length=1024)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create NER pipeline
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

# Target labels
target_labels = {
    'EMAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 
    'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME'
}

def detect_pii(text):
    results = ner_pipeline(text)
    
    filtered_entities = [{
        'text': entity['word'],
        'type': entity['entity_group'],
        'start': entity['start'],
        'end': entity['end'],
        'score': float(entity['score'])  # Convert numpy float to Python float for JSON serialization
    } for entity in results if entity['entity_group'] in target_labels]
    
    return filtered_entities

def merge_entities(entities):
    if not entities:
        return []
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    merged = []
    current = None

    for entity in sorted_entities:
        if current and entity['type'] == current['type'] and entity['start'] <= current['end']:
            # Extend the current entity if they overlap or are adjacent
            current['end'] = max(current['end'], entity['end'])
            current['text'] = current['text'] if entity['start'] == current['end'] else current['text'] + entity['text'][current['end']-entity['start']:]
            current['score'] = (current['score'] + entity['score']) / 2  # Average the scores
        else:
            if current:
                merged.append(current)
            current = entity.copy()
    
    if current:
        merged.append(current)
    
    return merged

def highlight_pii(text, entities):
    # Create an HTML string with highlights for each PII entity
    entities = merge_entities(entities)
    
    # Sort entities by start position in reverse to avoid index shifting
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
        
        # Add the highlighted entity with uniform highlight color
        highlight_html = f'<span class="pii-highlight" title="{entity_type}">{entity_text}</span>'
        html_parts.insert(0, highlight_html)
        
        last_end = start
    
    # Add any text before the first entity
    if last_end > 0:
        html_parts.insert(0, text[0:last_end])
    
    return ''.join(html_parts)

def analyze(text):
    if not text.strip():
        return "<p style='color:red;'>Please enter some text to analyze.</p>"
    
    pii = detect_pii(text)
    highlighted_text = highlight_pii(text, pii)
    
    # Return the highlighted text section with a title
    return f"""
    <h4>Highlighted Text:</h4>
    <div class="highlighted-text">{highlighted_text}</div>
    """

# Define custom CSS
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

# Define Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>PII Detection Tool</h1>")
    gr.Markdown("<p>Enter text to detect personally identifiable information (PII). The tool will highlight detected PII entities.</p>")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Input Text", 
            lines=10, 
            placeholder="Enter your text here...", 
            elem_id="textInput"
        )
    
    detect_btn = gr.Button("Detect PII", elem_id="detectButton")
    output = gr.HTML(elem_id="highlightedText")

    detect_btn.click(fn=analyze, inputs=text_input, outputs=output)

if __name__ == "__main__":
    print("Starting PII Detection App...")
    demo.launch()
    print("PII Detection App is running!")