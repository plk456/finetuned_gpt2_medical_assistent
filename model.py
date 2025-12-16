import gradio as gr
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline
)
from peft import PeftModel
import pymongo
from datetime import datetime
import uuid
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. MongoDB Integration
# ==========================================
class ChatLogger:
    def __init__(self):
        print("🔌 Initializing MongoDB connection...")
        self.connected = False
        
        # Fetch credentials from environment variables
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        
        if not username or not password:
            print("⚠️ MongoDB credentials not found in .env file. Running in offline mode.")
            return

        try:
            # URL encode the password to handle special characters
            password_encoded = quote_plus(password)
            mongo_uri = f"mongodb+srv://{username}:{password_encoded}@plk.3vytot0.mongodb.net/?appName=plk"
            
            # Connect with a 5-second timeout
            self.client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test the connection
            self.client.admin.command('ping')
            
            self.db = self.client["chatbot_db"]
            self.collection = self.db["gradio_logs"]
            self.connected = True
            print("✅ MongoDB Connected Successfully!")
            
        except Exception as e:
            print(f"❌ MongoDB Connection Error: {e}")
            self.connected = False

    def log_conversation(self, session_id, instruction, input_data, bot_res, status="success"):
        """Save the conversation to the database"""
        if self.connected:
            try:
                self.collection.insert_one({
                    "session_id": session_id,
                    "instruction": instruction,
                    "input": input_data,
                    "bot": bot_res,
                    "status": status,
                    "timestamp": datetime.utcnow()
                })
            except Exception as e:
                print(f"⚠️ Failed to save log: {e}")

    def get_history(self, session_id):
        """Fetch recent chat history for the sidebar"""
        if not self.connected:
            return "⚠️ History unavailable (Offline)"
            
        # Get last 10 messages
        cursor = self.collection.find({"session_id": session_id}).sort("timestamp", -1).limit(10)
        
        history_items = []
        for doc in cursor:
            # Create a label: Instruction (Input)
            label = f"📝 **{doc['instruction'][:40]}...**"
            if doc.get('input'):
                label += f"\n   *({doc['input'][:30]}...)*"
            
            # Add a status icon if it was refused
            if doc.get('status') == "refused":
                label = "⛔ [Blocked] " + label
                
            history_items.append(label)
            
        if not history_items:
            return "No messages yet."
            
        return "\n\n---\n\n".join(history_items)

# Initialize Logger
chat_logger = ChatLogger()
session_id = f"session_{uuid.uuid4().hex[:8]}"

# ==========================================
# 2. Model & Guardrail Setup
# ==========================================
print("\n🤖 Loading AI Models... (This may take a moment)")

# A. Guardrail (The "Traffic Cop")
# Using a lightweight classifier to check if the topic is medical
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_medical_query(text):
    """Returns True if the input is medical, False if random topic."""
    labels = ["medical question", "health advice", "random conversation", "politics", "sports", "technology"]
    
    try:
        result = classifier(text, labels)
        top_label = result['labels'][0]
        score = result['scores'][0]
        
        print(f"🔍 Classification: '{top_label}' (Confidence: {score:.2f})")
        
        # Allow medical topics OR generic greetings (to be polite)
        if top_label in ["medical question", "health advice"]:
            return True
        if top_label == "random conversation" and score < 0.3:
             # If it's ambiguous, give benefit of doubt
            return True
            
        return False
    except Exception as e:
        print(f"⚠️ Classifier error: {e}")
        return True # Fail open (allow query) if classifier breaks

# B. Main Medical Model (GPT-2 + LoRA)
model_name = "gpt2-large"
adapter_path = "./qlora_adapter"

# Check Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using Device: {device.upper()}")

# Load Base Model
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
else:
    # CPU Fallback
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load Adapter
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("✅ Fine-tuned Adapter Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading adapter: {e}")
    model = base_model # Fallback to base model if adapter fails

# ==========================================
# 3. Main Logic
# ==========================================
def predict(instr, inp, chat_history):
    # 1. Check Guardrail
    full_text = f"{instr} {inp}".strip()
    
    if not is_medical_query(full_text):
        refusal_msg = "🚫 I am a specialized Medical AI. I cannot answer questions about general topics, sports, or technology. Please ask a health-related question."
        
        # Log refusal
        chat_logger.log_conversation(session_id, instr, inp, "REFUSED", status="refused")
        
        # Update UI
        user_display = f"**Instruction:** {instr}"
        if inp.strip(): user_display += f"\n**Input:** {inp}"
        chat_history.append((user_display, refusal_msg))
        
        return "", "", chat_history, chat_logger.get_history(session_id)

    # 2. Prepare Prompt (Alpaca Format)
    prompt = f"Instruction: {instr}\n"
    if inp.strip():
        prompt += f"Input: {inp}\n"
    prompt += "Response: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. Generate Response
    print("⏳ Generating...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.1,    # Low temp for accuracy
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the AI's response (remove the prompt)
    response_text = full_response[len(prompt):].strip()
    
    # 4. Log and Update UI
    chat_logger.log_conversation(session_id, instr, inp, response_text, status="success")
    
    user_display = f"**Instruction:** {instr}"
    if inp.strip(): user_display += f"\n**Input:** {inp}"
    
    chat_history.append((user_display, response_text))
    
    return "", "", chat_history, chat_logger.get_history(session_id)

# ==========================================
# 4. Gradio Interface
# ==========================================
custom_css = """
#sidebar_col { background-color: #f7f7f7; padding: 20px; border-right: 1px solid #ddd; }
#chat_header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, fill_height=True) as demo:
    
    with gr.Row(elem_id="main_row"):
        # --- Sidebar (History) ---
        with gr.Column(scale=1, elem_id="sidebar_col"):
            gr.Markdown("### 🗄️ History")
            history_display = gr.Markdown("*Session history will appear here...*")
            gr.HTML("<hr>")
            
            # Status Indicator
            conn_status = "🟢 Online" if chat_logger.connected else "🔴 Offline"
            gr.Markdown(f"**DB Status:** {conn_status}")
            gr.Markdown(f"**Session:** `{session_id[:8]}...`")

        # --- Main Chat Area ---
        with gr.Column(scale=4):
            gr.HTML("""
                <div id="chat_header">
                    <h1 style='color: #2b5c2e;'>🏥 Medical Assistant AI</h1>
                    <p>Powered by Fine-Tuned GPT-2 • Ask me about symptoms, treatments, or diagnosis.</p>
                </div>
            """)
            
            chatbot = gr.Chatbot(
                height=500, 
                show_label=False,
                bubble_full_width=False,
                avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/3774/3774299.png") # User icon (default), Bot icon (Doctor)
            )
            
            with gr.Group():
                with gr.Row():
                    instr_input = gr.Textbox(
                        label="Instruction (Required)", 
                        placeholder="e.g. What are the symptoms of Type 2 Diabetes?", 
                        scale=3,
                        lines=2
                    )
                    extra_input = gr.Textbox(
                        label="Input Context (Optional)", 
                        placeholder="e.g. The patient is 45 years old...", 
                        scale=2,
                        lines=2
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Send Query", variant="primary", scale=2)
                    clear_btn = gr.ClearButton([instr_input, extra_input, chatbot], value="🧹 Clear", scale=1)

    # --- Event Wiring ---
    submit_btn.click(
        predict, 
        inputs=[instr_input, extra_input, chatbot], 
        outputs=[instr_input, extra_input, chatbot, history_display]
    )
    
    # Allow pressing "Enter" on the instruction box to submit
    instr_input.submit(
        predict, 
        inputs=[instr_input, extra_input, chatbot], 
        outputs=[instr_input, extra_input, chatbot, history_display]
    )

if __name__ == "__main__":
    print("🌐 Launching Interface at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)