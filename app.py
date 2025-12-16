import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pymongo
from datetime import datetime
import uuid
from urllib.parse import quote_plus
import os

# -------------------- MongoDB Integration with DEBUG --------------------
class ChatLogger:
    def __init__(self):
        print("\n" + "="*60)
        print("🔍 MONGODB CONNECTION DEBUG")
        print("="*60)
        
        # Get credentials
        from dotenv import load_dotenv
        load_dotenv()  # This loads the variables from your .env file

# Now update your ChatLogger class to use ONLY the environment variables:
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password)} (hidden)")
        
        # URL encode password
        password_encoded = quote_plus(password)
        mongo_uri = f"mongodb+srv://{username}:{password_encoded}@plk.3vytot0.mongodb.net/?appName=plk"
        
        print(f"Connection URI: mongodb+srv://{username}:****@plk.3vytot0.mongodb.net/?appName=plk")
        
        try:
            print("\n⏳ Attempting to connect to MongoDB...")
            self.client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
            
            # Force connection test
            self.client.admin.command('ping')
            
            self.db = self.client["chatbot_db"]
            self.collection = self.db["gradio_logs"]
            self.connected = True
            
            print("✅ MongoDB connected successfully!")
            print(f"📁 Database: chatbot_db")
            print(f"📋 Collection: gradio_logs")
            print("="*60 + "\n")
            
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"❌ Connection timeout: {e}")
            print("💡 Check: 1) Internet connection 2) IP whitelist in MongoDB Atlas")
            self.connected = False
            self.client = None
        except pymongo.errors.OperationFailure as e:
            print(f"❌ Authentication failed: {e}")
            print("💡 Check: Your MongoDB password is correct")
            self.connected = False
            self.client = None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print(f"Error type: {type(e).__name__}")
            self.connected = False
            self.client = None
    
    def log_conversation(self, session_id, instruction, input_text, response):
        """Save conversation to MongoDB with detailed logging"""
        if not self.connected:
            print("⚠️ MongoDB not connected - skipping log")
            return False
        
        try:
            document = {
                "session_id": session_id,
                "instruction": instruction,
                "input": input_text,
                "response": response,
                "timestamp": datetime.utcnow(),
                "model": "gpt2-large-qlora"
            }
            
            print(f"\n💾 Saving to MongoDB...")
            print(f"   Session: {session_id}")
            print(f"   Instruction: {instruction[:50]}...")
            
            result = self.collection.insert_one(document)
            
            print(f"✅ Saved successfully! Document ID: {result.inserted_id}")
            
            # Verify it was saved
            count = self.collection.count_documents({"session_id": session_id})
            print(f"📊 Total messages in this session: {count}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save to MongoDB: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_connection(self):
        """Test MongoDB read/write"""
        if not self.connected:
            return "❌ Not connected to MongoDB"
        
        try:
            # Try to write a test document
            test_doc = {
                "test": "connection",
                "timestamp": datetime.utcnow()
            }
            result = self.collection.insert_one(test_doc)
            
            # Try to read it back
            found = self.collection.find_one({"_id": result.inserted_id})
            
            # Delete test document
            self.collection.delete_one({"_id": result.inserted_id})
            
            return "✅ MongoDB read/write test successful!"
        except Exception as e:
            return f"❌ Test failed: {e}"

# Initialize MongoDB logger
print("\n🚀 Initializing application...")
chat_logger = ChatLogger()
session_id = f"gradio_{uuid.uuid4().hex[:8]}"
print(f"📝 Session ID: {session_id}\n")

# Test connection
if chat_logger.connected:
    test_result = chat_logger.test_connection()
    print(test_result)

# -------------------- Model & Adapter Setup --------------------
print("\n" + "="*60)
print("🤖 LOADING MODEL")
print("="*60)

model_name = "gpt2-large"
adapter_path = "./qlora_adapter"

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 4-bit QLoRA quantization (only if CUDA available)
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    # CPU fallback
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("✅ Model loaded successfully")
print("="*60 + "\n")

# -------------------- Chat Function with MongoDB Logging --------------------
def chat(instruction, input_text=""):
    """Generate response and log to MongoDB"""
    
    print("\n" + "="*60)
    print("💬 NEW CHAT REQUEST")
    print("="*60)
    
    # Validate input
    if not instruction.strip():
        print("⚠️ Empty instruction received")
        return "⚠️ Please enter an instruction."
    
    print(f"Instruction: {instruction}")
    print(f"Input: {input_text}")
    
    # Create prompt
    prompt = (
        f"Instruction: {instruction}. Respond using a main heading (##) and bullet points (*). "
        f"Be comprehensive but concise.\n"
        f"Input: {input_text}\n"
        f"Response:## "
    )
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("🔄 Generating response...")
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    response = full_response[prompt_length:].strip()
    
    print(f"✅ Response generated ({len(response)} chars)")
    
    # Log to MongoDB
    if chat_logger.connected:
        success = chat_logger.log_conversation(session_id, instruction, input_text, response)
        if success:
            print("✅ Logged to MongoDB")
        else:
            print("❌ Failed to log to MongoDB")
    else:
        print("⚠️ MongoDB not connected - conversation not saved")
    
    print("="*60 + "\n")
    
    return response

def check_mongodb_status():
    """Return MongoDB connection status"""
    if chat_logger.connected:
        try:
            count = chat_logger.collection.count_documents({"session_id": session_id})
            return f"✅ Connected | Messages in session: {count}"
        except:
            return "⚠️ Connected but cannot count documents"
    else:
        return "❌ Not Connected - Check console for errors"

# -------------------- Gradio Interface --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color:#4CAF50;">🧠 GPT2-Large + LoRA Medical Assistant</h1>
            <p style="font-size: 16px;">Enter an <b>instruction</b> and optional <b>input</b>. The model is prompted to use Markdown formatting (headings, lists) for a clean display.</p>
            <p style="font-size: 12px; color: #666;">💾 All conversations are automatically saved to MongoDB</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="e.g., Answer the medical question: What are the symptoms of diabetes?",
                lines=3,
                elem_id="instruction_box",
            )
            input_text = gr.Textbox(
                label="Input (optional)",
                placeholder="e.g., Include information about Type 2 diabetes",
                lines=2,
                elem_id="input_box",
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Generate Response", variant="primary")
                clear_btn = gr.Button("🧹 Clear")
                status_btn = gr.Button("📊 Check MongoDB Status")

        with gr.Column(scale=1):
            output_box = gr.Markdown(
                value="Click 'Generate Response' to see the structured, formatted output here.",
                elem_id="output_box",
                height=350 
            )

    status_box = gr.Textbox(
        label="MongoDB Status",
        value=check_mongodb_status(),
        interactive=False
    )

    # Example buttons
    gr.Examples(
        examples=[
            ["Answer the medical question: What are common treatments for Type 2 diabetes?", ""],
            ["Explain the symptoms of hypertension", "Include risk factors"],
            ["What causes migraine headaches?", "Focus on triggers and prevention"],
            ["Describe the process of diagnosing asthma", ""],
        ],
        inputs=[instruction, input_text],
    )

    # Button actions
    submit_btn.click(chat, inputs=[instruction, input_text], outputs=output_box)
    clear_btn.click(lambda: ("", "", ""), None, [instruction, input_text, output_box])
    status_btn.click(check_mongodb_status, inputs=None, outputs=status_box)
    
    # Footer with status
    gr.HTML(
        f"""
        <div style="text-align: center; margin-top: 30px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;">
            <p style="font-size: 12px; color: #666; margin: 5px;">
                <b>Session ID:</b> {session_id}
            </p>
            <p style="font-size: 12px; color: #666; margin: 5px;">
                <b>Device:</b> {device.upper()}
            </p>
            <p style="font-size: 12px; margin: 5px;">
                <b>Initial MongoDB Status:</b> <span style="color: {'green' if chat_logger.connected else 'red'};">
                    {'✓ Connected' if chat_logger.connected else '✗ Disconnected'}
                </span>
            </p>
            <p style="font-size: 10px; color: #999; margin: 5px;">
                Check console output for detailed connection logs
            </p>
        </div>
        """
    )

# -------------------- Launch --------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 LAUNCHING GRADIO INTERFACE")
    print("="*60)
    print("URL: http://127.0.0.1:7860")
    print("Check this terminal for MongoDB save confirmations")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )