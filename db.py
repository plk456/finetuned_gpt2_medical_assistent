import os
import pymongo
from datetime import datetime
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ChatMemory:
    def __init__(self, connection_string, db_name="chatbot_db", collection_name="chat_history"):
        try:
            self.client = pymongo.MongoClient(connection_string)
            self.client.server_info()
            print("✓ Successfully connected to MongoDB")
            
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            raise

    def save_message(self, session_id, role, message):
        document = {
            "session_id": session_id,
            "role": role,
            "content": message,
            "timestamp": datetime.utcnow()
        }
        result = self.collection.insert_one(document)
        return result.inserted_id

    def get_history(self, session_id, limit=10):
        cursor = self.collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit)
        
        history = list(cursor)
        return history

    def get_conversation_context(self, session_id, limit=5):
        """Get recent conversation as formatted string for context"""
        history = self.get_history(session_id, limit)
        context = ""
        for msg in history:
            context += f"{msg['role'].upper()}: {msg['content']}\n"
        return context

    def clear_session(self, session_id):
        result = self.collection.delete_many({"session_id": session_id})
        print(f"✓ Deleted {result.deleted_count} messages for session {session_id}")

    def close(self):
        self.client.close()
        print("✓ MongoDB connection closed")


class FineTunedChatbot:
    def __init__(self, memory, base_model_name="gpt2-large", adapter_path="./qlora_adapter"):
        """
        Initialize chatbot with your fine-tuned model
        
        Args:
            memory: ChatMemory instance
            base_model_name: Base model used for fine-tuning
            adapter_path: Path to your LoRA adapter
        """
        self.memory = memory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def generate_response(self, user_input, conversation_history="", max_length=256):
        """
        Generate response using your fine-tuned model
        
        Args:
            user_input: User's question
            conversation_history: Previous conversation context
            max_length: Maximum tokens to generate
        """
        # Format prompt in Alpaca style (same as training)
        prompt = f"Instruction: Answer the following medical question accurately.\nInput: {user_input}\nResponse:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Response:")
        if "Response:" in full_response:
            response = full_response.split("Response:")[-1].strip()
        else:
            response = full_response.strip()
        
        return response
    
    def chat_loop(self, session_id):
        """Main interactive chat loop"""
        print(f"\n{'='*70}")
        print(f"🏥 Medical Q&A Chatbot - Session: {session_id}")
        print(f"{'='*70}")
        print("Type 'exit' or 'quit' to end the conversation")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear conversation history")
        print(f"{'='*70}\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                bot_response = "Goodbye! Your conversation has been saved. Stay healthy!"
                print(f"Bot: {bot_response}")
                self.memory.save_message(session_id, "user", user_input)
                self.memory.save_message(session_id, "bot", bot_response)
                break
            
            if user_input.lower() == 'history':
                self.show_history(session_id)
                continue
            
            if user_input.lower() == 'clear':
                self.memory.clear_session(session_id)
                print("✓ Conversation history cleared!\n")
                continue
            
            # Save user message to MongoDB
            self.memory.save_message(session_id, "user", user_input)
            
            # Get conversation context
            context = self.memory.get_conversation_context(session_id, limit=3)
            
            # Generate response using your fine-tuned model
            print("Bot: [Thinking...]", end="\r")
            bot_response = self.generate_response(user_input, context)
            
            # Save bot response to MongoDB
            self.memory.save_message(session_id, "bot", bot_response)
            
            # Display bot response
            print(f"Bot: {bot_response}\n")
    
    def show_history(self, session_id):
        """Display conversation history"""
        history = self.memory.get_history(session_id, limit=20)
        
        if not history:
            print("No conversation history found.\n")
            return
        
        print(f"\n{'='*70}")
        print("CONVERSATION HISTORY")
        print(f"{'='*70}")
        for msg in history:
            timestamp = msg['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            role = msg['role'].upper()
            content = msg['content']
            print(f"[{timestamp}] {role}: {content}")
        print(f"{'='*70}\n")


# --- MAIN APPLICATION ---
if __name__ == "__main__":
    from urllib.parse import quote_plus

    # Try to load local .env (optional, requires python-dotenv)
    try:
        from dotenv import load_dotenv, find_dotenv

        dotenv_path = find_dotenv()
        loaded_path = None
        if dotenv_path:
            load_dotenv(dotenv_path)
            loaded_path = dotenv_path
        else:
            for alt in ['.env', '.env.local', '.evn']:
                if os.path.exists(alt):
                    load_dotenv(alt)
                    loaded_path = alt
                    break

        if loaded_path:
            print(f"💡 Loaded environment variables from: {loaded_path}")
        else:
            print("💡 No .env file found (checked .env, .env.local, .evn) - using process environment variables")
    except Exception as e:
        print(f"💡 dotenv not used: {e}")

    # Print masked presence of critical vars for debugging
    def _env_summary(key):
        v = os.getenv(key)
        if v is None:
            return f"{key}=<missing>"
        if len(v) > 6:
            return f"{key}={v[:3]}...{v[-3:]}"
        return f"{key}={v}"

    print("Environment summary:", _env_summary('MONGO_URI'), _env_summary('MONGO_USERNAME'), _env_summary('MONGO_PASSWORD'))

    # Use full MONGO_URI if provided, otherwise build from env variables
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        username = os.getenv("MONGO_USERNAME")
        password = os.getenv("MONGO_PASSWORD")
        host = os.getenv("MONGO_HOST", "plk.3vytot0.mongodb.net")

        if not username or not password:
            raise RuntimeError(
                "MongoDB credentials not found. Set MONGO_URI or set MONGO_USERNAME and MONGO_PASSWORD in the environment (or add them to .env)."
            )

        password_encoded = quote_plus(password)
        MONGO_URI = f"mongodb+srv://{username}:{password_encoded}@{host}/?appName=plk"
    
    try:
        # Initialize MongoDB memory
        memory = ChatMemory(MONGO_URI)
        
        # Initialize chatbot with YOUR fine-tuned model
        chatbot = FineTunedChatbot(
            memory=memory,
            base_model_name="gpt2-large",
            adapter_path="./qlora_adapter"  # Your trained adapter
        )
        
        # Generate session ID
        session_id = f"medical_chat_{uuid.uuid4().hex[:8]}"
        
        # Start the interactive chat
        chatbot.chat_loop(session_id)
        
        # Close connection
        memory.close()
        
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")
        memory.close()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if 'memory' in locals():
            memory.close()