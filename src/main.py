import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from colorama import Fore, Style, init
import json
import threading
import os

os.system('cls')
os.system('title Markus')

# Initialize colorama for cross-platform support
init(autoreset=True)

# Check if CUDA is available and select device
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    print(f"{Fore.RED}Warning: CUDA is not available. Running on CPU may be slow.{Style.RESET_ALL}")

with open("./src/config.json", "r") as f:
    config = json.load(f)

model_name = config["model_path"]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If the pad token is not set, assign it to the eos token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Initialize conversation history with a system message
conversation_history = [
    {"role": "system", "content": """
Your name is Markus.

You are a highly capable and intelligent personal assistant. Your role is to provide helpful, accurate, and context-aware responses while maintaining a friendly yet professional demeanor.

Efficiency: Answer questions concisely but with enough detail to be useful.
Proactiveness: Anticipate user needs and offer relevant suggestions or follow-ups.
Politeness & Professionalism: Always use a respectful and approachable tone.
Adaptability: Adjust your style based on user preferences and the situation.
Clarity: Ensure responses are easy to understand and well-structured.

When assisting the user, consider their past interactions (if available) to personalize responses. If asked for opinions, provide a balanced and well-reasoned perspective.

When handling tasks like scheduling, reminders, research, or summarizing content, focus on precision and actionable insights.

Keep interactions engaging but do not overuse unnecessary enthusiasm. Your goal is to be the ideal assistant: knowledgeable, efficient, and personable.
"""}
]

def add_message(role, content):
    """Adds a message to the conversation history."""
    conversation_history.append({"role": role, "content": content})

def build_prompt():
    """
    Converts the conversation history into a single prompt string.
    Each message is prepended with its role.
    """
    prompt = ""
    for message in conversation_history:
        role = message["role"].capitalize()
        prompt += f"{role}: {message['content'].strip()}\n"
    prompt += "Assistant: "
    return prompt

def generate_response():
    """
    Generates a response by streaming tokens from the model.
    Uses TextIteratorStreamer to print tokens as they are generated.
    """
    prompt = build_prompt()
    # Tokenize the prompt while requesting the attention mask and padding
    encoding = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    # Create a streamer that will yield tokens as they are generated.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Launch generation in a separate thread so we can stream output.
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 1000000000,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": streamer,
        }
    )
    thread.start()

    response = ""
    print(f"{Fore.GREEN}Assistant:", end="", flush=True)
    # Iterate over streamed tokens and print them immediately.
    for token in streamer:
        print(token, end="", flush=True)
        response += token
    thread.join()
    print(Style.RESET_ALL)
    return response

while True:
    try:
        print("")
        user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
        add_message("user", user_input)
        print("")
        response = generate_response()
        add_message("assistant", response)
    except KeyboardInterrupt:
        print("\nConversation interrupted. Exiting.")
        break
