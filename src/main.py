import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from colorama import Fore, Style, init
import json

# Initialize colorama for cross-platform support
init(autoreset=True)

# Check if CUDA is available and select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open("./src/config.json", "r") as f:
    config = json.load(f)

model_name = config["model_path"]
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using the pipeline (which supports chat history natively)
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

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
    """Adds messages to the conversation history."""
    conversation_history.append({"role": role, "content": content})

def generate_response():
    """Generates a response using the full conversation history."""
    outputs = pipe(conversation_history, max_new_tokens=100000000000)
    
    # Assuming the model output is a list of dictionaries and the response structure is correct
    response = outputs[0]["generated_text"]

    # Extracting the latest assistant message correctly
    if isinstance(response, list) and response:  # Ensure response is a list and not empty
        latest_assistant_message = response[-1].get("content", "").strip()
    else:
        latest_assistant_message = ""

    return latest_assistant_message

print("\nType 'exit', 'quit', or 'q' to end the conversation.\n")

while True:
    try:
        user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
        
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("Exiting conversation.")
            break

        add_message("user", user_input)

        response = generate_response()
        add_message("assistant", response)

        print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}\n")
    
    except KeyboardInterrupt:
        print("\nConversation interrupted. Exiting.")
        break
