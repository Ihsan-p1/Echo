import json
import random
import os

def generate_llm_dataset():
    print("=== Generating LLM Fine-Tuning Dataset (English Only) ===")
    
    intents = {
        "FORWARD": [
            "move forward", "go ahead", "onward", "advance", "step forward",
            "robot, move forward", "please go forward", "can you move forward?"
        ],
        "BACKWARD": [
            "move backward", "go back", "reverse", "step back",
            "back up a bit", "robot, move behind", "please go backwards"
        ],
        "LEFT": [
            "turn left", "rotate left", "go left", "look to the left",
            "spin left", "robot turn left", "can you look left?"
        ],
        "RIGHT": [
            "turn right", "rotate right", "go right", "look to the right",
            "spin right", "robot turn right", "can you look right?"
        ],
        "STOP": [
            "stop now", "halt", "stay there", "freeze", "don't move",
            "stop moving", "cease movement", "emergency stop", "wait here"
        ],
        "NONE": [
            "what do you see?", "identify the objects", "what is that?",
            "is the path clear?", "scan the room", "are there any people?",
            "what's in front of you?", "hello robot", "tell me a joke"
        ]
    }

    objects_pool = ["laptop", "bottle", "person", "chair", "table", "phone", "book", "cup", "keyboard", "mouse"]
    
    dataset = []
    
    for _ in range(200):
        intent = random.choice(list(intents.keys()))
        query = random.choice(intents[intent])
        
        # Random visual context
        num_objs = random.randint(0, 4)
        context = random.sample(objects_pool, num_objs)
        
        # Target response generation
        if intent == "FORWARD":
            response = f"Sure, I'll move forward. I see {', '.join(context) if context else 'a clear path'} ahead."
        elif intent == "BACKWARD":
            response = "Moving backward now. Reversing safely."
        elif intent == "LEFT":
            response = "Turning left. Scanning the area to my left side."
        elif intent == "RIGHT":
            response = "Turning right. Rotating my cameras for a better view."
        elif intent == "STOP":
            response = "Stopping immediately. I'm now stationary."
        else: # NONE
            if "joke" in query:
                response = "Why did the robot go to the doctor? Because it had a virus!"
            elif "see" in query or "identify" in query or "objects" in query:
                if context:
                    response = f"I can see several objects here: {', '.join(context)}."
                else:
                    response = "I don't see any specific objects in my current view."
            elif "clear" in query or "safe" in query:
                if "person" in context:
                    response = "Watch out! There's a person ahead, so it might not be perfectly safe."
                else:
                    response = "The path looks clear from my perspective."
            else:
                response = "Hello! I am your robot assistant. How can I help you today?"

        # Format as Instruction/Input/Output for better LoRA training
        instruction = "You are a friendly robot assistant. You can see through the camera. Analyze the visual context and respond to the user. Extract a motor command if needed (FORWARD, BACKWARD, LEFT, RIGHT, STOP, NONE)."
        input_text = f"Visual Context: {context}\nUser Query: {query}"
        output_text = f"<response>{response}</response>\n<command>{intent}</command>"
        
        dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    output_path = "robot-assistant/data/llm/finetune_data.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Generated {len(dataset)} samples at {output_path}")

if __name__ == "__main__":
    generate_llm_dataset()
