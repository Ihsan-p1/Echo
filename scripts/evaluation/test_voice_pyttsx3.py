import pyttsx3

def test_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(f"Found {len(voices)} voices.")
    
    target_voice = None
    for voice in voices:
        print(f"ID: {voice.id} | Name: {voice.name}")
        if "EN-US" in voice.id.upper() or "ENGLISH" in voice.name.upper():
            target_voice = voice.id
            print("  ^ Selected for English test")
            break
            
    if target_voice:
        engine.setProperty('voice', target_voice)
    engine.setProperty('rate', 170)
    engine.say("Hello, I am testing the English voice output for Echo.")
    engine.runAndWait()

if __name__ == "__main__":
    test_voices()
