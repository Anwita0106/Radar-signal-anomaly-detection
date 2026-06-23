import pyttsx3
import threading

def speak_alert(message):
    threading.Thread(
        target=_speak,
        args=(message,),
        daemon=True
    ).start()

def _speak(message):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    engine.say(message)
    engine.runAndWait()

    engine.stop()