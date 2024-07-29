# Standard imports
import time

# Local imports
from stretch.audio.text_to_speech import GTTS

engine = GTTS()
engine.voice_id = "com"
engine.is_slow = False
engine.say("Hello, my name is Stretch")
engine.voice_id = "co.uk"
engine.is_slow = True
engine.say_async("Please interrupt me, I am asking to be interrupted.")
time.sleep(1.0)
engine.stop()
