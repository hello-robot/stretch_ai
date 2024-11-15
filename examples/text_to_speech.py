# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard imports
import time

# Local imports
from stretch.audio.text_to_speech.gtts_engine import GTTSTextToSpeech

engine = GTTSTextToSpeech()
engine.voice_id = "com"
engine.is_slow = False
engine.say("Hello, my name is Stretch")
engine.voice_id = "co.uk"
engine.is_slow = True
engine.say_async("Please interrupt me, I am asking to be interrupted.")
time.sleep(1.0)
engine.stop()
