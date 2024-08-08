# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

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
