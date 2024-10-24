# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# This is for debugging SimpleAudio issues
# It will just play a simple sine wave tone

from math import pi

import numpy as np
import simpleaudio as sa

# Create a simple sine wave
frequency = 440  # Hz
duration = 1  # seconds
sample_rate = 44100  # samples per second

# Generate time points
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate a sine wave
audio_data = np.sin(2 * pi * frequency * t)

# Normalize to 16-bit range
audio_data *= 32767 / np.max(np.abs(audio_data))

# Convert to 16-bit data
audio_data = audio_data.astype(np.int16)

# Play the audio using simpleaudio
play_obj = sa.play_buffer(audio_data.tobytes(), 1, 2, sample_rate)

# Wait for playback to finish
play_obj.wait_done()
