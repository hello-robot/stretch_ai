# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard imports
import logging
import threading
import time
from enum import Enum
from typing import Optional

# Local imports
from ..base import AbstractTextToSpeech

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class TextToSpeechOverrideBehavior(Enum):
    """
    The TextToSpeechOverrideBehavior class enumerates the possible override
    behaviors for a text-to-speech executor.
    """

    INTERRUPT = 1
    QUEUE = 2


class TextToSpeechExecutor:
    """
    This class executes a text-to-speech engine. It can queue text to speak, stop
    ongoing speech, change the voice, and change the speed.
    """

    def __init__(
        self,
        engine: AbstractTextToSpeech,
        loop_sleep_secs: float = 0.2,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        """
        Initialize the text-to-speech executor.

        Parameters
        ----------
        engine : AbstractTextToSpeech
            The text-to-speech engine
        loop_sleep_secs : float, optional
            The sleep time between loops, by default 0.2
        """
        # Store the logger
        self._logger = logger

        # Declare the attributes for the text-to-speech engine
        self._engine = engine

        # Declare the attributes for the run thread
        self._loop_sleep_secs = loop_sleep_secs
        self._queue: list[tuple[str, str, bool]] = []
        self._queue_lock = threading.Lock()
        self._is_running = False
        self._is_running_lock = threading.Lock()
        self._run_thread: Optional[threading.Thread] = None

    def say_utterance(
        self,
        text: str,
        voice_id: str = "",
        is_slow: bool = False,
        override_behavior: TextToSpeechOverrideBehavior = TextToSpeechOverrideBehavior.QUEUE,
    ) -> None:
        """
        Speak the given text.

        Parameters
        ----------
        text : str
            The text to speak.
        voice_id : str, optional
            The voice ID to use. Do not change the voice if "" (default value).
        is_slow : bool, optional
            Whether to speak slowly, by default False
        override_behavior : TextToSpeechOverrideBehavior, optional
            Whether to interrupt or queue the text, by default queue
        """
        self._logger.debug(f"Received: {text}")

        # Interrupt if requested
        if override_behavior == TextToSpeechOverrideBehavior.INTERRUPT:
            self.stop_utterance()

        # Queue the text
        if len(text) > 0:
            with self._queue_lock:
                self._queue.append((text, voice_id, is_slow))

    def stop_utterance(self) -> None:
        """
        Stop speaking the current text.
        """
        self._logger.debug("Stopping current utterance")
        if self._engine._can_say_async:
            self._engine.stop()
            with self._queue_lock:
                self._queue.clear()
        else:
            self._logger.warning("Asynchronous stopping is not supported for this engine.")

    def start(self) -> None:
        """
        Start the text-to-speech engine.
        """
        with self._is_running_lock:
            if self._is_running:
                self._logger.error("Text-to-speech engine already running")
                return
            self._is_running = True
        self._run_thread = threading.Thread(target=self.run)
        self._run_thread.start()
        self._logger.debug("Started text-to-speech engine")

    def stop(self) -> None:
        """
        Stop the text-to-speech engine.
        """
        with self._is_running_lock:
            if not self._is_running:
                self._logger.error("Text-to-speech engine not running")
                return
            self._is_running = False
        self._run_thread.join()
        self._run_thread = None
        self._logger.debug("Stopped text-to-speech engine")

    def run(self) -> None:
        """
        Run the text-to-speech engine.
        """
        while True:
            # Check if the thread should stop
            with self._is_running_lock:
                if not self._is_running:
                    break

            # Send a single queued utterance to the text-to-speech engine
            if not self._engine.is_speaking():
                text, voice_id, is_slow = "", "", False
                got_text = False
                with self._queue_lock:
                    if len(self._queue) > 0:
                        text, voice_id, is_slow = self._queue.pop(0)
                        got_text = True
                if got_text:
                    # Process the voice
                    if len(voice_id) > 0:
                        if voice_id != self._engine.voice_id:
                            self._engine.voice_id = voice_id

                    # Process the speed
                    if is_slow != self._engine.is_slow:
                        self._engine.is_slow = is_slow

                    # Speak the text
                    self._logger.debug(f"Saying: {text}")
                    if self._engine._can_say_async:
                        self._engine.say_async(text)
                    else:
                        self._engine.say(text)

            # Sleep
            time.sleep(self._loop_sleep_secs)
