# Standard imports
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import pyttsx3
import simpleaudio
import sounddevice  # suppress ALSA warnings # noqa: F401
from gtts import gTTS
from overrides import override
from pydub import AudioSegment

# Create the default logger
logging.basicConfig(level=logging.INFO)
DEFAULT_LOGGER = logging.getLogger(__name__)


class TextToSpeechEngineType(Enum):
    """
    The TextToSpeechEngineType class enumerates the possible text-to-speech
    engines.
    """

    PYTTSX3 = 1
    GTTS = 2


class TextToSpeechEngine(ABC):
    """
    Abstract base class for a text-to-speech engine that supports:
      - Setting the voice ID.
      - Setting the speed to default or slow.
      - Asynchronously speaking text.
      - Interrupting speech.
    """

    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        """
        Initialize the text-to-speech engine.

        Parameters
        ----------
        logger : logging.Logger
            The logger to use for logging messages.
        """
        self._logger = logger
        self._voice_ids: List[str] = []
        self._voice_id = ""
        self._is_slow = False

        # Whether or not this engine can speak asynchronously or not.
        self._can_say_async = False

    @property
    def voice_ids(self) -> List[str]:
        """
        Get the list of voice IDs available for the text-to-speech engine.

        Returns
        -------
        List[str]
            The list of voice IDs.
        """
        return self._voice_ids

    @property
    def voice_id(self) -> str:
        """
        Get the current voice ID for the text-to-speech engine.

        Returns
        -------
        str
            The current voice ID.
        """
        return self._voice_id

    @voice_id.setter
    def voice_id(self, voice_id: str) -> None:
        """
        Set the current voice ID for the text-to-speech engine.

        Parameters
        ----------
        voice_id : str
            The voice ID to set.
        """
        if voice_id in self._voice_ids:
            self._voice_id = voice_id
        else:
            self._logger.error(f"Invalid voice ID: {voice_id}")

    @property
    def is_slow(self) -> bool:
        """
        Get whether the text-to-speech engine is set to speak slowly.

        Returns
        -------
        bool
            Whether the text-to-speech engine is set to speak slowly.
        """
        return self._is_slow

    @is_slow.setter
    def is_slow(self, is_slow: bool) -> None:
        """
        Set whether the text-to-speech engine is set to speak slowly.

        Parameters
        ----------
        is_slow : bool
            Whether to set the text-to-speech engine to speak slowly
        """
        self._is_slow = is_slow

    @abstractmethod
    def say_async(self, text: str) -> None:
        """
        Speak the given text asynchronously.

        Parameters
        ----------
        text : str
            The text to speak.
        """
        raise NotImplementedError

    @abstractmethod
    def is_speaking(self) -> bool:
        """
        Return whether the text-to-speech engine is currently speaking.

        Returns
        -------
        bool
            Whether the text-to-speech engine is currently speaking.
        """
        raise NotImplementedError

    @abstractmethod
    def say(self, text: str) -> None:
        """
        Speak the given text synchronously.

        Parameters
        ----------
        text : str
            The text to speak.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Stop speaking the current text.
        """
        raise NotImplementedError

    @staticmethod
    def is_file_type_supported(filepath: str) -> bool:
        """
        Checks whether the file type is supported by the text-to-speech engine.
        This is a static method to enforce that every text-to-speech engine
        supports the same file type(s). Currently, only MP3 is supported.

        Parameters
        ----------
        filepath : str
            The path of the file to check.

        Returns
        -------
        bool
            Whether the file type is supported.
        """
        return filepath.lower().strip().endswith(".mp3")

    @abstractmethod
    def save_to_file(self, text: str, filepath: str, **kwargs: Dict[str, Any]):
        """
        Save the given text to an audio file.

        Parameters
        ----------
        text : str
            The text to save.
        filepath : str
            The path to save the audio file.
        """
        raise NotImplementedError


class GTTS(TextToSpeechEngine):
    """
    Text-to-speech engine using gTTS.
    """

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)
        self._can_say_async = True

        # Initialize the voices.
        # https://gtts.readthedocs.io/en/latest/module.html#gtts.lang.tts_langs
        self._voice_ids = [
            "com",  # Default
            "us",  # United States
            "com.au",  # Australia
            "co.uk",  # United Kingdom
            "ca",  # Canada
            "co.in",  # India
            "ie",  # Ireland
            "co.za",  # South Africa
            "com.ng",  # Nigeria
        ]
        self.voice_id = "com"
        self._playback: Optional[simpleaudio.PlayObject] = None

    def __synthesize_and_play_text(self, text: str) -> simpleaudio.PlayObject:
        """
        Get the playback object for the given text.

        Parameters
        ----------
        text : str
            The text to speak.

        Returns
        -------
        simpleaudio.PlayObject
            The playback object.
        """
        tts = gTTS(text=text, lang="en", tld=self.voice_id, slow=self.is_slow)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio = AudioSegment.from_file(fp, format="mp3")
        self._playback = simpleaudio.play_buffer(
            audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate
        )

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        self.__synthesize_and_play_text(text)

    @override  # inherit the docstring from the parent class
    def is_speaking(self) -> bool:
        if self._playback is None:
            return False
        if not self._playback.is_playing():
            self._playback = None
            return False
        return True

    @override  # inherit the docstring from the parent class
    def say(self, text: str) -> None:
        self.__synthesize_and_play_text(text)
        self._playback.wait_done()
        self._playback = None

    @override  # inherit the docstring from the parent class
    def stop(self):
        if self._playback is not None:
            self._playback.stop()
            self._playback = None

    @override  # inherit the docstring from the parent class
    def save_to_file(self, text: str, filepath: str, **kwargs: Dict[str, Any]) -> None:
        if not TextToSpeechEngine.is_file_type_supported(filepath):
            self._logger.error(f"Unsupported file type: {filepath} . Must end in `.mp3`")
            return
        tts = gTTS(text=text, lang="en", tld=self.voice_id, slow=self.is_slow)
        tts.save(filepath)


class PyTTSx3(TextToSpeechEngine):
    """
    Text-to-speech engine using pyttsx3. A big benefit of pyttsx3 compared
    to other enginers is that it runs offline. However, its Linux voices tend
    to be less natural than other engines.
    """

    @override  # inherit the docstring from the parent class
    def __init__(self, logger: logging.Logger = DEFAULT_LOGGER):
        super().__init__(logger)
        self._engine = pyttsx3.init()

        # Initialize the voices
        voices = self._engine.getProperty("voices")
        # Variants documentation: https://espeak.sourceforge.net/languages.html
        variants = [
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "f1",
            "f2",
            "f3",
            "f4",
            "croak",
            "whisper",
        ]
        for voice in voices:
            self._voice_ids.append(voice.id)
            for variant in variants:
                self._voice_ids.append(voice.id + "+" + variant)
        self.voice_id = "default"

        # Initialize the speeds
        self.slow_speed = 100  # wpm
        self.default_speed = 150  # wpm

    @TextToSpeechEngine.voice_id.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def voice_id(self, voice_id: str) -> None:
        self._voice_id = voice_id
        self._engine.setProperty("voice", voice_id)

    @TextToSpeechEngine.is_slow.setter  # type: ignore
    @override  # inherit the docstring from the parent class
    def is_slow(self, is_slow: bool) -> None:
        self._is_slow = is_slow
        if is_slow:
            self._engine.setProperty("rate", self.slow_speed)
        else:
            self._engine.setProperty("rate", self.default_speed)

    @override  # inherit the docstring from the parent class
    def say_async(self, text: str) -> None:
        self._logger.warning("Asynchronous speaking is not supported for PyTTSx3 on Linux.")

    @override  # inherit the docstring from the parent class
    def is_speaking(self) -> bool:
        # Because asynchronous speaking is not supported in pyttsxy on Linux,
        # if this function is called, it is assumed that the engine is not speaking.
        # This works as long as `is_speaking` and `say` will be called from
        # the same thread.
        return False

    @override  # inherit the docstring from the parent class
    def say(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    @override  # inherit the docstring from the parent class
    def stop(self) -> None:
        # Although interruptions are nominally supported in pyttsx3
        # (https://pyttsx3.readthedocs.io/en/latest/engine.html#examples),
        # in practice, the Linux implementation spins of an ffmpeg process
        # which can't be interrupted in its current implementation:
        # https://github.com/nateshmbhat/pyttsx3/blob/5d3755b060a980f48fcaf81df018dd06cbd17a8f/pyttsx3/drivers/espeak.py#L175 # noqa: E501
        self._logger.warning("Asynchronous stopping is not supported for PyTTSx3 on Linux.")

    @override  # inherit the docstring from the parent class
    def save_to_file(self, text: str, filepath: str, **kwargs: Any) -> None:
        if not TextToSpeechEngine.is_file_type_supported(filepath):
            self._logger.error(f"Unsupported file type: {filepath} . Must end in `.mp3`")
            return

        # Get the parameters. In practice, because pyttsx3 spawns an ffmpeg
        # process when saving, it does not wait for the process to finish.
        # Therefore, the sleep_secs parameter is used to wait for some time
        # before returning.
        sleep_secs = 2.0
        if "sleep_secs" in kwargs:
            sleep_secs = float(kwargs["sleep_secs"])

        self._engine.proxy.setBusy(True)
        self._engine.save_to_file(text, filepath)
        self._engine.runAndWait()
        time.sleep(sleep_secs)
        self._engine.proxy.setBusy(False)


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
        engine_type: TextToSpeechEngineType = TextToSpeechEngineType.GTTS,
        loop_sleep_secs: float = 0.2,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        """
        Initialize the text-to-speech executor.

        Parameters
        ----------
        engine_type : TextToSpeechEngineType, optional
            The text-to-speech engine type, by default TextToSpeechEngineType.GTTS
        loop_sleep_secs : float, optional
            The sleep time between loops, by default 0.2
        """
        # Store the logger
        self._logger = logger

        # Declare the attributes for the text-to-speech engine
        self._engine_type = engine_type
        self._engine: Optional[TextToSpeechEngine] = None
        self._initialized = False

        # Declare the attributes for the run thread
        self._loop_sleep_secs = loop_sleep_secs
        self._queue: List[Tuple[str, str, bool]] = []
        self._queue_lock = threading.Lock()
        self._is_running = False
        self._is_running_lock = threading.Lock()
        self._run_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """
        Initialize the text-to-speech engine.
        """
        if self._engine_type == TextToSpeechEngineType.PYTTSX3:
            self._engine = PyTTSx3(self._logger)
            self._initialized = True
        elif self._engine_type == TextToSpeechEngineType.GTTS:
            self._engine = GTTS(self._logger)
            self._initialized = True
        else:
            self._logger.error(f"Unsupported text-to-speech {self._engine_type}")
            return
        self._logger.debug(f"Initialized text-to-speech engine: {self._engine_type}")

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
        if not self._initialized:
            self._logger.error("Text-to-speech engine not initialized")
            return
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
        if not self._initialized:
            self._logger.error("Text-to-speech engine not initialized")
            return
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
        if not self._initialized:
            self._logger.error("Text-to-speech engine not initialized")
            return
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
        if not self._initialized:
            self._logger.error("Text-to-speech engine not initialized")
            return
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
