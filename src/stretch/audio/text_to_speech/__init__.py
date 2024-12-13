# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from .executor import AbstractTextToSpeech, TextToSpeechExecutor, TextToSpeechOverrideBehavior


def get_text_to_speech(name: str) -> AbstractTextToSpeech:
    """
    Get the text-to-speech engine by name.

    Parameters
    ----------
    name : str
        The name of the text-to-speech engine.
    logger : logging.Logger, optional
        The logger to use, by default DEFAULT_LOGGER

    Returns
    -------
    AbstractTextToSpeech
        The text-to-speech engine.
    """
    name = name.lower()
    if name == "google_cloud":
        from .google_cloud_engine import GoogleCloudTextToSpeech

        return GoogleCloudTextToSpeech()
    if name == "gtts":
        from .gtts_engine import GTTSTextToSpeech

        return GTTSTextToSpeech()
    if name == "pyttsx3":
        from .pyttsx3_engine import PyTTSx3TextToSpeech

        return PyTTSx3TextToSpeech()
    raise ValueError(f"Unsupported text-to-speech engine: {name}")
