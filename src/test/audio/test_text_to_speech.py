# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard Imports
import argparse
import logging
import os
from tempfile import NamedTemporaryFile

# Local imports
from stretch.audio.text_to_speech import (
    GoogleCloudTextToSpeech,
    GTTSTextToSpeech,
    PyTTSx3TextToSpeech,
)
from stretch.audio.utils.metrics import spectral_contrast_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_to_speech(
    save: bool = False,
    similarity_threshold: float = 0.8,
    run_google_cloud_engine: bool = False,
    verbose: bool = False,
) -> None:
    """
    Test the text-to-speech engines.

    Parameters
    ----------
    save : bool, optional
        Whether or not to save the generated audio files, by default False. If True,
        this function **only** saves the ground truth, and does not test the engine.
        This step only needs to be done once.
    similarity_threshold : float, optional
        To pass the test, the similarity between the generated audio and the ground truth
        audio must be greater than this threshold in [0, 1]. The default is 0.9. This
        threshold was computed by comparing a GTTSTextToSpeech generation of an utterance to a GTTSTextToSpeech
        generation of all the other utternaces (difference text, voices, and speeds), and
        picking a threshold greater than their similarities.
    run_google_cloud_engine : bool, optional
        Whether to run the test for the Google Cloud engine, by default False.
        Running it requires the Google Credentials to be set (e.g., with the env
        variable GOOGLE_APPLICATION_CREDENTIALS), which isn't set on Github Actions.
    verbose : bool, optional
        Whether to enable verbose logs, by default False.
    """

    # Configure the test cases
    engines_and_voice_ids = [
        (GTTSTextToSpeech(logger), ["com", "co.uk"]),
        (PyTTSx3TextToSpeech(logger), ["english+m1", "english+f1"]),
    ]
    if run_google_cloud_engine:
        engines_and_voice_ids.append(
            (GoogleCloudTextToSpeech(logger), ["en-US-Neural2-I", "en-US-Standard-F"])
        )
    engine_names = {
        GoogleCloudTextToSpeech: "GoogleCloud",
        GTTSTextToSpeech: "GTTS",
        PyTTSx3TextToSpeech: "PyTTSx3",
    }
    texts = {
        "intro": "Hello, my name is Stretch, and I am a robot here to assist you.",
        "question": "What can I do for you today?",
        "typos": "Hellow thees ees ay haurriblee misspelled stringd esigned to test the. engine,",
    }
    ext = "mp3"

    # Run the test cases
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
    )
    for engine, voice_ids in engines_and_voice_ids:
        engine_name = engine_names[engine.__class__]
        for voice_id in voice_ids:
            engine.voice_id = voice_id
            for filename, text in texts.items():
                for is_slow in [False, True]:
                    engine.is_slow = is_slow

                    # Get the ground truth filepath
                    voice_id_cleaned = voice_id.replace(".", "_").replace("+", "_")
                    full_filename = f"{engine_name}_{voice_id_cleaned}_{'slow_' if is_slow else ''}{filename}.{ext}"

                    ground_truth_filepath = os.path.join(save_dir, full_filename)

                    # Either save the ground truth audio, or save a tempfile and compare
                    # it to the groundtruth audio.
                    if save:
                        engine.save_to_file(text, ground_truth_filepath)
                        logger.info(f"Saved ground truth audio to {ground_truth_filepath}")
                    else:
                        # Save to a tempfile
                        tempfile = NamedTemporaryFile(suffix=f".{ext}")
                        engine.save_to_file(text, tempfile.name)

                        if verbose:
                            with open(tempfile.name, "rb") as f:
                                logger.info(
                                    f"Saved {len(f.read())} bytes of audio to {tempfile.name}"
                                )

                        logger.info(f"Checking similarity for {full_filename}...")

                        similarity = spectral_contrast_similarity(
                            ground_truth_filepath, tempfile.name
                        )

                        # Log the results
                        logger.info(f"...similarity for {full_filename}: {similarity}")

                        # Assert the similarity
                        assert similarity > similarity_threshold, (
                            f"Similarity for {full_filename} was {similarity}, "
                            f"below the threshold of {similarity_threshold}"
                        )
    if save:
        logger.info("Saved the ground truth audio files. Now run the tests without --save.")
    else:
        logger.info("All tests passed!")


if __name__ == "__main__":
    # Configure the arguments
    parser = argparse.ArgumentParser(description="Test the text-to-speech engines.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the generated audio files. This only needs to be done once.",
    )
    parser.add_argument(
        "--run_google_cloud_engine",
        action="store_true",
        help="Whether to run the test for the Google Cloud engine.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logs.",
    )
    args = parser.parse_args()

    # Run the tests
    test_text_to_speech(
        save=args.save, run_google_cloud_engine=args.run_google_cloud_engine, verbose=args.verbose
    )
