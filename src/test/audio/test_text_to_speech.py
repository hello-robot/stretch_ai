# Standard Imports
import logging
import os
from tempfile import NamedTemporaryFile

# Third-party imports
import librosa
import numpy as np

# Local imports
from stretch.audio.text_to_speech import GTTS, PyTTSx3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Adapted from https://github.com/markstent/audio-similarity/blob/main/audio_similarity/audio_similarity.py
# Note that that script has other audio similarity metrics as well
def spectral_contrast_similarity(ground_truth_filepath, comparison_filepath, sample_rate=16000):
    """
    Calculate the spectral contrast similarity between audio files.

    Parameters
    ----------
    ground_truth_filepath : str
        The path to the ground truth audio file.
    comparison_filepath : str
        The path to the comparison audio file.
    sample_rate : int, optional
        The sample rate of the audio files, by default 16000.

    Note
    ----
    The spectral contrast similarity is calculated by comparing the spectral contrast of the audio signals.
    Spectral contrast measures the difference in magnitudes between peaks and valleys in the spectrum,
    representing the perceived amount of spectral emphasis. The spectral contrast similarity score is
    obtained by comparing the spectral contrast of the original and compare audio files and calculating
    the average normalized similarity. The similarity score ranges between 0 and 1, where a higher score
    indicates greater similarity.

    """
    # Load the audio files
    ground_truth_audio, _ = librosa.core.load(ground_truth_filepath, sr=sample_rate)
    comparison_audio, _ = librosa.core.load(comparison_filepath, sr=sample_rate)

    # Get the spectral contrasts
    ground_truth_contrast = librosa.feature.spectral_contrast(y=ground_truth_audio, sr=sample_rate)
    comparison_contrast = librosa.feature.spectral_contrast(y=comparison_audio, sr=sample_rate)
    min_columns = min(ground_truth_contrast.shape[1], comparison_contrast.shape[1])
    ground_truth_contrast = ground_truth_contrast[:, :min_columns]
    comparison_contrast = comparison_contrast[:, :min_columns]

    # Get the similarity
    contrast_similarity = np.mean(np.abs(ground_truth_contrast - comparison_contrast))
    normalized_similarity = 1 - contrast_similarity / np.max(
        [np.abs(ground_truth_contrast), np.abs(comparison_contrast)]
    )

    return normalized_similarity


def test_text_to_speech(
    save: bool = False, similarity_threshold: float = 0.9, verbose: bool = False
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
        threshold was computed by comparing a GTTS generation of an utterance to a GTTS
        generation of all the other utternaces (difference text, voices, and speeds), and
        picking a threshold greater than their similarities.
    verbose : bool, optional
        Whether to enable verbose logs, by default False.
    """
    # Configure the test cases
    engines_and_voice_ids = [
        (GTTS(logger), ["com", "co.uk"]),
        (PyTTSx3(logger), ["english+m1", "english+f1"]),
    ]
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
        engine_name = engine.__class__.__name__
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
    logger.info("All tests passed!")


if __name__ == "__main__":
    # # Run this one-time to generate the ground-truth files. This may need to
    # # be re-run from time-to-time to update the ground-truth, (e.g., if Google
    # # updates their TTS engine).
    # test_text_to_speech(save=True)

    test_text_to_speech()
