# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Third-party imports
import librosa
import numpy as np


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
