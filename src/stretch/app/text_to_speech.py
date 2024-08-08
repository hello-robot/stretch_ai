# # Copyright (c) Hello Robot, Inc.
# #
# # This source code is licensed under the APACHE 2.0 license found in the
# # LICENSE file in the root directory of this source tree.
# #
# # Some code may be adapted from other open-source works with their respective licenses. Original
# # licence information maybe found below, if so.
#

# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


"""
This script adds a command-line interface (CLI) for text-to-speech. The CLI supports:
- Typing text to convert to speech (added to queue).
- Stopping ongoing speech.
- Using the up arrow key to access the history.
- Using tab auto-complete to search through the history.
- Passing in a custom file to load the history (including pre-seeded text) from
  and save it to.
"""
# Standard imports
import argparse
import os
import readline  # Improve interactive input, e.g., up to access history, tab auto-completion.

from stretch.audio.text_to_speech.executor import TextToSpeechExecutor
from stretch.audio.text_to_speech.gtts_engine import GTTSTextToSpeech

# Local imports
from stretch.audio.utils.cli import HistoryCompleter


class TextToSpeechComandLineInterface:
    """
    A command-line interface to use text-to-speech.
    """

    def __init__(self):
        """
        Initialize the TextToSpeechComandLineInterface.
        """
        self._executor = TextToSpeechExecutor(
            engine=GTTSTextToSpeech(),
        )

    def start(self) -> None:
        """
        Start the text-to-speech command line interface.
        """
        self._executor.start()

    def stop(self) -> None:
        """
        Stop the text-to-speech command line interface.
        """
        self._executor.stop()

    def run(self):
        """
        Run the text-to-speech command line interface.
        """
        # Create the input prompt
        print("****************************************************************")
        print("Instructions:")
        print("    Type a message to convert to speech.")
        print("    Press S to stop the current message.")
        print("    Press Q to exit and stop the current message.")
        print("    Press Ctrl-C to exit without stopping the current message")
        print("****************************************************************")

        # Get the user input
        while True:
            # Get the user input
            message = input("\nMessage (S to stop, Q to exit): ").strip()

            # Process the special 1-character commands
            if len(message) == 0:
                continue
            elif len(message) == 1:
                if message.upper() == "Q":
                    self._executor.stop_utterance()
                    readline.remove_history_item(readline.get_current_history_length() - 1)
                    raise KeyboardInterrupt
                elif message.upper() == "S":
                    # Stop the current message
                    self._executor.stop_utterance()
                    readline.remove_history_item(readline.get_current_history_length() - 1)
                    continue

            # Publish the message
            self._executor.say_utterance(message)


def get_args() -> argparse.Namespace:
    """
    Get the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Text-to-speech command line interface.")
    parser.add_argument(
        "--history_file",
        type=str,
        default="",
        help="The history file to load and save.",
    )
    return parser.parse_args()


def main():
    """
    Run the text-to-speech command line interface.
    """
    # Get the arguments
    args = get_args()

    # Load the history
    if len(args.history_file) > 0 and os.path.exists(args.history_file):
        readline.read_history_file(args.history_file)
        print(f"Loaded the history from {args.history_file}")
    readline.set_completer(HistoryCompleter().complete)
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims("")  # Match the entire string, not individual words

    # Initialize the text-to-speech command line interface
    cli = TextToSpeechComandLineInterface()
    cli.start()

    # Run the text-to-speech command line interface
    try:
        cli.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Save the history
        if len(args.history_file) > 0:
            readline.write_history_file(args.history_file)
            print(f"Saved the history to {args.history_file}")
        else:
            print("Did not save the history. To do so, pass in --history_file")

        # Stop the text-to-speech command line interface
        cli.stop()

        print("Cleanly terminated.")


if __name__ == "__main__":
    main()
