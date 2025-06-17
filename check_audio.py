import argparse
import glob
import os
import shutil

import sounddevice as sd

from transcribe import audio_from_file

class Bcolors:
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    CYAN = '\033[96m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

class _Getch:
    """
    Gets a single character from standard input.  Does not echo to the
    screen.
    """
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

getch = _Getch()

def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Plays a series of files and verifies the labeling"
    )

    parser.add_argument(
        "-g", "--segment-folder",
        help="The root folder to put the individual audio segments.  If missing, audio segments will not be generated.",
        dest="audio_folder",
        required=True
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    root_path = args.audio_folder

    # The overall structure of the samples folder is...
    # <arbritrary_path>/<youtube id>/<label>/*.wav.
    # We might have been give the arbitrary path, 
    # the path to a specific youtube id, or the path 
    # to the label. If we've got wav files in our glob,
    # we're already in the label folder and our parent
    # is the youtube.
    label = ""
    youtube_id = ""


    if not os.path.isdir(root_path):
        print(f"There's nothing I can do with {root_path}")
        quit()
    
    paths_to_wavs = []

    def add_folder(folders):
    
        if not folders:
            return

        for f in folders:
            to_process = glob.glob(os.path.join(f, "*"))
            waves = [ w for w in to_process if w.endswith(".wav")]
            if waves:
                paths_to_wavs.append(f)
            else:
                folders = [f for f in to_process if os.path.isdir(f)]
                add_folder(folders)

    add_folder([root_path])


    if not paths_to_wavs:
        print(f"I couldn't find any wav files to process in {root_path}")
        quit()

    for folder in paths_to_wavs:
        head, label = os.path.split(os.path.normpath(folder))
        _, youtube_id = os.path.split(head)
        wavs = [wv for wv in glob.glob(os.path.join(folder, "*")) if wv.endswith(".wav")]
        other_labels = [ os.path.basename(os.path.normpath(lbl)) for lbl in glob.glob(os.path.join(head, "*")) ]

        for wav in wavs:
            file = os.path.basename(os.path.normpath(wav))
            dta, _ = audio_from_file(wav, sampling_rate=44100)
            def p():
                sd.stop()
                sd.play(dta)
            p()
            b = Bcolors.BOLD
            e = f'{Bcolors.ENDC}'
            vc = f'{b}{Bcolors.OKGREEN}'
            vl = f'{b}{Bcolors.OKBLUE}'
            vf = f'{b}{Bcolors.MAGENTA}'
            w = Bcolors.WHITE
            print(
                f'\nVid: {vc}{youtube_id}{e}, label: {vl}{label}{e}, file: {vf}{file}{e}'
            )
            prompt = True
            while prompt:
                print(f"{w}{b}M){e}ove, {w}{b}D){e}elete, {w}{b}K){e}eep, {w}{b}R){e}eplay, {w}{b}Q){e}uit")
                command = getch().upper()
                if command == 'Q' or command == '\x03':
                    prompt = False
                    quit()
                elif command == 'R':
                    p()
                elif command == 'D':
                    print(f'Deleting {file}')
                    os.remove(wav)
                    prompt = False
                elif command == 'K':
                    print(f'Keeping {file}')
                    prompt = False
                elif command == 'M':
                    print(other_labels)
                    new_label = input('Enter the new label: ')
                    if new_label in other_labels:
                        print(f'Moving to {new_label}')
                        new_path = os.path.join(head, new_label, file)
                        shutil.move(wav, new_path)
                        prompt = False


