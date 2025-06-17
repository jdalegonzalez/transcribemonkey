import argparse
import os

def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video"
    )
    parser.add_argument(
        "-a", "--annote",
        help="Use PyAnnote for diarization",
        dest="use_pyannote",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-c", "--clips",
        help="A set of clips to request in the form of start1,stop1,start2,stop2",
        dest="clips",
        default=None
    )
    parser.add_argument(
        "-d", "--doc", 
        help="Create a word document from a json file.  The -t argument must also be passed.",
        dest='word_doc',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-e", "--episode",
        help="The episode number, if there is one (and you know it).",
        dest="episode",
        default=""
    )
    parser.add_argument(
        "-g", "--segment-folder",
        help="The root folder to put the individual audio segments.",
        dest="audio_folder",
        default=""
    )
    parser.add_argument('-k', '--kill_anomalies',
        help="Whether or not to drop dialog that looks like a hallucination.",
        dest='kill_anomalies',
        default=False,
        action='store_true'
    )
    parser.add_argument('--lang',
        help="The language the audio is in, if known.",
        dest='lang',
        default="en"
    )
    parser.add_argument(
        '-l', '--log_level',
        help="The python logging level for output",
        dest='log_level',
        default="WARNING"
    )
    parser.add_argument(
        "-o", "--out",
        help="The path output audio file.  The default is ./<youtube_id>.mp4",
        dest="filename",
        default=""
    )
    parser.add_argument(
        "-p", "--plot",
        help="Shows the melspectorgram for a section of audio.  The clip argument must also be passed in.",
        dest="do_plot",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "-s", "--short",
        help="Whether the ID is a video short or a full length video.",
        dest="is_a_short",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-t", "--transcript",
        help="If we've been asked to transcribe, this is either the path create the json with the default name or, if the argument ends in .json, the full path and name of the file.  The default name is <transcript_json>/<youtube_id>.transcription.json. If we're creating the word doc, it's the full path to the json file or the json folder.",
        dest="transcript_json",
        default=""
    )
    parser.add_argument(
        "-w", "--whisper",
        help="Use the whisper model instead of the faster-whisper one.",
        dest="use_whisper",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-v", "--video", 
        help="The Youtube ID for the video to be transcribed",
        dest='video_id',
        default=None
    )
    parser.add_argument(
        "-x", "--text", 
        help="Print a minimal, text-based transcript to stdout.",
        dest='print_text',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    # We're not going to require a video id if we've been given
    # the full path to a transcript.
    transcript_exists = os.path.isfile(args.transcript_json)
    if args.video_id is None and not transcript_exists:
        parser.error("Youtube ID (-v, --video) is required if you're not referrencing an existing transcript.")

    if args.word_doc or args.print_text:
        exists = args.transcript_json and os.path.exists(args.transcript_json)
        if not exists:
            parser.error("A path to the json transcript (-t, --transcript) is required when generating a MS Word or text transcript")
        is_file = os.path.isfile(args.transcript_json)
        if not is_file and args.video_id is None:
            parser.error("A full path to a json file is required when generating an MS Word doc or transcript if you don't also supply a video ID")
        if not is_file:
            new_file = os.path.join(args.transcript_json, f'{args.video_id}.transcript.json')
            if not os.path.exists(new_file) or not os.path.isfile(new_file):
                parser.error(f'The transcript {new_file} does not exist')
            args.transcript_json = new_file

    if args.do_plot and not args.clips:
        parser.error("Clips must be specified to plot the spectogram.  Use -c '0' if you really want to do the entire file.")

    if args.clips:
        clips = [float(clip) for clip in args.clips.split(",")]
        args.clips = clips

    return args

