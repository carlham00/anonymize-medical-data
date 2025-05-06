import os
import whisperx
import ffmpeg
import torch
import subprocess
import gc
from collections import defaultdict
from pydub import AudioSegment
import InputToTranscript
import nlp_differenciate_people as RedactReplace

import spacy

# Define paths
base_path = "xxx" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
videos_folder = os.path.join(base_path, "videos")
audios_folder = os.path.join(base_path, "audios")
transcripts_folder = os.path.join(base_path, "transcripts")
model_folder = os.path.join(base_path, "model")
annonym_folder = os.path.join(base_path, "annonym")
config_file = "philter-config.json"

# Create transcripts folder if doesn't exist
if not os.path.exists(transcripts_folder):
    os.makedirs(transcripts_folder)

# Create annonym folder if doesn't exist
if not os.path.exists(annonym_folder):
    os.makedirs(annonym_folder)

# Create model folder if doesn't exist
if not os.path.exists(model_folder):
    os.makedirs(model_folder)


##############################################
### CREATE TRANSCRIPTS FROM VIDEO OR AUDIO ###
##############################################

# 1. Check if videos folder exists, extract audios if needed
if os.path.exists(videos_folder):
    print("Found 'videos' folder. Extracting audio from MP4 files...")

    if not os.path.exists(audios_folder):
        os.makedirs(audios_folder)

    for file in os.listdir(videos_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(videos_folder, file)
            audio_filename = os.path.splitext(file)[0] + ".mp3"
            audio_path = os.path.join(audios_folder, audio_filename)

            if not os.path.exists(audio_path):  #Avoid redundant processing
                print(f"Extracting audio from {video_path}...")
                ffmpeg.input(video_path).output(audio_path, format="mp3", audio_bitrate="192k").run(overwrite_output=True)
                print(f"Saved extracted audio: {audio_path}")
            else:
                print(f"Audio already extracted: {audio_path}")

else:
    print("No 'videos' folder found. Skipping video extraction.")


# 2. Transcribe audio
if os.path.exists(audios_folder):
    print("Found 'audios' folder. Starting transcription process...")
    # InputToTranscript.transcript_from_audio()
    
    #######################################
    ###     DE-IDENTIFY TRANSCRIPTS     ###
    #######################################
    nlp = spacy.load('de_core_news_sm')

    for file in os.listdir(transcripts_folder):
        file_path = os.path.join(transcripts_folder, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        doc = nlp(text)

        names = RedactReplace.redact_names_in_doc(doc)
        ages = RedactReplace.redact_age_in_doc(names)

        # Save the anonymized text to a new file
        anonymized_file_path = os.path.join(annonym_folder, os.path.splitext(file)[0] + "_anonymized.txt")
        with open(anonymized_file_path, 'w', encoding='utf-8') as f:
            f.write(str(ages))
        print(f"Anonymized transcript saved: {anonymized_file_path}")
else:
    print("No 'audios' folder found. No files to transcribe.")