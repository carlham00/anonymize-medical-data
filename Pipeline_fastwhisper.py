import os
# import whisperx
import ffmpeg
import torch
# import subprocess
# import gc
from collections import defaultdict
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# This file is updated to run on the VERANDA vm
# change *** marks if neccessary

### Functions ###

# Function to merge all consecutive segments of one speaker to one coherent text block
def merge_consecutive_speaker_segments(segments):
    merged_segments = []
    prev_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]

        if prev_segment and prev_segment["speaker"] == speaker:
            prev_segment["text"] += " " + text
        else:
            if prev_segment:
                merged_segments.append(prev_segment)
            prev_segment = {"speaker": speaker, "text": text}

    if prev_segment:
        merged_segments.append(prev_segment)

    return merged_segments


# Define paths
base_path = "/home/brendan/git/Audio-Transcript-Anonymizer-TUB-AP" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
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

    # *** adjust to your device features (or features of the VM)
    device = "cuda"  # options: "cpu", "cuda" (if GPU available)
    batch_size = 32  # reduce if low on GPU mem
    # using int8 for local testing (cluster could use float32)
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # Load WhisperX model
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_folder)

    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # *** min & max number of speakers in the audio
    min_speakers = 2
    max_speakers = 4

    for file in os.listdir(audios_folder):
        if file.endswith(".mp3"):
            audio_path = os.path.join(audios_folder, file)
            print(f"Processing: {audio_path}")

            # Load and transcribe audio
            # audio = whisperx.load_audio(audio_path)
            segments, info = model.transcribe(audio_path, beam_size=5, language="en", condition_on_previous_text=False)

            segments = list(segments)

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            # # Save transcript
            transcript_file = os.path.join(transcripts_folder, os.path.splitext(file)[0] + ".txt")
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in segments:
                        text = segment.text
                        f.write(f"{text}\n")
            except Exception as e:
                print("Error writing transcript:", e)

            print(f"Transcription saved: {transcript_file}")




                        # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model

            # Align Whisper output
            # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            # print(result["segments"])  # After alignment

            # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

            # Diarization & assign speaker labels
            # diarize_model = whisperx.DiarizationPipeline(use_auth_token="", device=device)
            # diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers) #add min/max number of speakers if known

            # result = whisperx.assign_word_speakers(diarize_segments, result) # segments are now assigned speaker IDs

            # Merge consecutive speaker segments
            # result["segments"] = merge_consecutive_speaker_segments(result["segments"])


            # anonymized_file = os.path.join(annonym_folder, os.path.basename(transcript_file))


            # command = [
            #     "python", "-m", "philter_ucsf",
            #     "-i", transcript_file,
            #     "-o", anonymized_file,
            # ]

            # #run subprocess with
            # subprocess.run(command, check=True)

else:
    print("No 'audios' folder found. No files to transcribe.")

