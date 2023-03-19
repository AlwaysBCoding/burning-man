import librosa
import pyaudio
import threading
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from typing import NamedTuple
import wave

class StreamParams(NamedTuple):
  format: int = pyaudio.paInt16
  channels: int = 1
  rate: int = 44100
  frames_per_buffer: int = 1024
  input: bool = True
  output: bool = False
  input_device_index: int = 2

  def to_dict(self) -> dict:
    return self._asdict()

class Recorder:
  def __init__(self, stream_params: StreamParams) -> None:
    self.stream_params = stream_params
    self._pyaudio = None
    self._stream = None
    self._wav_file = None

  def record(self, duration: int, save_path: str) -> None:
    print("Start recording...")
    self._create_recording_resources(save_path)
    self._write_wav_file_reading_from_stream(duration)
    self._close_recording_resources()
    print("Stop recording.")

  def _create_recording_resources(self, save_path: str) -> None:
    self._pyaudio = pyaudio.PyAudio()
    self._stream = self._pyaudio.open(**self.stream_params.to_dict())
    self._create_wav_file(save_path)

  def _create_wav_file(self, save_path: str):
    self._wav_file = wave.open(save_path, "wb")
    self._wav_file.setnchannels(self.stream_params.channels)
    self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
    self._wav_file.setframerate(self.stream_params.rate)

  def _write_wav_file_reading_from_stream(self, duration: int) -> None:
    for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
      audio_data = self._read(self.stream_params.frames_per_buffer)
      self._wav_file.writeframes(audio_data)

  def _close_recording_resources(self) -> None:
    self._wav_file.close()
    self._stream.close()
    self._pyaudio.terminate()

if __name__ == "__main__":
  stream_params = StreamParams().to_dict()
  # recorder = Recorder(stream_params)
  # recorder.record(5, "output.wav")

  checkpoint = "microsoft/speecht5_asr"
  processor = SpeechT5Processor.from_pretrained(checkpoint)
  model = SpeechT5ForSpeechToText.from_pretrained(checkpoint)

  chunk = 1024
  output_filename = "output.wav"
  audio = pyaudio.PyAudio()
  stream = audio.open(**stream_params)
  is_recording = True

  def record_audio(frames):
    while is_recording:
      data = stream.read(chunk)
      frames.append(data)

  def stop_recording():
    global is_recording
    is_recording = False
    print("Finished recording.")

  frames = []
  recording_thread = threading.Thread(target=record_audio, args=(frames,))
  recording_thread.start()

  print("Recording... Press 'Enter' to stop recording.")

  input()

  stop_recording()
  recording_thread.join()

  stream.stop_stream()
  stream.close()

  audio.terminate()

  with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(stream_params.get('channels'))
    wf.setsampwidth(audio.get_sample_size(stream_params.get('format')))
    wf.setframerate(stream_params.get('rate'))
    wf.writeframes(b''.join(frames))

  waveform, sample_rate = librosa.load(output_filename)
  inputs = processor(audio=waveform, sampling_rate=16000, return_tensors="pt")
  predicted_ids = model.generate(**inputs, max_length=400)
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
  print(f"{transcription[0]}")

# =====================

# import librosa
# import torch
# import sounddevice as sd
# from transformers import SpeechT5Processor, SpeechT5ForSpeechToText

# checkpoint = "microsoft/speecht5_asr"
# processor = SpeechT5Processor.from_pretrained(checkpoint)
# model = SpeechT5ForSpeechToText.from_pretrained(checkpoint)

# def process_audio(sampling_rate, waveform):
#   # convert from int16 to floating point
#   waveform = waveform / 32678.0

#   # convert to mono if stereo
#   if len(waveform.shape) > 1:
#     waveform = librosa.to_mono(waveform.T)

#   # resample to 16kHz if necessary
#   if sampling_rate != 16000:
#     waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)

#   # limit to 30 seconds
#   waveform = waveform[:16000*30]

#   # make PyTorch tensor
#   waveform = torch.tensor(waveform)
#   return waveform

# def predict(audio, mic_audio=None):
#   # audio = tuple (sample_rate, frames) or (sample_rate, (frames, channels))
#   if mic_audio is not None:
#     sampling_rate, waveform = mic_audio
#   elif audio is not None:
#     sampling_rate, waveform = audio
#   else:
#     return "(please provide audio)"

#   waveform = process_audio(sampling_rate, waveform)
#   inputs = processor(audio=waveform, sampling_rate=16000, return_tensors="pt")
#   predicted_ids = model.generate(**inputs, max_length=400)
#   transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#   return transcription[0]

# =====================

# from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
# from datasets import load_dataset

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
# example_speech = dataset[0]["audio"]["array"]

# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
# model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

# inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

# predicted_ids = model.generate(**inputs, max_length=100)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription[0])
