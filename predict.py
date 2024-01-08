# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess
import nemo.collections.asr as nemo_asr
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")

    def predict(self, audio_file: Path = Input(description="Input audio file to be transcribed by the ASR model"),) -> str:
        """Transcribes the input audio file using the ASR model and returns the transcription as a string"""

        # The model returns a tuple of lists, each containing a single string (transcription)
        transcription = self.asr_model.transcribe([str(audio_file)])
        if transcription and transcription[0]:
            return transcription[0][0]
        else:
            print("Error: Transcription failed or returned empty result.")
            return ""
