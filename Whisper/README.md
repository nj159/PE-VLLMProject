
## Install whisper
- conda create -n whisper python=3.10 -y
- conda activate whisper
- pip install -U openai-whisper
- pip install git+https://github.com/openai/whisper.git 
- pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

### Install ffmpeg
- sudo apt update
- sudo apt install ffmpeg

### Install Speaker Diarization Tool
pip install pyannote.audio
pip install transformers==4.51.3 accelerate

pip install --upgrade pip
pip install -r requirements.txt

## Steps for Code Execution
### clip_filter4.py
### whisper_pyannote_image.py
### summary_generator.py
### analysisimage.py



