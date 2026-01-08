---
tags:
- pyannote
- pyannote-audio
- pyannote-audio-model
- audio
- voice
- speech
- voice-activity-detection
- speech-to-noise ratio
- snr
- room acoustics
- c50
datasets:
- LibriSpeech
- AudioSet
- EchoThief
- MIT-Acoustical-Reverberation-Scene
license: openrail
extra_gated_prompt: "The collected information will help acquire a better knowledge of this model userbase and help its maintainers apply for grants to improve it further. "
extra_gated_fields:
  Company/university: text
  Website: text
  I plan to use this model for (task, type of audio data, etc): text
---

# 🎙️🥁🚨🔊 Brouhaha

![Sample Brouhaha predictions](brouhaha.gif)

**Joint voice activity detection, speech-to-noise ratio, and C50 room acoustics estimation**

[TL;DR](https://twitter.com/LavechinMarvin/status/1585645131251605504) | [Paper](https://arxiv.org/abs/2210.13248) | [Code](https://github.com/marianne-m/brouhaha-vad) | [And Now for Something Completely Different](https://www.youtube.com/watch?v=8ZyOAS22Moo)



## Installation

This model relies on [pyannote.audio](https://github.com/pyannote/pyannote-audio) and [brouhaha-vad](https://github.com/marianne-m/brouhaha-vad).

```bash
pip install pyannote-audio
pip install https://github.com/marianne-m/brouhaha-vad/archive/main.zip
```

## Usage

```python
# 1. visit hf.co/pyannote/brouhaha and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/brouhaha", 
                              use_auth_token="ACCESS_TOKEN_GOES_HERE")

# apply model 
from pyannote.audio import Inference
inference = Inference(model)
output = inference("audio.wav")

# iterate over each frame
for frame, (vad, snr, c50) in output:
    t = frame.middle
    print(f"{t:8.3f} vad={100*vad:.0f}% snr={snr:.0f} c50={c50:.0f}")

#  ...
# 12.952 vad=100% snr=51 c50=17
# 12.968 vad=100% snr=52 c50=17
# 12.985 vad=100% snr=53 c50=17
# ...
```

## Citation

```bibtex
@article{lavechin2022brouhaha,
  Title   = {{Brouhaha: multi-task training for voice activity detection, speech-to-noise ratio, and C50 room acoustics estimation}},
  Author  = {Marvin Lavechin and Marianne Métais and Hadrien Titeux and Alodie Boissonnet and Jade Copet and Morgane Rivière and Elika Bergelson and Alejandrina Cristia and Emmanuel Dupoux and Hervé Bredin},
  Year    = {2022},
  Journal = {arXiv preprint arXiv: Arxiv-2210.13248}
}

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```
