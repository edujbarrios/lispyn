; %% [markdown]
; ## TorchAudio features on a synthetic waveform
; No media files, downloads, or GPU needed.
; %% [code]
(import torch)
(import torchaudio)
(setv rate 16000)
(setv sample (torch.arange 0 2048 :dtype torch.float32))
(setv signal (torch.sin (* (/ (* 2 torch.pi 440) rate) sample)))
(setv waveform (.unsqueeze signal 0))
(setv transform (torchaudio.transforms.MelSpectrogram :sample_rate rate :n_mels 32))
(setv mel (transform waveform))
(print "mel spectrogram:" mel.shape)
