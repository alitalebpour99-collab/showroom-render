# showroom-render

Reel render pipeline for the Taleb Pour Car City Instagram page (style «گالری طلایی» / Golden Gallery).
CPU only: ffmpeg + OpenCV + librosa + open-image-models (plate detection). Runs on GitHub Actions.

- `tools/make_lut.py` - house LUT (33^3 .cube)
- `tools/render_detail.py` - template «جزئیات روی بیت» (10 s, beat-synced details)
- `inputs/bundle.enc` - footage and owned music, AES-256 encrypted (key = repo secret). Nothing private is readable here.
