# showroom-render

Reel render pipeline for the Taleb Pour Car City Instagram page (style «گالری طلایی» / Golden Gallery).
CPU only: ffmpeg + OpenCV + librosa + open-image-models (plate detection). Runs on GitHub Actions.

- `tools/make_lut.py` - house LUT (33^3 .cube)
- `tools/render.py <work> detail|send|versus` - the Reel engine (one template per matrix job)
- `tools/sync_check.py` - independent check of a finished Reel: beats from its audio, cuts from its frames
- `inputs/bundle.enc` - footage and older owned music, AES-256 encrypted (key = repo secret)
- `inputs/music.enc` - owned Setavaz tracks made for the page + `music/showroom.json`, same key

## Beat sync (every Reel, automatically)

Each template is written in **beats**, not seconds: `TEMPLATES` gives the number of whole bars and the bar where the
reveal starts; `template_specs()` lists (shot, beats, transition). The engine then:

1. tracks the beats of the template's track and fits them to a straight line (steady tempo, exact beat times);
2. finds the drop (largest low-end jump, near the hint in `showroom.json`) and counts bars from it;
3. cuts the music window so the reveal lands on the drop and the end card owns the last bar;
4. puts every cut on a beat (or an exact half beat), adds a small zoom pulse on every beat inside a shot,
   whips (motion blur + whoosh peaking on the cut) and a gold flash + impact on the drop;
5. closes the last bar like a filter and dissolves the last frames into the first, so the loop restarts on a downbeat;
6. logs the plan's own sync error; `sync_check.py` then measures the finished file independently.

A template that does not fill its bars exactly, or has no shot starting on the drop, stops the render.
