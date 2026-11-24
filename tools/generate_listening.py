"""Re-record the level test's listening clips (listen1-5.mp3) from tools/listening_scripts.json.

Voices: Kokoro (open-source, kokoro-onnx). Needs, once:
  pip install kokoro-onnx soundfile numpy   ·   ffmpeg on the PATH
  kokoro-v1.0.onnx + voices-v1.0.bin from https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
Run from the repo root:  python tools/generate_listening.py path/to/kokoro-v1.0.onnx path/to/voices-v1.0.bin [listen3 ...]
Then bump the ?v= number next to item.audio in index.html, so browsers fetch the new files."""
import json, re, subprocess, sys
import numpy as np
from kokoro_onnx import Kokoro

SR = 24000
TARGET_DB = -18.0                      # speech loudness of the clips (dBFS RMS over the voiced parts)

def trim(a, thr=0.008):
    i = np.where(np.abs(a) > thr)[0]
    return a[max(0, i[0] - 240): i[-1] + 480] if len(i) else a

def speech_db(a):
    v = a[np.abs(a) > 0.01]
    return 20 * np.log10(np.sqrt(np.mean(v ** 2))) if len(v) else -99

def silence(sec):
    return np.zeros(int(sec * SR), np.float32)

def main():
    model, voices = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3:])
    k = Kokoro(model, voices)
    clips = json.load(open("tools/listening_scripts.json", encoding="utf-8"))["clips"]
    for name, c in clips.items():
        if only and name not in only:
            continue
        parts = [silence(0.4)]
        for n, (who, voice, text) in enumerate(c["lines"]):
            if n:
                parts.append(silence(c["turn_pause"] - c["pause"]))
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
                # "fix_phonemes" swaps sounds a voice gets wrong, e.g. the flapped t in "writing"
                # (ɹˈaɪɾɪŋ), which some voices make sound like "rioting"
                fixes = c.get("fix_phonemes", {})
                ph = k.tokenizer.phonemize(sentence, lang="en-us")
                if any(f in ph for f in fixes):
                    for bad, good in fixes.items():
                        ph = ph.replace(bad, good)
                    a, _ = k.create(ph, voice=voice, speed=c["speed"], lang="en-us", is_phonemes=True)
                else:
                    a, _ = k.create(sentence, voice=voice, speed=c["speed"], lang="en-us")
                parts += [trim(np.asarray(a, np.float32)), silence(c["pause"])]
        audio = np.concatenate(parts + [silence(0.4)])
        audio *= 10 ** ((TARGET_DB - speech_db(audio)) / 20)
        audio = np.clip(audio, -0.98, 0.98)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-c:a", "libmp3lame", "-b:a", "96k", "-ar", str(SR), "-ac", "1", name + ".mp3"],
                       input=audio.astype(np.float32).tobytes(), check=True)
        print(name, round(len(audio) / SR, 1), "s")

if __name__ == "__main__":
    main()
