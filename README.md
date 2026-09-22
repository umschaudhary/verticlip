# verticlip

Turn long videos into captioned 9:16 clips, on your own machine, for free.

```
YouTube URL ─► ingest ─► transcribe ─► pick ─► render ─► publish ─► export
                yt-dlp    whisper      LLM     ffmpeg     APIs      submissions.csv
```

Everything runs locally: transcription is Whisper, highlight-picking is an Ollama
model on your machine, rendering is ffmpeg. No account, no API key, no per-minute
billing. The only network traffic is fetching your source and — if you switch
publishing on — uploading to platforms you authenticated yourself.

Every stage is idempotent and state lives in `data/ledger.db`, so a scheduler can
fire it repeatedly and it picks up where it left off.

---

## Before anything else: clip only what you have the right to clip

This tool uploads video. What you point it at is your responsibility, and "a
script did it" is not a defence.

- **Most YouTube videos are all-rights-reserved.** The default Standard YouTube
  Licence grants you no permission to re-upload, in whole or in part.
- Re-uploading someone else's video can earn a **Content ID claim** (monetisation
  redirected, regional blocks) or a **copyright strike** (three ends the channel).
- Legitimate sources: your own footage, **Creative Commons** licensed material,
  content you have written permission for, and **creator clipping programs**
  (Whop, Vyro, Clipster and similar) that explicitly grant the right to clip a
  specific asset. The campaign system here exists for exactly that case.
- Fair use / fair dealing is real but fact-specific. It is not blanket permission,
  and nothing in this repo evaluates it for you.

Provided under the MIT licence, with no responsibility for what you publish.

---

## Quickstart

```bash
git clone https://github.com/umschaudhary/verticlip && cd verticlip
./install.sh                      # macOS or Linux
ollama pull qwen2.5:7b-instruct   # the highlight picker
./clip                            # asks for a URL and the rest
```

`./clip` with no arguments walks you through it. **Nothing is published unless
you pass `--post`.**

```bash
./clip https://youtu.be/XYZ                      # 1 clip
./clip https://youtu.be/XYZ 6                    # 6 clips
./clip https://youtu.be/XYZ 6 -c pop -t none -l card
./clip --render                                  # re-render queued clips, new styles
./clip --status                                  # what's in the ledger
./clip --last                                    # open the newest run folder
```

Clips land in `data/runs/<timestamp>_<video-id>/` with a `clips.txt` and a
`manifest.json` recording what was cut and why. **That is the only place clips
are ever written** — one folder per run, nothing scattered elsewhere. A run that
produces nothing removes its own folder.

Under `./clip` sits `run.py`, if you want the stages directly:

```bash
python run.py --url URL --clips 6 --no-publish
python run.py --stage render          # one stage, repeatable
python run.py --skip transcribe       # everything except
python run.py --dry-run               # walk every publish gate, upload nothing
python run.py --loop 20               # every 20 minutes
```

---

## Requirements

| | |
|---|---|
| Python | 3.11+ |
| ffmpeg | **built with libass** — captions are burned in via the `subtitles` filter |
| Ollama | optional but recommended; the default highlight picker |

`ffmpeg -filters | grep subtitles` must print something. Homebrew's current
`ffmpeg` ships **without** libass; `brew install ffmpeg@7` gives you one that has
it, and clipper finds keg-only builds automatically. Set `$FFMPEG_BIN` or the
`tools:` block in `config.yaml` to force a specific binary.

---

## The options that matter

### Captions — `-c`

| | |
|---|---|
| `boxed` | 3-word line on an opaque band. Most legible over busy footage. **Default.** |
| `pop` | one big word at a time, scaling as it lands |
| `karaoke` | full line, active word colour-swaps with a glow |
| `lyric` | small lowercase with a soft bloom, centred |
| `none` | no burned-in captions |

**Name the font weight.** `font: Montserrat` resolves to the variable font's
*light* default and renders hairline-thin at any size — use `Montserrat Black`.
Check with `fc-list : family style`; a name fontconfig can't match falls back
silently. Colours are ASS `&HAABBGGRR` — blue-green-red, not RGB.

### Framing — `-l`

| | |
|---|---|
| `auto` | crop to 9:16 following the speaker; stacks two people when it sees them. **Default.** |
| `smart` | same, never stacks |
| `card` | whole frame as an inset panel over a blurred backdrop — nothing cropped, and it sits clear of the app's buttons |
| `center` | plain centre crop |
| `blur_bg` | whole frame, blurred backdrop, edge to edge |

A 9:16 crop of a 16:9 source discards **44% of the width**. `auto` accepts that to
fill the screen; `card` keeps everything and gives it margins instead.
`render.card.fill_pct` sets how much of the screen height the panel takes and
crops only as much as that requires.

### Choosing clips — `-p`

| | |
|---|---|
| `llm` | read the transcript and pick the good bits. **Default.** |
| `scenes` | start each clip on a detected shot change |
| `even` | space them across the runtime |

`scenes` and `even` skip transcription entirely when captions are off — much
faster, and the right mode for material where the words aren't the point.

---

## Configuration

`config.yaml` is commented throughout. What people change first:

```yaml
download:
  max_height: 1080        # a 9:16 crop keeps only height*0.5625 of the width,
                          # so 720p means a 2.7x upscale to 1080x1920

highlighter:
  provider: ollama        # ollama | anthropic | openai
  model: qwen2.5:7b-instruct
  max_parallel: 4         # transcript windows asked at once

limits:
  clips_per_source: 1
  min_clip_seconds: 18
  max_clip_seconds: 58

retention:
  delete_source_after_render: true   # sources are ~300 MB, clips ~20 MB
```

Hosted models are opt-in: set `CLIPPER_LLM_PROVIDER` plus a key in `.env` and
they override `config.yaml`, so a personal setup never has to be committed.

---

## Campaigns

A campaign is a folder with a `campaign.yaml` grouping sources, styling and
posting rules. It also records a payout rate, so `data/submissions.csv` can tell
you what each posted clip was worth. See `campaigns/example/`.

Two modes:

- **`mode: clip`** (default) — cut highlights from the campaign's own video.
- **`mode: edits`** — the campaign supplies a *track*, you supply the footage.
  Beat detection cuts shots to the music, with transitions, grading and lyrics on
  screen. Built for music clipping programs.

`./clip <url>` needs no campaign — it creates an `adhoc` one for you.

---

## Publishing

Off by default. Nothing uploads without `--post`.

| | |
|---|---|
| YouTube | Data API v3, OAuth. ~6 uploads/day on the default 10,000-unit quota. |
| Instagram | Graph API. Needs a Business/Creator account, a linked Facebook Page, and a public URL for the file (e.g. Cloudflare R2). |
| TikTok | Content Posting API. **Private-only until TikTok audits your app.** |

`.env.example` documents every variable. Set `publish.youtube.privacy: unlisted`
while testing — a bad clip posted publicly is not easily undone. `--dry-run`
walks every gate and prints the exact file, title and caption each platform would
receive, touching neither the network nor the queue.

> While your Google OAuth app is in **Testing** status, the refresh token expires
> after 7 days and unattended runs start failing silently. Publish the consent
> screen to production before relying on a schedule.

Whop/Vyro/Clipster have no submission API, so `data/submissions.csv` collects
every posted link ready to paste into their forms.

---

## Tests

```bash
python tests/test_pipeline.py     # ingest → render → queue → export, mocked LLM
python tests/test_edits.py        # beat detection, shot selection, render overrides
python tests/test_effects.py      # every transition and effect renders
```

All three run offline on a fresh clone. Fixtures are generated with ffmpeg on
first use — no binaries committed, nothing downloaded. The beat tracker is
checked against a click track at a known tempo, not merely for not crashing.

---

## Housekeeping

```bash
./cleanup --dry-run    # what would go
./cleanup              # delete rendered clips and sources, after confirming
./cleanup --all        # also transcripts and the ledger — a full reset
```

Your footage under `campaigns/*/footage/` is never touched.

---

## Layout

```
clip  cleanup  run.py        entry points
config.yaml                  pipeline settings
campaigns/<slug>/            campaign.yaml + footage/
clipper/
  config.py  ledger.py  binaries.py
  ingest.py  transcribe.py  highlight.py  beats.py  render.py  submissions.py
  publish/   youtube.py  instagram.py  tiktok.py
data/
  runs/<timestamp>/          every clip, plus clips.txt and manifest.json
  downloads/                 source video (deleted once its clips render)
  transcripts/  logs/  ledger.db
secrets/                     OAuth files (git-ignored)
```

---

## Status and known limits

**Working:** ingest, transcription, highlight-picking, the layout engine (speaker
tracking, two-person stacking, card framing, scene-aware crops), captions,
beat-cut edits mode, per-run output folders, YouTube upload.

**Not there yet:**

- **Facecam layout** — screen-share with a corner camera isn't detected.
- **MediaPipe** face detection aborts the process on some macOS builds, so
  detection uses OpenCV Haar cascades with profile sweeps. `CLIPPER_MEDIAPIPE=1`
  opts in; a subprocess probe refuses it if it would crash.
- **Instagram and TikTok publishers are written but unproven** — no live run yet.
- **Speed.** A 29-minute source takes ~16 minutes end to end, most of it Whisper.

## Licence

MIT — see `LICENSE`. Bundled third-party assets are listed in `NOTICE`.
