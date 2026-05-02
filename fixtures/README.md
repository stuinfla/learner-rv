# Test Fixtures

Hermetic test data for learn-rs integration and unit tests. All files are
checked in except binary audio (generated on demand).

## Files

| File | Purpose |
|------|---------|
| `short.vtt` | Three-cue WebVTT file for VTT parser integration tests |
| `short.info.json` | Fake yt-dlp `*.info.json` payload matching the VTT fixture |
| `sample.mp3` | **NOT checked in** — generated on demand (see below) |

## Generating `sample.mp3`

`sample.mp3` is a 3-second 1 kHz sine wave at 16 kHz mono. It is not
committed to the repository because binary blobs inflate `git clone` time and
provide no signal in `git diff`.

Run the helper script to generate it before running tests that require audio:

```bash
bash tests/generate_fixtures.sh
```

Requirements: `ffmpeg` must be on PATH (install via `brew install ffmpeg` on
macOS or `apt-get install ffmpeg` on Debian/Ubuntu).

The script is idempotent — it skips generation if `fixtures/sample.mp3`
already exists.
