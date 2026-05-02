//! Audio decoding: shell out to `ffmpeg` to produce 16 kHz mono f32le PCM.

use camino::Utf8Path;
use learn_core::{LearnError, Result};
use std::process::Command;
use tracing::debug;

/// Decode an audio file (any format ffmpeg supports) to 16 kHz mono f32 samples
/// by piping `ffmpeg -i <input> -ar 16000 -ac 1 -f f32le -` to stdout.
pub fn decode_to_pcm(audio_path: &Utf8Path) -> Result<Vec<f32>> {
    debug!(path = %audio_path, "decoding audio via ffmpeg");

    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            audio_path.as_str(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-",
        ])
        .output()
        .map_err(|e| LearnError::Transcribe(format!("ffmpeg spawn failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LearnError::Transcribe(format!(
            "ffmpeg exited {}: {stderr}",
            output.status
        )));
    }

    Ok(parse_f32le_bytes(&output.stdout))
}

/// Convert a raw f32le byte buffer to `Vec<f32>`.
///
/// Bytes that don't form a complete f32 (i.e. trailing partial sample) are dropped.
pub fn parse_f32le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_bytes_returns_empty() {
        assert!(parse_f32le_bytes(&[]).is_empty());
    }

    #[test]
    fn parse_partial_bytes_drops_trailing() {
        // 5 bytes: one complete f32 (4 bytes) + 1 trailing byte
        let mut data = 1.0_f32.to_le_bytes().to_vec();
        data.push(0xAB);
        let result = parse_f32le_bytes(&data);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0_f32);
    }

    #[test]
    fn parse_known_byte_pattern() {
        // 0.0 as f32le = [0x00, 0x00, 0x00, 0x00]
        // 1.0 as f32le = [0x00, 0x00, 0x80, 0x3F]
        let mut bytes = 0.0_f32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        let samples = parse_f32le_bytes(&bytes);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0.0_f32);
        assert_eq!(samples[1], 1.0_f32);
    }

    #[test]
    fn parse_negative_and_fractional() {
        let values: &[f32] = &[-1.5, 0.25, f32::MAX];
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let result = parse_f32le_bytes(&bytes);
        assert_eq!(result.len(), 3);
        for (a, b) in result.iter().zip(values.iter()) {
            assert_eq!(a, b);
        }
    }
}
