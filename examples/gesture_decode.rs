//! Offline CTC probe. Pass a verified gesture TSV and encoder emissions JSON.
//! `cargo run --release --example gesture_decode -- <vocabulary.tsv> <frames.json>`
use obadh_engine::gesture::GestureLexicon;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vocabulary = args.next().ok_or("missing vocabulary TSV")?;
    let emissions = args.next().ok_or("missing emissions JSON")?;
    if args.next().is_some() {
        return Err("expected exactly two paths".into());
    }
    let model = GestureLexicon::from_tsv(&std::fs::read(vocabulary)?)
        .map_err(|error| format!("invalid gesture vocabulary: {error:?}"))?;
    let raw: Vec<Vec<f32>> = serde_json::from_slice(&std::fs::read(emissions)?)?;
    let frames: Vec<[f32; 27]> = raw
        .into_iter()
        .map(|row| {
            if row.len() != 65 {
                return Err(format!("expected 65 emissions, got {}", row.len()));
            }
            let mut frame = [0.0; 27];
            frame[..26].copy_from_slice(&row[..26]);
            frame[26] = row[64];
            Ok(frame)
        })
        .collect::<Result<_, _>>()?;
    let started = std::time::Instant::now();
    for candidate in model.decode(&frames, 100, 5) {
        println!(
            "{}\t{}\t{:.3}",
            candidate.spelling, candidate.word, candidate.score
        );
    }
    eprintln!("decode: {:?}", started.elapsed());
    Ok(())
}
