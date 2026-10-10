//! Offline human-trace decoder probe. Input JSON is `[target_roman, [[65 logits]; 32]]` rows.
//! This benchmark does not claim Bengali human-gesture accuracy when run with
//! an English swipe corpus; it measures the shared Roman search channel only.
use obadh_engine::gesture::GestureLexicon;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vocabulary = args.next().ok_or("missing vocabulary TSV")?;
    let emissions = args.next().ok_or("missing JSON file")?;
    let model = GestureLexicon::from_tsv(&std::fs::read(vocabulary)?)
        .map_err(|error| format!("invalid vocabulary: {error:?}"))?;
    let samples: Vec<(String, Vec<Vec<f32>>)> = serde_json::from_slice(&std::fs::read(emissions)?)?;
    if samples.is_empty() {
        return Err("empty sample set".into());
    }
    let mut top1 = 0;
    let mut top3 = 0;
    let mut timings = Vec::with_capacity(samples.len());
    let mut misses = Vec::new();
    for (target, raw) in &samples {
        let frames: Vec<[f32; 27]> = raw
            .iter()
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
        let candidates = model.decode(&frames, 100, 3);
        timings.push(started.elapsed().as_secs_f64() * 1_000.0);
        let target = target.to_ascii_lowercase();
        if candidates
            .first()
            .is_some_and(|candidate| candidate.spelling == target)
        {
            top1 += 1;
        } else if misses.len() < 20 {
            misses.push((
                target.clone(),
                candidates
                    .first()
                    .map(|candidate| candidate.spelling.clone()),
            ));
        }
        if candidates
            .iter()
            .any(|candidate| candidate.spelling == target)
        {
            top3 += 1;
        }
    }
    timings.sort_by(f64::total_cmp);
    let n = samples.len();
    println!(
        "n={n} top1={:.3} top3={:.3} p95_decode_ms={:.2}",
        top1 as f64 / n as f64,
        top3 as f64 / n as f64,
        timings[(n * 95 / 100).min(n - 1)]
    );
    println!("first misses: {misses:?}");
    Ok(())
}
