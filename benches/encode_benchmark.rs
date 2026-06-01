use rand::RngExt;
use raptorq::{ObjectTransmissionInformation, SourceBlockEncoder, SourceBlockEncodingPlan};
use std::time::Instant;

const TARGET_TOTAL_BYTES: usize = 8 * 1024 * 1024;
const SYMBOL_COUNTS: [usize; 4] = [10, 100, 250, 500];
const CI_TARGET_TOTAL_BYTES: usize = 2 * 1024 * 1024;
const CI_SYMBOL_COUNTS: [usize; 2] = [10, 100];

fn black_box(value: u64) {
    if value == rand::rng().random() {
        println!("{value}");
    }
}

fn ci_mode_enabled() -> bool {
    std::env::args().any(|arg| arg == "--ci")
}

fn benchmark(
    symbol_size: u16,
    pre_plan: bool,
    target_total_bytes: usize,
    symbol_counts: &[usize],
) -> u64 {
    let mut black_box_value = 0;
    for &symbol_count in symbol_counts.iter() {
        let elements = symbol_count * symbol_size as usize;
        let mut data: Vec<u8> = vec![0; elements];
        for byte in data.iter_mut() {
            *byte = rand::rng().random();
        }

        let plan = if pre_plan {
            Some(SourceBlockEncodingPlan::generate(symbol_count as u16))
        } else {
            None
        };

        let now = Instant::now();
        let iterations = (target_total_bytes / elements).max(1);
        let config = ObjectTransmissionInformation::new(0, symbol_size, 0, 1, 1);
        for _ in 0..iterations {
            let encoder = if let Some(ref plan) = plan {
                SourceBlockEncoder::with_encoding_plan(1, &config, &data, plan)
            } else {
                SourceBlockEncoder::new(1, &config, &data)
            };
            let packets = encoder.repair_packets(0, 1);
            black_box_value += packets[0].data()[0] as u64;
        }
        let elapsed = now.elapsed();
        let elapsed = elapsed.as_secs() as f64 + elapsed.subsec_millis() as f64 * 0.001;
        let throughput = (elements * iterations * 8) as f64 / 1024.0 / 1024.0 / elapsed;
        let processed_mib = (elements * iterations) as f64 / 1024.0 / 1024.0;
        println!(
            "symbol count = {}, encoded {:.2} MB in {:.3}secs, throughput: {:.1}Mbit/s",
            symbol_count, processed_mib, elapsed, throughput
        );
    }
    black_box_value
}

fn main() {
    let symbol_size = 1280;
    let (target_total_bytes, symbol_counts) = if ci_mode_enabled() {
        println!("Running CI benchmark subset");
        (CI_TARGET_TOTAL_BYTES, CI_SYMBOL_COUNTS.as_slice())
    } else {
        (TARGET_TOTAL_BYTES, SYMBOL_COUNTS.as_slice())
    };

    println!("Symbol size: {symbol_size} bytes (without pre-built plan)");
    black_box(benchmark(
        symbol_size,
        false,
        target_total_bytes,
        symbol_counts,
    ));
    println!();
    println!("Symbol size: {symbol_size} bytes (with pre-built plan)");
    black_box(benchmark(
        symbol_size,
        true,
        target_total_bytes,
        symbol_counts,
    ));
}
