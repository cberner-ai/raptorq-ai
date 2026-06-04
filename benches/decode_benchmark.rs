mod ci_symbol_counts;

use ci_symbol_counts::CI_DECODE_MIXED_ONE_REPAIR_SYMBOL_COUNTS;
use ci_symbol_counts::CI_DECODE_SYMBOL_COUNTS;
use rand::RngExt;
use raptorq::{
    EncodingPacket, ObjectTransmissionInformation, SourceBlockDecoder, SourceBlockEncoder,
};
use std::time::{Duration, Instant};

const TARGET_TOTAL_BYTES: usize = 128 * 1024 * 1024;
// Keep repair-only decode rows bounded; larger overhead solves time out or hit solver limits.
const SYMBOL_COUNTS: [usize; 5] = [10, 100, 250, 500, 1000];
const CI_TARGET_TOTAL_BYTES: usize = 8 * 1024 * 1024;

fn black_box(value: u64) {
    if value == rand::rng().random() {
        println!("{value}");
    }
}

fn ci_mode_enabled() -> bool {
    std::env::args().any(|arg| arg == "--ci")
}

fn elapsed_seconds(elapsed: Duration) -> f64 {
    elapsed.as_nanos() as f64 / 1_000_000_000.0
}

fn mixed_one_repair_packets(
    encoder: &SourceBlockEncoder,
    symbol_count: usize,
    iterations: usize,
) -> Vec<EncodingPacket> {
    let mut source_packets = encoder.source_packets();
    source_packets.pop();
    let repair_packets = encoder.repair_packets(0, iterations as u32);

    let mut packets = Vec::with_capacity(iterations * symbol_count);
    for repair_packet in repair_packets {
        packets.extend(source_packets.iter().cloned());
        packets.push(repair_packet);
    }
    packets
}

fn benchmark(
    symbol_size: u16,
    overhead: f64,
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

        let iterations = (target_total_bytes / elements).max(1);
        let config = ObjectTransmissionInformation::new(0, symbol_size, 0, 1, 1);
        let encoder = SourceBlockEncoder::new(1, &config, &data);
        let elements_and_overhead = (symbol_count as f64 * (1.0 + overhead)) as u32;
        let mut packets = encoder.repair_packets(0, iterations as u32 * elements_and_overhead);
        let now = Instant::now();
        for _ in 0..iterations {
            let mut decoder = SourceBlockDecoder::new(1, &config, elements as u64);
            let start = packets.len() - elements_and_overhead as usize;
            if let Some(result) = decoder.decode(packets.drain(start..)) {
                black_box_value += result[0] as u64;
            }
        }
        let elapsed = elapsed_seconds(now.elapsed());
        let throughput = (elements * iterations * 8) as f64 / 1024.0 / 1024.0 / elapsed;
        let processed_mib = (elements * iterations) as f64 / 1024.0 / 1024.0;
        println!(
            "symbol count = {}, decoded {:.2} MB in {:.9}secs using {:.1}% overhead, throughput: {:.3}Mbit/s",
            symbol_count,
            processed_mib,
            elapsed,
            100.0 * overhead,
            throughput
        );
    }

    black_box_value
}

fn benchmark_mixed_one_repair(
    symbol_size: u16,
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

        let iterations = (target_total_bytes / elements).max(1);
        let config = ObjectTransmissionInformation::new(0, symbol_size, 0, 1, 1);
        let encoder = SourceBlockEncoder::new(1, &config, &data);
        let mut packets = mixed_one_repair_packets(&encoder, symbol_count, iterations);
        let now = Instant::now();
        for _ in 0..iterations {
            let mut decoder = SourceBlockDecoder::new(1, &config, elements as u64);
            let start = packets.len() - symbol_count;
            if let Some(result) = decoder.decode(packets.drain(start..)) {
                black_box_value += result[0] as u64;
            }
        }
        let elapsed = elapsed_seconds(now.elapsed());
        let throughput = (elements * iterations * 8) as f64 / 1024.0 / 1024.0 / elapsed;
        let processed_mib = (elements * iterations) as f64 / 1024.0 / 1024.0;
        println!(
            "symbol count = {}, decoded {:.2} MB in {:.9}secs using 0.0% overhead, throughput: {:.3}Mbit/s",
            symbol_count, processed_mib, elapsed, throughput
        );
    }

    black_box_value
}

fn main() {
    let symbol_size = 1280;
    let (target_total_bytes, symbol_counts) = if ci_mode_enabled() {
        println!("Running CI benchmark subset");
        (CI_TARGET_TOTAL_BYTES, CI_DECODE_SYMBOL_COUNTS.as_slice())
    } else {
        (TARGET_TOTAL_BYTES, SYMBOL_COUNTS.as_slice())
    };

    println!("Symbol size: {symbol_size} bytes");
    black_box(benchmark(
        symbol_size,
        0.0,
        target_total_bytes,
        symbol_counts,
    ));
    println!();
    black_box(benchmark(
        symbol_size,
        0.05,
        target_total_bytes,
        symbol_counts,
    ));
    println!();
    println!("Symbol size: {symbol_size} bytes (mixed one-repair)");
    black_box(benchmark_mixed_one_repair(
        symbol_size,
        target_total_bytes,
        CI_DECODE_MIXED_ONE_REPAIR_SYMBOL_COUNTS.as_slice(),
    ));
}
