use raptorq::{ObjectTransmissionInformation, SourceBlockDecoder, SourceBlockEncoder};

const SYMBOL_SIZE: u16 = 8;
const SYMBOL_COUNTS: [usize; 3] = [10, 12, 50];

fn deterministic_data(symbol_count: usize) -> Vec<u8> {
    (0..symbol_count * SYMBOL_SIZE as usize)
        .map(|i| ((i * 31 + 17) & 0xff) as u8)
        .collect()
}

#[test]
fn source_only_round_trip_for_small_systematic_sizes() {
    for symbol_count in SYMBOL_COUNTS {
        let data = deterministic_data(symbol_count);
        let config = ObjectTransmissionInformation::new(0, SYMBOL_SIZE, 0, 1, 1);
        let encoder = SourceBlockEncoder::new(0, &config, &data);
        let mut decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);

        let result = decoder.decode(encoder.source_packets());

        assert_eq!(
            result,
            Some(data),
            "source-only failed for K={symbol_count}"
        );
    }
}

#[test]
fn repair_only_round_trip_for_small_systematic_sizes() {
    for symbol_count in SYMBOL_COUNTS {
        let data = deterministic_data(symbol_count);
        let config = ObjectTransmissionInformation::new(0, SYMBOL_SIZE, 0, 1, 1);
        let encoder = SourceBlockEncoder::new(0, &config, &data);
        let mut decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);

        let result = decoder.decode(encoder.repair_packets(0, symbol_count as u32 + 8));

        assert_eq!(
            result,
            Some(data),
            "repair-only failed for K={symbol_count}"
        );
    }
}

#[test]
#[should_panic(expected = "integer division result exceeds u32")]
fn oti_rejects_symbol_counts_that_would_wrap_u32() {
    ObjectTransmissionInformation::new(942_574_504_275, 1, 1, 1, 1);
}
