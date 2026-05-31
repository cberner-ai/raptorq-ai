use raptorq::{
    EncodingPacket, ObjectTransmissionInformation, SourceBlockDecoder, SourceBlockEncoder,
    SourceBlockEncodingPlan,
};

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
fn contradictory_redundant_repair_rows_do_not_decode() {
    let symbol_count = 10;
    let data = deterministic_data(symbol_count);
    let config = ObjectTransmissionInformation::new(0, SYMBOL_SIZE, 0, 1, 1);
    let encoder = SourceBlockEncoder::new(0, &config, &data);
    let repair_count = symbol_count as u32 + 12;

    let repair_packets = encoder.repair_packets(0, repair_count);
    let mut good_decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);
    assert_eq!(
        good_decoder.decode(repair_packets.clone()),
        Some(data.clone())
    );

    let mut contradictory_packets = repair_packets;
    let last = contradictory_packets.last_mut().unwrap();
    let mut corrupt_payload = last.data().to_vec();
    corrupt_payload[0] ^= 0xff;
    *last = EncodingPacket::new(last.payload_id().clone(), corrupt_payload);

    let mut decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);
    assert_eq!(decoder.decode(contradictory_packets), None);
}

#[test]
fn corrupt_exact_no_hdpc_repair_set_does_not_decode() {
    let symbol_count = 10;
    let data = deterministic_data(symbol_count);
    let config = ObjectTransmissionInformation::new(0, SYMBOL_SIZE, 0, 1, 1);
    let encoder = SourceBlockEncoder::new(0, &config, &data);
    let repair_count = 20;

    let repair_packets = encoder.repair_packets(0, repair_count);
    let mut good_decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);
    assert_eq!(
        good_decoder.decode(repair_packets.clone()),
        Some(data.clone())
    );

    let mut corrupt_packets = repair_packets;
    let first = corrupt_packets.first_mut().unwrap();
    let mut corrupt_payload = first.data().to_vec();
    corrupt_payload[0] ^= 0xff;
    *first = EncodingPacket::new(first.payload_id().clone(), corrupt_payload);

    let mut decoder = SourceBlockDecoder::new(0, &config, data.len() as u64);
    assert_eq!(decoder.decode(corrupt_packets), None);
}

#[test]
fn large_source_block_encoding_plan_generation_does_not_panic() {
    let _plan = SourceBlockEncodingPlan::generate(5_000);
}

#[cfg(not(feature = "std"))]
#[test]
fn large_zero_no_std_source_block_encoder_does_not_panic() {
    let data = vec![0; 5_000 * SYMBOL_SIZE as usize];
    let config = ObjectTransmissionInformation::new(0, SYMBOL_SIZE, 0, 1, 1);
    let encoder = SourceBlockEncoder::new(0, &config, &data);

    assert_eq!(encoder.source_packets().len(), 5_000);
}

#[test]
#[should_panic(expected = "integer division result exceeds u32")]
fn oti_rejects_symbol_counts_that_would_wrap_u32() {
    ObjectTransmissionInformation::new(942_574_504_275, 1, 1, 1, 1);
}
