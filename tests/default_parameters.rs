use raptorq::{Encoder, ObjectTransmissionInformation, calculate_block_offsets};

#[test]
fn defaults_split_one_mib_into_solver_supported_blocks() {
    let data = vec![0u8; 1024 * 1024];
    let encoder = Encoder::with_defaults(&data, 1024);
    let config = encoder.get_config();

    assert_eq!(config.source_blocks(), 2);

    let largest_block_symbols = calculate_block_offsets(&data, &config)
        .into_iter()
        .map(|(start, end)| (end - start) / config.symbol_size() as usize)
        .max()
        .unwrap();
    assert_eq!(largest_block_symbols, 512);
}

#[test]
#[should_panic(expected = "default encoding parameters require 256 source blocks")]
fn defaults_reject_objects_requiring_wrapped_source_block_count() {
    ObjectTransmissionInformation::with_defaults(248_064_001, 1024);
}
