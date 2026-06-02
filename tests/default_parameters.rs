use raptorq::{ObjectTransmissionInformation, calculate_block_offsets};

#[test]
fn defaults_keep_one_mib_in_one_solver_supported_block() {
    let data = vec![0u8; 1024 * 1024];
    let config = ObjectTransmissionInformation::with_defaults(data.len() as u64, 1024);

    assert_eq!(config.source_blocks(), 1);

    let largest_block_symbols = calculate_block_offsets(&data, &config)
        .into_iter()
        .map(|(start, end)| (end - start) / config.symbol_size() as usize)
        .max()
        .unwrap();
    assert_eq!(largest_block_symbols, 1024);
}

#[test]
#[should_panic(expected = "default encoding parameters require 256 source blocks")]
fn defaults_reject_objects_requiring_wrapped_source_block_count() {
    ObjectTransmissionInformation::with_defaults(510_228_481, 1024);
}
