//! GPU/CPU readback integrity hash from DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e
//! host/torn.rs. The source's concurrent copy-queue suspect tracking is
//! not safe to enable until detached flights exist. The current serial
//! path checks complete aligned pieces before any landing and retries a
//! mismatched batch while no later title submission can run.

use std::sync::OnceLock;

/// `VICTORY_FETCH_CHECK=1`; off by default until a run has shown how often a
/// piece is caught (the frame-rate line says so) and what the hashes cost
/// (3-4 ms for 96 MB beside the merge, spread over the threads that land).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FETCH_CHECK").is_some_and(|v| v == "1"))
}

/// Of a result of the compare: whether the piece differs.
pub(super) fn flagged(result: u32) -> bool {
    result & 1 != 0
}

/// Of a result of the compare: the piece's hash, 31 bits of it.
pub(super) fn folded(result: u32) -> u32 {
    result >> 1
}

/// The hash the compare on the card ([`CARD_COMPARE`]) makes of a piece
/// that holds `bytes` (a multiple of 16 of them), as [`folded`].
pub(super) fn hash(bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the feature is there.
        return unsafe { hash_avx2(bytes) };
    }
    hash_plain(bytes)
}

const MUL: [u32; 4] = [0x0100_0193, 0x9E37_79B1, 0x85EB_CA77, 0xC2B2_AE3D];
const SEED: [u32; 4] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];

fn step(h: &mut [u32; 4], with: [u32; 4]) {
    let x: [u32; 4] = std::array::from_fn(|i| (h[i] ^ with[i]).wrapping_mul(MUL[i]));
    *h = std::array::from_fn(|i| x[(i + 1) % 4] ^ (x[i] >> 15));
}

/// The 64 threads' hashes as one, folded to 31 bits.
fn finish(parts: &[[u32; 4]; 64]) -> u32 {
    let mut h = parts[0];
    for part in &parts[1..] {
        step(&mut h, *part);
    }
    let f = h[0] ^ h[1].wrapping_mul(3) ^ h[2].wrapping_mul(5) ^ h[3].wrapping_mul(7);
    f & 0x7FFF_FFFF
}

fn seeds() -> [[u32; 4]; 64] {
    std::array::from_fn(|thread| SEED.map(|s| s.wrapping_add(thread as u32)))
}

fn tail(parts: &mut [[u32; 4]; 64], bytes: &[u8]) {
    for (thread, chunk) in bytes.as_chunks::<16>().0.iter().enumerate() {
        let with = std::array::from_fn(|i| u32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]));
        step(&mut parts[thread], with);
    }
}

fn hash_plain(bytes: &[u8]) -> u32 {
    let mut parts = seeds();
    for row in bytes.chunks(1024) {
        tail(&mut parts, row);
    }
    finish(&parts)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hash_avx2(bytes: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let mut parts = seeds();
    let (rows, rest) = bytes.as_chunks::<1024>();
    let mul = _mm256_setr_epi32(
        MUL[0] as i32, MUL[1] as i32, MUL[2] as i32, MUL[3] as i32, MUL[0] as i32, MUL[1] as i32, MUL[2] as i32, MUL[3] as i32,
    );
    let state = parts.as_mut_ptr().cast::<__m256i>();
    for row in rows {
        let from = row.as_ptr().cast::<__m256i>();
        for k in 0..32 {
            // SAFETY: two threads' hashes of 16 bytes each, and their 32 of
            // the row's 1,024 bytes.
            unsafe {
                let h = _mm256_xor_si256(_mm256_loadu_si256(state.add(k)), _mm256_loadu_si256(from.add(k)));
                let x = _mm256_mullo_epi32(h, mul);
                // y, z, w, x of each.
                let turned = _mm256_shuffle_epi32::<0b00_11_10_01>(x);
                _mm256_storeu_si256(state.add(k), _mm256_xor_si256(turned, _mm256_srli_epi32::<15>(x)));
            }
        }
    }
    tail(&mut parts, rest);
    finish(&parts)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wide_hash_is_the_plain_one() {
        let bytes: Vec<u8> = (0..65_536u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for n in [0usize, 16, 32, 1008, 1024, 1040, 5000 & !15, 65_536] {
            assert_eq!(hash(&bytes[..n]), hash_plain(&bytes[..n]), "{n} bytes");
        }
        let mut other = bytes.clone();
        other[40_000] ^= 1;
        assert_ne!(hash(&bytes), hash(&other));
    }
}
