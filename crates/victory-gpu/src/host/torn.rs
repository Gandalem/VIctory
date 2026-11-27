//! GPU/CPU readback integrity hash from DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e
//! host/torn.rs. The source's concurrent copy-queue suspect tracking is
//! not safe to enable until detached flights exist. The current serial
//! path checks complete aligned pieces before any landing and retries a
//! mismatched batch while no later title submission can run.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

/// `VICTORY_FETCH_CHECK=1`; off by default until a run has shown how often a
/// piece is caught (the frame-rate line says so) and what the hashes cost
/// (3-4 ms for 96 MB beside the merge, spread over the threads that land).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FETCH_CHECK").is_some_and(|v| v == "1"))
}

/// A piece: its copy's number, and the piece.
pub(super) type Piece = (u64, u32);

/// Checked results: accepted pieces, mismatches, and repaired suspects.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The pieces whose last fetch was not what a flight ended with.
#[derive(Default)]
pub(super) struct Suspects(HashSet<Piece>);

impl Suspects {
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// A released Shadow cannot be fetched again. Its monotonic ID is
    /// never reused, even if its allocator or title address is reused.
    pub(super) fn retain_live(&mut self, ids: impl IntoIterator<Item = u64>) {
        if !self.0.is_empty() {
            let live: HashSet<_> = ids.into_iter().collect();
            self.0.retain(|(id, _)| live.contains(id));
        }
    }

    /// Whether a flight that may write `piece` fetches it, its compare
    /// having flagged it or not.
    pub(super) fn wanted(&self, piece: Piece, flagged: bool) -> bool {
        flagged || (!self.0.is_empty() && self.0.contains(&piece))
    }

    /// `piece` was fetched; `same`: it was what the last of the flights
    /// landed ended with.
    pub(super) fn fetched(&mut self, piece: Piece, same: bool) {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        if same {
            if self.0.remove(&piece) {
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.0.insert(piece);
            STATS[1].fetch_add(1, Ordering::Relaxed);
        }
    }
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

    /// A model of one piece. The card runs flights in order; each writes the
    /// piece several times and its compare flags it if it ends otherwise
    /// than the flight before. Each landing fetches what is wanted, on a
    /// queue beside the drawing one: the fetch sees the piece as it is while
    /// a later flight is at some step. Returns what the CPU's side holds in
    /// the end, and the fetches.
    fn landed(flights: &[&[u32]], seen: &[(usize, usize)], check: bool) -> (u32, u32) {
        let piece: Piece = (1, 0);
        let mut suspects = Suspects::default();
        let (mut held, mut ended, mut fetches) = (0, 0, 0);
        for (f, steps) in flights.iter().enumerate() {
            let before = ended;
            ended = *steps.last().expect("a step");
            let flagged = ended != before;
            if if check { !suspects.wanted(piece, flagged) } else { !flagged } {
                continue;
            }
            fetches += 1;
            // While flight `at` is after its step `step`.
            let (at, step) = seen[f];
            assert!(at >= f);
            held = flights[at][step];
            suspects.fetched(piece, held == ended);
        }
        (held, fetches)
    }

    /// A counter that each flight clears and counts up to the same: the
    /// first flight's fetch sees it cleared by the second, whose compare
    /// flags nothing.
    #[test]
    fn a_piece_caught_in_between_is_fetched_again() {
        let count: &[u32] = &[0, 3, 5];
        // The first is fetched while the second has cleared it.
        assert_eq!(landed(&[count, count], &[(1, 0), (1, 2)], false), (0, 1), "the CPU's side keeps what was in between");
        assert_eq!(landed(&[count, count], &[(1, 0), (1, 2)], true), (5, 2));
        // And again, while the third was at it; the fourth finds it whole.
        let four = [count; 4];
        assert_eq!(landed(&four, &[(1, 0), (2, 1), (3, 0), (3, 2)], true), (5, 4));
        // Fetched whole: no more fetches than without the check.
        assert_eq!(landed(&four, &[(0, 2), (1, 2), (2, 2), (3, 2)], true), (5, 1));
        assert_eq!(landed(&four, &[(0, 2), (1, 2), (2, 2), (3, 2)], false), (5, 1));
    }

    #[test]
    fn suspects_are_kept_per_copy_and_pruned_only_when_released_or_verified() {
        let mut suspects = Suspects::default();
        suspects.fetched((11, 2), false);
        suspects.fetched((12, 2), false);
        assert!(suspects.wanted((11, 2), false));
        assert!(!suspects.wanted((11, 3), false));
        suspects.retain_live([11, 13]);
        assert!(suspects.wanted((11, 2), false));
        assert!(!suspects.wanted((12, 2), false));
        assert!(!suspects.wanted((13, 2), false));
        suspects.fetched((11, 2), true);
        assert!(suspects.is_empty());
    }

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
