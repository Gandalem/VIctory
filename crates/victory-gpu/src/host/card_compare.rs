//! Synchronous GPU changed-piece compare, adapted from the reference renderer.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, host.rs (CARD_COMPARE, card_pass,
//! make_card_work), shadow.rs (CardSpan, range helpers and tests), host/torn.rs
//! (CPU hash). The optional two-sided Shadow path keeps existing CPU
//! copy-back and fences; optional relay merge of declared picked ranges
//! uses the same serial submission contract, not background landing.

use super::*;
const PIECE: usize = 64 << 10;

const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 parts[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_GroupID, uint thread : SV_GroupIndex) {
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) + thread;
    uint at = lo + thread * 16u;
    // Eight loads at once: one after the other, each waits for memory.
    [loop]
    for (; at + 7168u < hi; at += 8192u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 1024u);
        uint4 v2 = now.Load4(at + 2048u);
        uint4 v3 = now.Load4(at + 3072u);
        uint4 v4 = now.Load4(at + 4096u);
        uint4 v5 = now.Load4(at + 5120u);
        uint4 v6 = now.Load4(at + 6144u);
        uint4 v7 = now.Load4(at + 7168u);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    parts[thread] = h;
    GroupMemoryBarrierWithGroupSync();
    if (thread != 0u) return;
    h = parts[0];
    [loop]
    for (uint i = 1u; i < 64u; ++i) {
        h = (h ^ parts[i]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        // The flag, and above it 31 bits of the hash (`torn`).
        uint folded = h.x ^ (h.y * 3u) ^ (h.z * 5u) ^ (h.w * 7u);
        differs.Store(out_at + id.x * 4u, (folded << 1u) | (any(h != hashes.Load4(piece * 16u)) ? 1u : 0u));
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CardSpan {
    from: u32,
    to: u32,
    first: u32,
    pieces: u32,
    skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// The offsets hashed for a card's buffer of `size`
/// bytes for title memory from `start` on.
fn card_hashed_of(start: usize, size: usize, piece: usize) -> (usize, usize) {
    let (lo, hi) = piece_at(start, size, piece);
    (lo & !15, hi.next_multiple_of(16))
}

fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::new();
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// One pass of [`CARD_COMPARE`] over a span's pieces, into `differs` from
/// dword `at` on (`mode` as the shader's `compare`): a group per piece, in
/// parts a dispatch can hold.
///
/// # Safety
/// The list is open with the compare's root signature, pipeline and buffers
/// set, and the pieces lie in the buffer.
unsafe fn card_pass(list: &ID3D12GraphicsCommandList, span: &CardSpan, size: u32, at: u32, mode: u32) {
    const MOST: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    let mut done = 0;
    while done < span.pieces {
        let n = (span.pieces - done).min(MOST);
        let constants = [span.first + done, span.skew, size, n, (at + done) * 4, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
        }
        done += n;
    }
}

/// Compile the original compute shader, even when CI has no D3D12 adapter.
fn compile_compare() -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: static source and names live through the call; the output is ours.
    unsafe {
        D3DCompile(
            CARD_COMPARE.as_ptr().cast(), CARD_COMPARE.len(), s!("card compare"),
            None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

#[test]
fn reference_card_compare_compiles() {
    let blob = compile_compare().expect("compile reference CARD_COMPARE");
    // SAFETY: the compiled blob is live.
    assert!(unsafe { blob.GetBufferSize() } > 0);
}

/// Joined runs hold every piece of the runs given, once, and no other.
#[test]
fn piece_runs_join_without_losing_or_adding_pieces() {
    let run = |first: u32, pieces: u32| CardSpan {
        from: first * 65536,
        to: (first + pieces) * 65536,
        first,
        pieces,
        skew: 7,
    };
    let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
    let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let joined = joined_piece_runs(runs);
    let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
    assert_eq!(got, wanted);
    assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
    assert!(joined.iter().all(|r| r.skew == 7));
    assert!(joined_piece_runs(Vec::new()).is_empty());
}

/// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
/// in host.rs), before it widens them to 16 bytes.
#[cfg(test)]
fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
    let piece = span.first + i;
    let lo = piece * 65536;
    let lo = lo.saturating_sub(span.skew);
    let hi = (piece + 1) * 65536 - span.skew;
    (lo.max(span.from), hi.min(span.to))
}

/// The pieces of a span cover it without gaps, and the shader's
/// arithmetic names the same bytes as ours.
#[test]
fn the_shader_and_the_merge_cut_the_same_pieces() {
    for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
        let len = 5 * PIECE + 4096;
        for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
            let span = card_span_of(start, from, to);
            let mut at = from;
            for i in 0..span.pieces {
                let (lo, hi) = card_piece_of(start, len, &span, i);
                assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                assert!(hi > lo && hi - lo <= PIECE);
                assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                at = hi;
            }
            assert_eq!(at, to, "{start:#x} {from} {to}");
        }
    }
}

/// Changed ranges are joined where they overlap or touch, and only there.
/// What is fetched of a piece to hash it is what the shader hashes.
#[test]
fn a_piece_is_hashed_here_as_on_the_card() {
    for start in [0usize, 0x10_0000, 0xd_5edb_c780, 0x12_3456_7008] {
        for size in [1usize << 16, 5 << 16, 0x20_0000] {
            let span = card_span_of(start, 0, size);
            for i in 0..span.pieces {
                // As `CARD_COMPARE` has it.
                let (piece, skew, size) = (span.first + i, span.skew, size as u32);
                let lo = (piece * 65536).saturating_sub(skew);
                let hi = ((piece + 1) * 65536 - skew).min(size);
                let shader = (lo & !15, (hi + 15) & !15);
                let here = card_hashed_of(start, size as usize, piece as usize);
                assert_eq!((here.0 as u32, here.1 as u32), shader, "start {start:#x} size {size:#x} piece {piece}");
            }
        }
    }
}

#[cfg(test)]
fn uav_buffer(device: &ID3D12Device, size: u64) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size,
        Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a valid UAV buffer description and writable output pointer.
    unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut buffer).unwrap();
    }
    buffer.unwrap()
}

/// The source's make_card_work PSO setup, without the production flight queue.
fn compare_pipeline(device: &ID3D12Device) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let code = compile_compare()?;
    // SAFETY: the shader blob remains live and includes the root signature;
    // the borrowed root pointer is not released by the PSO description.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok((root, pso))
    }
}

fn compare_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: the caller records into an open list; a null UAV means all UAVs.
    unsafe { list.ResourceBarrier(&[barrier]) };
}


fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// Reference CardWork::fetched: a shared growable readback, packed by
/// changed range rather than sized for every live Shadow. Only the
/// submission lock permits GPU/CPU access to its contents; a cloned owner
/// keeps the mapping live if concurrent admission grows the pool.
#[derive(Clone)]
pub(super) struct CardFetch {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

const FETCH_GRANULARITY: u64 = 16 << 20;

impl Host {
    /// Adapt the source's 16 MiB growth rule. Do not discard a working
    /// allocation when growth fails. Release the pool lock before recording
    /// or waiting for GPU work; the returned COM owner pins its mapping.
    fn card_fetch_at_least(&self, bytes: u64) -> Option<CardFetch> {
        let mut pool = self.card_fetch.lock().unwrap_or_else(|e| e.into_inner());
        if pool.as_ref().is_none_or(|f| f.size < bytes) {
            let size = bytes.checked_next_multiple_of(FETCH_GRANULARITY)?;
            if let Some((buffer, data)) = self.mapped_buffer(size, true) {
                *pool = Some(CardFetch { buffer, data: data as usize, size });
            }
        }
        pool.clone()
    }
}

/// As in the reference card_back, a packed range names its source copy,
/// its offsets within that copy and its offset in the shared readback.
#[derive(Debug, PartialEq, Eq)]
struct FetchPiece {
    copy: usize,
    lo: usize,
    hi: usize,
    at: u64,
}

/// Source torn::Check, specialized to this branch's contiguous packed
/// fetches and one completed flight. Widen reads to the exact shader
/// hash extent, but keep landing/rehashing confined to the original
/// logical flagged interval, including at skewed title-page boundaries.
#[derive(Debug)]
struct CheckedPiece {
    from: usize,
    to: usize,
    hash: u32,
}

#[derive(Debug)]
struct CheckedRun {
    lo: usize,
    hi: usize,
    pieces: Vec<CheckedPiece>,
}

impl CheckedRun {
    fn new(start: usize, size: usize, lo: usize, hi: usize,
        expected: &std::collections::BTreeMap<usize, u32>) -> Self
    {
        let pieces = pieces_of(start, lo, hi).map(|number| {
            let (from, to) = card_hashed_of(start, size, number);
            CheckedPiece { from, to, hash: expected[&number] }
        }).collect();
        Self { lo, hi, pieces }
    }

    fn extent(&self) -> (usize, usize) {
        (self.lo & !15, self.hi.next_multiple_of(16))
    }

    fn logical(&self, fetched: &FetchPiece) -> FetchPiece {
        assert_eq!(self.extent(), (fetched.lo, fetched.hi));
        FetchPiece { copy: fetched.copy, lo: self.lo, hi: self.hi,
            at: fetched.at + (self.lo - fetched.lo) as u64 }
    }

    /// Return the first mismatching shader piece, if any. Never merge
    /// bytes of even a partially verified batch.
    fn mismatch(&self, bytes: &[u8]) -> Option<&CheckedPiece> {
        let (from, to) = self.extent();
        assert_eq!(bytes.len(), to - from);
        self.pieces.iter().find(|p|
            super::torn::hash(&bytes[p.from - from..p.to - from]) != p.hash)
    }
}

type CheckedRuns = std::collections::BTreeMap<(usize, usize, usize), CheckedRun>;

/// # Safety
/// The batch is a completed fetch into this live, exclusive mapping.
/// The caller keeps it stable until all checks and landing finish.
unsafe fn check_fetch_batch(pool: &CardFetch, batch: &[FetchPiece], checks: &CheckedRuns) -> bool {
    for piece in batch {
        let check = &checks[&(piece.copy, piece.lo, piece.hi)];
        // SAFETY: caller's checked fence and packed in-bounds offsets.
        let bytes = unsafe { std::slice::from_raw_parts(
            (pool.data as *const u8).add(piece.at as usize), piece.hi - piece.lo) };
        if let Some(bad) = check.mismatch(bytes) {
            tracing::warn!(copy = piece.copy, from = bad.from, to = bad.to,
                "GPU readback hash mismatch");
            return false;
        }
    }
    true
}

#[test]
fn checked_fetch_hashes_aligned_pieces_but_lands_only_logical_bytes() {
    let (start, size) = (3usize, 2 * PIECE);
    let (lo, hi) = piece_at(start, size, 1);
    let (a, b) = card_hashed_of(start, size, 1);
    let mut bytes: Vec<u8> = (0..b-a).map(|i| (i.wrapping_mul(19) >> 3) as u8).collect();
    let expected = std::collections::BTreeMap::from([(1, super::torn::hash(&bytes))]);
    let check = CheckedRun::new(start, size, lo, hi, &expected);
    assert_eq!(check.extent(), (a, b));
    assert!(a < lo && hi < b);
    assert!(check.mismatch(&bytes).is_none());
    let fetched = FetchPiece { copy: 2, lo: a, hi: b, at: 128 };
    assert_eq!(check.logical(&fetched), FetchPiece { copy: 2, lo, hi, at: 128 + (lo - a) as u64 });
    // Even bytes read only for the shader's aligned load are checked.
    bytes[0] ^= 1;
    assert!(check.mismatch(&bytes).is_some());
    bytes[0] ^= 1;
    assert!(check.mismatch(&bytes).is_none());
}

#[test]
fn checked_fetch_validates_every_piece_in_a_joined_run() {
    let size = 2 * PIECE;
    let mut bytes = vec![0x55; size];
    let hash = super::torn::hash(&bytes[..PIECE]);
    let expected = std::collections::BTreeMap::from([(0, hash), (1, hash)]);
    let check = CheckedRun::new(0, size, 0, size, &expected);
    assert!(check.mismatch(&bytes).is_none());
    bytes[PIECE + 17] ^= 1;
    assert_eq!(check.mismatch(&bytes).unwrap().from, PIECE);
    bytes[PIECE + 17] ^= 1;
    assert!(check.mismatch(&bytes).is_none());
}

/// Usually all pieces fit in one batch. If growth fails, the already
/// admitted pool holds at least the largest single copy, so we can reuse
/// it across fenced batches without dropping any flagged bytes.
fn pack_fetches(ranges: &[(usize, usize, usize)], capacity: u64) -> Vec<Vec<FetchPiece>> {
    let mut batches: Vec<Vec<FetchPiece>> = Vec::new();
    let mut at = 0u64;
    for &(copy, lo, hi) in ranges {
        assert!(lo < hi);
        let n = (hi - lo) as u64;
        assert!(n <= capacity);
        if batches.is_empty() || n > capacity - at {
            batches.push(Vec::new());
            at = 0;
        }
        batches.last_mut().unwrap().push(FetchPiece { copy, lo, hi, at });
        at += n;
    }
    batches
}

#[test]
fn shared_fetch_packs_ranges_and_preserves_all_bytes_across_batches() {
    let ranges = [(0, 3, 9), (0, 16, 20), (2, 100, 108), (3, 0, 2)];
    let batches = pack_fetches(&ranges, 10);
    assert_eq!(batches, [
        vec![FetchPiece { copy: 0, lo: 3, hi: 9, at: 0 },
             FetchPiece { copy: 0, lo: 16, hi: 20, at: 6 }],
        vec![FetchPiece { copy: 2, lo: 100, hi: 108, at: 0 },
             FetchPiece { copy: 3, lo: 0, hi: 2, at: 8 }],
    ]);
    assert_eq!(pack_fetches(&ranges, 20).len(), 1);
    assert!(pack_fetches(&[], 0).is_empty());
    assert_eq!(pack_fetches(&[(0, 0, 10), (1, 0, 10)], 10).len(), 2);
}

/// Per-copy ownership of the source CardWork's comparison resources.
/// The existing submission lock replaces the reference's in-flight relay
/// pool. The large fetched-byte allocation is now shared by the host.
pub(crate) struct CardWork {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    hashes: ID3D12Resource,
    differs: ID3D12Resource,
    flags: ID3D12Resource,
    flags_data: usize,
}

impl CardWork {
    pub(crate) fn new(host: &Host, size: u64) -> Option<Self> {
        let (root, pso) = host.card_pipeline.get_or_init(|| {
            match compare_pipeline(&host.device) {
                Ok(pipeline) => Some(pipeline),
                Err(e) => {
                    tracing::warn!(error = %e, "card compare pipeline unavailable; use shared buffers");
                    None
                }
            }
        }).as_ref()?;
        let hashes = host.uav_buffer(card_hash_bytes(size))?;
        let flag_bytes = card_hash_bytes(size) / 4;
        let differs = host.uav_buffer(flag_bytes)?;
        let (flags, flags_data) = host.mapped_buffer(flag_bytes, true)?;
        // Reserve enough shared space before publishing the card buffer:
        // failure still selects the original CPU-visible Shadow path.
        if host.card_fetch_at_least(size)?.size < size {
            return None;
        }
        Some(Self {
            root: root.clone(), pso: pso.clone(), hashes, differs, flags,
            flags_data: flags_data as usize,
        })
    }

    /// # Safety
    /// An open list, live card buffer and resources kept through its fence.
    unsafe fn bind(&self, list: &ID3D12GraphicsCommandList, buffer: &ID3D12Resource) {
        // SAFETY: the caller keeps the root signature, PSO, buffers and list live.
        unsafe {
            list.SetPipelineState(&self.pso);
            list.SetComputeRootSignature(&self.root);
            list.SetComputeRootShaderResourceView(1, buffer.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.hashes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, self.differs.GetGPUVirtualAddress());
        }
    }
}

/// Reference CardWork::sync/check/fetch lists and card_done ownership.
/// A synchronous submission needs at most one allocator/list per phase,
/// rather than a new HostList (and a second prelude) at every dispatch.
pub(super) struct CardPrelude {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

#[derive(Clone, Copy)]
enum CardPhase {
    Sync = 0,
    Check = 1,
    Fetch = 2,
}

impl Host {
    /// Adapt cool_fox's make_prelude, initially closed so every lease,
    /// including the first, uses the same Reset path.
    fn make_card_prelude(&self) -> windows::core::Result<CardPrelude> {
        self.make_card_prelude_for(D3D12_COMMAND_LIST_TYPE_DIRECT)
    }

    fn make_card_prelude_for(&self, kind: D3D12_COMMAND_LIST_TYPE) -> windows::core::Result<CardPrelude> {
        // SAFETY: plain object creation.
        unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(kind)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, kind, &allocator, None)?;
                list.Close()?;
                Ok(CardPrelude { allocator, list })
            })()
        }
    }

    /// Called with submitted held. Take the closed list out of the pool,
    /// reset it only after its previous checked fence, then record into it.
    fn card_list(&self, phase: CardPhase) -> CardPrelude {
        let slot = self.card_lists.lock().unwrap_or_else(|e| e.into_inner())[phase as usize].take();
        let work = slot.unwrap_or_else(|| self.make_card_prelude().unwrap_or_else(|e|
            fail(format_args!("could not create card prelude: {e}"))));
        // SAFETY: fresh closed list or a list returned after its checked
        // completion fence. The submission lock excludes all other leases.
        let reset = unsafe {
            work.allocator.Reset().and_then(|()| work.list.Reset(&work.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("could not reset card prelude: {e}"));
        }
        work
    }

    /// The source's card_done adapted to our existing synchronous fence.
    /// The caller owns every referenced resource through this call.
    fn run_card_list(&self, phase: CardPhase, work: CardPrelude, submitted: &mut u64) {
        // SAFETY: this is our open direct list, its owners are live, and
        // submitted serializes access to the queue, fence and event.
        unsafe {
            let ran = (|| -> windows::core::Result<()> {
                work.list.Close()?;
                let command: ID3D12CommandList = work.list.cast()?;
                self.queue.ExecuteCommandLists(&[Some(command)]);
                *submitted += 1;
                self.queue.Signal(&self.fence, *submitted)?;
                self.fence.SetEventOnCompletion(*submitted, self.event)?;
                Ok(())
            })();
            if let Err(e) = ran {
                fail(format_args!("card prelude submission failed: {e}"));
            }
            if WaitForSingleObject(self.event, INFINITE) != windows::Win32::Foundation::WAIT_OBJECT_0
                || self.fence.GetCompletedValue() < *submitted
            {
                fail(format_args!("card prelude fence wait failed"));
            }
            if let Err(e) = self.device.GetDeviceRemovedReason() {
                fail(format_args!("card prelude device removed: {e}"));
            }
        }
        self.recycle_card_list(phase, work);
    }

    /// Called only after the fence covering the list was checked, including
    /// the title-fence path. The caller holds the submission lock.
    fn recycle_card_list(&self, phase: CardPhase, work: CardPrelude) {
        // Only completed work is ever put back. Keep one per phase;
        // it is impossible to overwrite another in-flight allocator.
        let mut lists = self.card_lists.lock().unwrap_or_else(|e| e.into_inner());
        assert!(lists[phase as usize].is_none(), "card prelude slot already occupied");
        lists[phase as usize] = Some(work);
    }
}

/// Source RelayFetch: an independent list, fence and event, optionally
/// on a dedicated copy queue. Reused only after checked completion.
/// The existing shared CardFetch pool remains the bounded readback owner.
/// The copy queue is serial with title work until detached flights and
/// persistent suspect tracking are available.
pub(super) struct RelayFetch {
    work: Option<CardPrelude>,
    queue: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
}

impl Drop for RelayFetch {
    fn drop(&mut self) {
        // SAFETY: this private event is owned here; the serial fetch
        // contract has completed all waits before releasing the state.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.event) };
    }
}

fn independent_fetch_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_RELAY_FETCH").is_some_and(|v| v == "1"))
}

/// The source copy queue has no compute capability. Keep the direct
/// path for mode-1 compare/deferred GPU rehash, and require CPU hash
/// validation before enabling the independent queue.
fn relay_copy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_COPY").is_some_and(|v| v == "1"))
        && super::torn::on() && !deferred_hashes()
}

impl RelayFetch {
    fn take_list(&mut self) -> CardPrelude {
        let work = self.work.take().expect("one private fetch list lease");
        // SAFETY: new closed list or a list returned only after this
        // fetch state's checked completion. No other caller owns it.
        let reset = unsafe {
            work.allocator.Reset().and_then(|()| work.list.Reset(&work.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("could not reset relay fetch list: {e}"));
        }
        work
    }

    /// Source relay_card_back's copy/direct queue branches. Return the
    /// private fence, not a number to use in SentRuns. The caller holds
    /// the submission lock and owns the buffers until completion.
    /// A copy queue fetch explicitly waits for the already checked main
    /// fence, and the CPU waits for the copy fence before reuse.
    fn run(&mut self, host: &Host, work: CardPrelude, after: u64) -> (u64, u64) {
        let sent = self.sent.checked_add(1).filter(|n| *n != u64::MAX)
            .expect("relay fetch fence exhausted");
        // SAFETY: exclusive open list, live referenced resources, and
        // a private fence/event. Direct-queue buffers have decayed to
        // COMMON after completion; copy lists contain only copies.
        let ran = unsafe {
            (|| -> windows::core::Result<()> {
                work.list.Close()?;
                let list: ID3D12CommandList = work.list.cast()?;
                let queue = self.queue.as_ref().unwrap_or(&host.queue);
                if self.queue.is_some() {
                    let completed = host.fence.GetCompletedValue();
                    if completed < after || completed == u64::MAX {
                        fail(format_args!("relay copy queue main fence incomplete"));
                    }
                    queue.Wait(&host.fence, after)?;
                }
                queue.ExecuteCommandLists(&[Some(list)]);
                queue.Signal(&self.fence, sent)?;
                self.fence.SetEventOnCompletion(sent, self.event)?;
                Ok(())
            })()
        };
        if let Err(e) = ran {
            fail(format_args!("relay fetch submission failed: {e}"));
        }
        // SAFETY: the live private event is only waited by this owner.
        let waited = unsafe { WaitForSingleObject(self.event, INFINITE) };
        // SAFETY: plain queries after the wait.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if waited != windows::Win32::Foundation::WAIT_OBJECT_0
            || completed < sent || completed == u64::MAX
        {
            fail(format_args!("relay fetch fence wait failed"));
        }
        // SAFETY: query the current device status before exposing bytes.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            fail(format_args!("relay fetch device removed: {e}"));
        }
        self.sent = sent;
        assert!(self.work.is_none());
        self.work = Some(work);
        (sent, completed)
    }
}

impl Host {
    /// Source make_relay_fetch: the copy allocator/list and high-priority
    /// copy queue are distinct from the direct queue. If creation fails,
    /// retain the private direct path, then the main-fence fallback.
    fn make_relay_fetch(&self, copying: bool) -> Option<RelayFetch> {
        let copying = copying.then(|| {
            let work = self.make_card_prelude_for(D3D12_COMMAND_LIST_TYPE_COPY)?;
            // SAFETY: plain object creation for the matching list type.
            let queue: ID3D12CommandQueue = unsafe {
                self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    ..Default::default()
                })?
            };
            Ok::<_, windows::core::Error>((work, queue))
        });
        let (work, queue) = match copying {
            Some(Ok((work, queue))) => (work, Some(queue)),
            Some(Err(e)) => {
                tracing::warn!(error = %e, "no copy queue; retaining direct relay fetch");
                (self.make_card_prelude().ok()?, None)
            }
            None => (self.make_card_prelude().ok()?, None),
        };
        // SAFETY: independent objects owned by this state, no work yet.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch { work: Some(work), queue, fence, event, sent: 0 })
        }
    }

    fn relay_fetch(&self) -> Option<std::sync::MutexGuard<'_, RelayFetch>> {
        if !independent_fetch_enabled() {
            return None;
        }
        self.relay_fetch.get_or_init(|| {
            let fetch = self.make_relay_fetch(relay_copy_enabled()).map(Mutex::new);
            if fetch.is_none() {
                tracing::warn!("could not create private relay fetch; retaining main-fence fetch");
            }
            fetch
        }).as_ref().map(|fetch| fetch.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Source declared_writes / CardPlan: group ranges by copy identity.
/// Adapt to the marked-page and withholding paths of this branch; use
/// a full compare whenever the list does not have bounded declarations.
type CardScope = std::collections::BTreeMap<usize, Vec<(usize, usize)>>;

fn ranges_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_COMPARE_RANGES").is_some_and(|v| v == "1"))
}

fn add_range(scope: &mut CardScope, shadow: &std::sync::Arc<crate::shadow::Shadow>,
    address: usize, len: usize, bounded: bool)
{
    if shadow.card().is_none() {
        return;
    }
    let (start, _) = shadow.range();
    let size = shadow.card_hash_size() as usize;
    let start = start as usize;
    let (from, to) = if bounded {
        (address.saturating_sub(start).min(size),
         address.saturating_add(len).saturating_sub(start).min(size))
    } else {
        // Root UAVs have no reliable length in the current branch.
        (0, size)
    };
    if from < to {
        scope.entry(std::sync::Arc::as_ptr(shadow) as usize).or_default().push((from, to));
    }
}

/// Returns None for the original full-coverage path. An empty declaration
/// set is not evidence that arbitrary command lists cannot have written.
fn declared_scope(runs: &[&HostList], marks: Option<&[crate::shadow::MarkRange]>) -> Option<CardScope> {
    if !ranges_enabled() {
        return None;
    }
    let mut scope = CardScope::new();
    let mut declared = false;
    for list in runs {
        let d = list.deferred();
        if d.card_unbounded || !d.after.is_empty() || !d.missed.is_empty()
            || d.copies.iter().any(|s| matches!(s, Step::Heap(..) | Step::Memory(..)))
        {
            return None;
        }
        for step in &d.copies {
            if let Step::GpuWrite(shadow, address, len, marked, _) = step {
                declared = true;
                add_range(&mut scope, shadow, *address, *len, *marked);
            }
        }
    }
    if !declared {
        return None;
    }
    for (shadow, address, len) in marks? {
        add_range(&mut scope, shadow, *address, *len, true);
    }
    Some(scope)
}

/// Reference picked::declare runs after the final CPU refresh and before
/// the title lists. Keep the candidates separate from recorded GpuWrite
/// steps: changing recorded lists would affect later replays.
pub(super) fn prepare_picked(runs: &[&HostList], flight: u64) -> Vec<crate::picked::Range> {
    if !crate::picked::lands() || crate::memory::withholding() {
        return Vec::new();
    }
    let mut shown = Vec::new();
    let mut declared = Vec::new();
    for list in runs {
        let d = list.deferred();
        if !d.after.is_empty() || !d.missed.is_empty()
            || d.copies.iter().any(|s| matches!(s, Step::Memory(..) | Step::Run(..) | Step::Host(..)))
        {
            return Vec::new();
        }
        for step in &d.copies {
            match step {
                Step::Heap(viewed) if d.card_picked => shown.extend(viewed.picked_ranges()),
                Step::GpuWrite(shadow, address, len, _, _) => {
                    declared.push((shadow.clone(), *address, *len));
                }
                _ => {}
            }
        }
    }
    let ranges = crate::picked::known_ranges(shown, &declared);
    for (shadow, address, len) in &ranges {
        // Snapshot only after this submission's refreshes. Previously
        // discovered GPU bytes are part of the baseline, not new writes.
        shadow.writing_for_flight(*address, *len, false, flight);
    }
    ranges
}

/// The reference's direct changed-byte relay is only safe here for the
/// already-declared picked ranges, not arbitrary hash discoveries.
/// Select after the title fence, before any fetch lands. The existing
/// picked candidate selection excludes overlapping declared writes.
pub(super) type RelayPicked = std::collections::BTreeMap<usize, (usize, usize)>;

pub(super) fn relay_picked(ranges: &[crate::picked::Range], withholding: bool,
    marks: Option<&[crate::shadow::MarkRange]>) -> RelayPicked {
    if !crate::shadow::card_relay_merge_enabled() || withholding || ranges.is_empty() {
        return RelayPicked::new();
    }
    let Some(marks) = marks else { return RelayPicked::new() };
    ranges.iter().filter(|range| !marks.iter().any(|mark| crate::picked::overlaps(range, mark)))
        .filter_map(|(shadow, address, len)| {
            let (start, size) = shadow.range();
            let from = address.checked_sub(start as usize)?;
            let to = from.checked_add(*len)?;
            (from < to && to <= size as usize)
                .then_some((std::sync::Arc::as_ptr(shadow) as usize, (from, to)))
        }).collect()
}

fn relay_overlap(lo: usize, hi: usize, range: Option<&(usize, usize)>) -> Option<(usize, usize)> {
    let &(from, to) = range?;
    let (a, b) = (lo.max(from), hi.min(to));
    (a < b).then_some((a, b))
}

#[test]
fn relay_fetch_only_merges_the_declared_intersection() {
    assert_eq!(relay_overlap(0, 64, Some(&(16, 48))), Some((16, 48)));
    assert_eq!(relay_overlap(32, 64, Some(&(16, 48))), Some((32, 48)));
    assert_eq!(relay_overlap(16, 48, Some(&(0, 64))), Some((16, 48)));
    assert_eq!(relay_overlap(48, 64, Some(&(16, 48))), None);
    assert_eq!(relay_overlap(0, 16, None), None);
}

/// Source relay_land's completed-piece landing, adapted to the exact
/// declared picked ranges and existing staging path. Both inline and
/// worker paths use the same merge and return its byte count.
/// The submission owner guarantees an exclusive lease on all mappings
/// and the readback remains completed and stable through the call.
fn land_fetch_batch(pool: &CardFetch, copies: &[std::sync::Arc<crate::shadow::Shadow>],
    batch: Vec<FetchPiece>, relay: &RelayPicked, fetched_after: u64,
    relay_title: &mut Vec<u8>) -> u64
{
    let mut written = 0;
    for piece in batch {
        // SAFETY: the checked readback fence completed; this
        // owner pins its mapping and the submission owner keeps
        // the pool contents stable until landing is acknowledged.
        let bytes = unsafe { std::slice::from_raw_parts(
            (pool.data as *const u8).add(piece.at as usize), piece.hi - piece.lo) };
        let shadow = &copies[piece.copy];
        let owner = std::sync::Arc::as_ptr(shadow) as usize;
        if let Some((lo, hi)) = relay_overlap(piece.lo, piece.hi, relay.get(&owner)) {
            // Only this intersection is a declared picked write.
            // Outside it update the staging mirror as before,
            // never turn an undeclared discovery into title data.
            let (a, b) = (lo - piece.lo, hi - piece.lo);
            if a != 0 {
                shadow.land_card(piece.lo, &bytes[..a], fetched_after);
            }
            let n = shadow.card_merge_relay(lo, &bytes[a..b], fetched_after, relay_title);
            written += n as u64;
            if b < bytes.len() {
                shadow.land_card(hi, &bytes[b..], fetched_after);
            }
        } else {
            shadow.land_card(piece.lo, bytes, fetched_after);
        }
    }
    written
}

/// Reference picked::with / hand_back: land newly written bytes before
/// the title's completion signal. The current branch's mark and withholding
/// semantics take precedence; release cancelled snapshots without merging.
pub(super) fn merge_picked(ranges: &[crate::picked::Range], withholding: bool, relay: &RelayPicked,
    marks: Option<&[crate::shadow::MarkRange]>) {
    if ranges.is_empty() {
        return;
    }
    let marks = if withholding { None } else { marks };
    for range in ranges {
        let (shadow, address, len) = range;
        if relay.contains_key(&(std::sync::Arc::as_ptr(shadow) as usize)) {
            // Its changed pieces have landed directly from the fenced
            // fetch. Retirement happens with all the flight's ranges.
            continue;
        }
        let safe = marks.is_some_and(|marks|
            !marks.iter().any(|mark| crate::picked::overlaps(range, mark)));
        if safe {
            let written = shadow.copy_back_merge(*address, *len);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
    }
}

/// The source's picked-view discovery adapted to the existing full
/// compare. Do not change which pieces are hashed, fetched or landed.
/// Only a current UAV hull with no declaration or mark for the entire
/// flagged piece is reported. Ambiguous title-memory paths are skipped.
struct PickedObservation {
    shown: CardScope,
    declared: CardScope,
    // At most one unexplained piece per candidate owner, since the
    // reference's found registry promotes the whole copy.
    evidence: std::collections::BTreeMap<usize, (usize, usize)>,
}

fn picked_observation(runs: &[&HostList], marks: Option<&[crate::shadow::MarkRange]>) -> Option<PickedObservation> {
    if !crate::picked::on() {
        return None;
    }
    let mut shown = CardScope::new();
    let mut declared = CardScope::new();
    for list in runs {
        let d = list.deferred();
        if !d.after.is_empty() || !d.missed.is_empty()
            || d.copies.iter().any(|s| matches!(s, Step::Memory(..)))
        {
            return None;
        }
        for step in &d.copies {
            match step {
                Step::Heap(viewed) if d.card_picked => {
                    for (shadow, address, len) in viewed.picked_ranges() {
                        add_range(&mut shown, &shadow, address, len, true);
                    }
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    add_range(&mut declared, shadow, *address, *len, *marked);
                }
                _ => {}
            }
        }
    }
    if shown.is_empty() {
        return None;
    }
    for (shadow, address, len) in marks? {
        add_range(&mut declared, shadow, *address, *len, true);
    }
    Some(PickedObservation { shown, declared, evidence: Default::default() })
}

impl PickedObservation {
    fn flagged(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, lo: usize, hi: usize) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        let Some(shown) = self.shown.get(&key) else { return };
        let declared = self.declared.get(&key).map_or(&[][..], Vec::as_slice);
        if crate::picked::unexplained((lo, hi), shown, declared) {
            if crate::picked::rotates() {
                self.evidence.entry(key).or_insert((lo, hi));
            } else {
                crate::picked::found(shadow, lo);
            }
        }
    }

    /// Feed the source's rotating scheduler only after consuming the
    /// real GPU comparison flags. Staging and GPU baseline are unaffected.
    fn finish_observation(self, copies: &[std::sync::Arc<crate::shadow::Shadow>]) {
        if !crate::picked::rotates() {
            return;
        }
        let mut shown = Vec::new();
        let mut evidence = Vec::new();
        for shadow in copies {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            if let Some(ranges) = self.shown.get(&key) {
                shown.extend(ranges.iter().map(|&(lo, hi)| (shadow.clone(), lo, hi - lo)));
            }
            if let Some(&(lo, hi)) = self.evidence.get(&key) {
                evidence.push((shadow.clone(), lo, hi - lo));
            }
        }
        crate::picked::rotate(shown, evidence);
    }
}

/// Join the reference's 64 KiB piece runs. The comparison hashes a whole
/// piece and updates its baseline, so fetch the whole piece too, not just
/// a declared subrange. Distinct spans have disjoint result offsets.
fn compare_spans(start: usize, size: usize, ranges: Option<&[(usize, usize)]>) -> Vec<CardSpan> {
    let Some(ranges) = ranges else {
        return (size != 0).then(|| card_span_of(start, 0, size)).into_iter().collect();
    };
    let runs = ranges.iter().filter_map(|&(from, to)| {
        let to = to.min(size);
        if from >= to {
            return None;
        }
        let pieces = pieces_of(start, from, to);
        let lo = piece_at(start, size, pieces.start).0;
        let hi = piece_at(start, size, pieces.end - 1).1;
        Some(card_span_of(start, lo, hi))
    }).collect();
    joined_piece_runs(runs)
}

#[test]
fn declared_compare_spans_cover_each_touched_piece_once() {
    let start = 4096;
    let size = 6 * PIECE;
    let ranges = [(3, 9), (7, 12), (PIECE - start - 1, PIECE - start + 1),
        (4 * PIECE, 4 * PIECE + 4), (size, size + 1)];
    let spans = compare_spans(start, size, Some(&ranges));
    assert_eq!(spans.iter().map(|s| (s.first, s.pieces)).collect::<Vec<_>>(), [(0, 2), (4, 1)]);
    assert_eq!(spans[0].from, 0);
    assert_eq!(spans[0].to as usize, 2 * PIECE - start);
    assert_eq!(spans[1].from as usize, 4 * PIECE - start);
    assert_eq!(spans[1].to as usize, 5 * PIECE - start);
    assert!(compare_spans(start, size, Some(&[])).is_empty());
    assert_eq!(compare_spans(start, size, None), [card_span_of(start, 0, size)]);
    // First and last partial pages must cover all padded GPU bytes.
    let edge = compare_spans(start, size, Some(&[(size - 1, size)]));
    assert_eq!(edge[0].to as usize, size);
}

/// Mode 0 hashes 16-byte loads, including a load crossing a title-page
/// boundary when the copy start is not 16-byte aligned. Rehash every
/// touched piece, clip padding before constructing spans.
fn dirty_hash_spans(start: usize, size: usize, ranges: &[(usize, usize)]) -> Vec<CardSpan> {
    let aligned: Vec<_> = ranges.iter().filter_map(|&(from, to)| {
        let to = to.min(size);
        (from < to).then(|| (from & !15, to.next_multiple_of(16).min(size)))
    }).collect();
    compare_spans(start, size, Some(&aligned))
}

#[test]
fn dirty_hash_spans_clip_padding_and_include_shared_loads() {
    let size = PIECE + 32;
    let spans = dirty_hash_spans(0, size, &[(0, 2 * PIECE)]);
    assert_eq!(spans, [card_span_of(0, 0, size)]);
    assert!(dirty_hash_spans(0, size, &[(size, 2 * PIECE)]).is_empty());
    // Title-page boundary at PIECE-1, shared 16-byte load.
    let spans = dirty_hash_spans(1, size, &[(PIECE - 16, PIECE - 15)]);
    assert_eq!(spans, [card_span_of(1, 0, size)]);
    let spans = dirty_hash_spans(0, size, &[(PIECE + 17, PIECE + 18)]);
    assert_eq!(spans, [card_span_of(0, PIECE, size)]);
}

/// The source compares with mode 1 on the synchronous path, leaving the
/// old hash intact, and mode 2 for in-flight writes. This branch's CPU
/// staging mirror must remain coherent even for undeclared writes, so
/// explicitly rehash only the pieces whose readback has been recorded.
/// Opt-in until the production path is tested on a D3D12 adapter.
fn deferred_hashes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_LOOK_ONLY").is_some_and(|v| v == "1"))
}

/// Each fetched piece is complete (including the short first/last piece).
/// Group by exact copy for the following piece-span coalescing pass.
fn fetched_hash_runs(batch: &[FetchPiece]) -> std::collections::BTreeMap<usize, Vec<(usize, usize)>> {
    let mut out: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
    for piece in batch {
        out.entry(piece.copy).or_default().push((piece.lo, piece.hi));
    }
    out
}

#[test]
fn deferred_hashes_commit_only_fetched_pieces() {
    let start = 4096;
    let size = 4 * PIECE;
    let (a, b) = piece_at(start, size, 0);
    let (c, d) = piece_at(start, size, 2);
    let (e, f) = piece_at(start, size, 3);
    let batch = [
        FetchPiece { copy: 1, lo: a, hi: b, at: 0 },
        FetchPiece { copy: 1, lo: c, hi: d, at: (b - a) as u64 },
        FetchPiece { copy: 1, lo: e, hi: f, at: (b - a + d - c) as u64 },
        FetchPiece { copy: 2, lo: 0, hi: PIECE, at: (b - a + d - c + f - e) as u64 },
    ];
    let runs = fetched_hash_runs(&batch);
    let spans = compare_spans(start, size, Some(&runs[&1]));
    assert_eq!(spans.iter().map(|s| (s.first, s.pieces)).collect::<Vec<_>>(), [(0, 1), (2, 2)]);
    assert_eq!(compare_spans(0, size, Some(&runs[&2])), [card_span_of(0, 0, PIECE)]);
    assert!(fetched_hash_runs(&[]).is_empty());
}

type CardPlans = Vec<(usize, Vec<CardSpan>)>;

/// Source CardWork::taken/out: one bounded area, append results of a
/// plan behind all outstanding plans, and rewind only when none is out.
/// Unlike the source's CPU-visible UAV, this branch retains per-copy
/// device-local differs scratch and copies into a packed readback arena.
const RESULT_BYTES: u64 = 1 << 20;
const RESULT_WORDS: u32 = (RESULT_BYTES / 4) as u32;

#[derive(Default)]
struct FlagCursor {
    taken: u32,
    out: u32,
}

impl FlagCursor {
    fn take(&mut self, words: u32) -> Option<u32> {
        if words == 0 {
            return None;
        }
        let at = self.taken;
        let end = at.checked_add(words).filter(|&end| end <= RESULT_WORDS)?;
        self.out = self.out.checked_add(1)?;
        self.taken = end;
        Some(at)
    }

    fn done(&mut self) {
        self.out = self.out.checked_sub(1).expect("unmatched card flag lease");
        if self.out == 0 {
            self.taken = 0;
        }
    }
}

pub(super) struct FlagArena {
    buffer: ID3D12Resource,
    data: usize,
    cursor: Mutex<FlagCursor>,
}

impl FlagArena {
    fn make(host: &Host) -> Option<std::sync::Arc<Self>> {
        let (buffer, data) = host.mapped_buffer(RESULT_BYTES, true)?;
        crate::host::name(&buffer, || "card compare flight results".into());
        Some(std::sync::Arc::new(Self {
            buffer, data: data as usize, cursor: Mutex::new(FlagCursor::default()),
        }))
    }

    fn take(self: &std::sync::Arc<Self>, words: u32) -> Option<FlagLease> {
        let at = self.cursor.lock().unwrap_or_else(|e| e.into_inner()).take(words)?;
        Some(FlagLease { arena: self.clone(), at, words })
    }
}

/// A plan's private dwords in the mapped arena, retained until the
/// check fence has completed and every result has been read. Offsets
/// are relative to the plan, not the shadow's piece number.
struct FlagLease {
    arena: std::sync::Arc<FlagArena>,
    at: u32,
    words: u32,
}

impl FlagLease {
    fn offset(&self, at: u32, words: u32) -> u64 {
        assert!(at.checked_add(words).is_some_and(|end| end <= self.words));
        u64::from(self.at + at) * 4
    }

    /// # Safety
    /// This plan's fence completed; this lease owns the requested
    /// interval exclusively and no GPU writes it again before drop.
    unsafe fn read(&self, at: u32, words: u32) -> &[u32] {
        let offset = self.offset(at, words);
        // SAFETY: the caller checked completion and the span is inside
        // the live, persistently mapped readback arena.
        unsafe { std::slice::from_raw_parts(
            (self.arena.data as *const u32).add((offset / 4) as usize), words as usize) }
    }
}

impl Drop for FlagLease {
    fn drop(&mut self) {
        self.arena.cursor.lock().unwrap_or_else(|e| e.into_inner()).done();
    }
}

fn packed_flags_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_FLAGS_OWN").is_some_and(|v| v == "1"))
}

fn plan_words(plans: &CardPlans) -> Option<u32> {
    plans.iter().flat_map(|(_, spans)| spans).try_fold(0u32, |n, span| n.checked_add(span.pieces))
}

#[test]
fn flag_cursor_rewinds_only_after_last_plan() {
    let mut cursor = FlagCursor::default();
    assert_eq!(cursor.take(0), None);
    assert_eq!(cursor.take(16), Some(0));
    assert_eq!(cursor.take(24), Some(16));
    // Even if the first plan finished, the later plan still owns
    // results. Do not rewind or allocate over its range.
    cursor.done();
    assert_eq!(cursor.take(RESULT_WORDS), None);
    assert_eq!(cursor.take(RESULT_WORDS - 40), Some(40));
    assert_eq!(cursor.take(1), None);
    cursor.done();
    assert_eq!(cursor.take(1), None);
    cursor.done();
    assert_eq!(cursor.take(RESULT_WORDS), Some(0));
    cursor.done();
    assert_eq!(cursor.taken, 0);
    assert_eq!(cursor.out, 0);
}

#[test]
fn packed_result_words_count_spans_not_piece_indices() {
    let plans = vec![
        (3, vec![card_span_of(0, 8 * PIECE, 10 * PIECE)]),
        (9, vec![card_span_of(0, 0, PIECE), card_span_of(0, 7 * PIECE, 8 * PIECE)]),
    ];
    assert_eq!(plan_words(&plans), Some(4));
}


/// Source CardPlan::check and result ownership. Keep the carded
/// snapshot and optional packed result lease until consumption.
pub(super) struct CardCheck {
    work: CardPrelude,
    commands: [Option<ID3D12CommandList>; 1],
    plans: CardPlans,
    flags: Option<FlagLease>,
}

fn check_tail_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_CHECK_TAIL").is_some_and(|v| v == "1"))
}

fn card_plans(copies: &[std::sync::Arc<crate::shadow::Shadow>], scope: Option<&CardScope>) -> CardPlans {
    copies.iter().enumerate().filter_map(|(copy, shadow)| {
            shadow.card()?;
            let (start, _) = shadow.range();
            let ranges = scope.as_ref().map(|s| s.get(&(std::sync::Arc::as_ptr(shadow) as usize))
                .map_or(&[][..], Vec::as_slice));
            let spans = compare_spans(start as usize, shadow.card_hash_size() as usize, ranges);
            (!spans.is_empty()).then_some((copy, spans))
    }).collect()
}

impl Host {
    fn reserve_card_flags(&self, plans: &CardPlans) -> Option<FlagLease> {
        if !packed_flags_enabled() {
            return None;
        }
        let words = plan_words(plans)?;
        if words == 0 || words > RESULT_WORDS {
            return None;
        }
        let arena = self.card_flags.get_or_init(|| {
            let arena = FlagArena::make(self);
            if arena.is_none() {
                tracing::warn!("no packed card compare arena; retaining serial per-copy flags");
            }
            arena
        }).as_ref()?;
        // A full arena is allowed to fall back only under the current
        // one-flight contract. A future multi-flight path must settle
        // before this fallback or reserve space before sending.
        arena.take(words)
    }

    fn record_card_check(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>],
        plans: &CardPlans, deferred: bool, flags: Option<&FlagLease>) -> CardPrelude
    {
        let list = self.card_list(CardPhase::Check);
        let mut out_at = 0u32;
        // Source CardPlan::check begins with a global UAV barrier.
        // The separate ExecuteCommandLists call preserves buffer decay
        // between the title command group and this check.
        compare_barrier(&list.list);
        for (copy, spans) in plans {
            let shadow = &copies[*copy];
            let card = shadow.card().expect("planned card copy");
            // SAFETY: the check executes after title work. All buffers live,
            // and spans are disjoint complete pieces with valid flag offsets.
            unsafe {
                card.work.bind(&list.list, &shadow.buffer);
                for span in spans {
                    // Source's synchronous compare-only mode: no hash
                    // is overwritten until the changed bytes are fetched.
                    card_pass(&list.list, span, shadow.card_hash_size() as u32, span.first, if deferred { 1 } else { 2 });
                }
                compare_barrier(&list.list);
                list.list.ResourceBarrier(&[transition(&card.work.differs,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                for span in spans {
                    let at = u64::from(span.first) * 4;
                    let (buffer, dst) = if let Some(flags) = flags {
                        let dst = flags.offset(out_at, span.pieces);
                        out_at += span.pieces;
                        (&flags.arena.buffer, dst)
                    } else {
                        (&card.work.flags, at)
                    };
                    list.list.CopyBufferRegion(buffer, dst, &card.work.differs, at,
                        u64::from(span.pieces) * 4);
                }
            }
        }
        if let Some(flags) = flags {
            assert_eq!(out_at, flags.words);
        }
        list
    }

    /// The reference records CardPlan::check before sending title work.
    /// Marks are not available until the fence, so the post-fence
    /// declared-range option deliberately keeps the older path.
    pub(super) fn prepare_card_tail(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>]) -> Option<CardCheck> {
        if !check_tail_enabled() || ranges_enabled() {
            return None;
        }
        let plans = card_plans(copies, None);
        if plans.is_empty() {
            return None;
        }
        let flags = self.reserve_card_flags(&plans);
        let work = self.record_card_check(copies, &plans, deferred_hashes(), flags.as_ref());
        // SAFETY: a freshly recorded open list; all owners live in the
        // caller's carded snapshot for the entire submission.
        let command = unsafe { work.list.Close() }.and_then(|()| work.list.cast())
            .unwrap_or_else(|e| fail(format_args!("could not close card check tail: {e}")));
        Some(CardCheck { work, commands: [Some(command)], plans, flags })
    }

    /// # Safety
    /// The submission lock is held, all title groups have been queued,
    /// and the next checked fence covers this list too.
    pub(super) unsafe fn enqueue_card_tail(&self, check: &CardCheck) {
        // SAFETY: the caller owns the closed command list and its resources.
        // Use a distinct queue execution, not the same list array: the
        // current renderer relies on D3D12 buffer COMMON decay.
        unsafe { self.queue.ExecuteCommandLists(&check.commands) };
    }
}

impl Host {
    /// Adapt the source card_before: copy CPU-dirty ranges, then hash the
    /// joined pieces on the device before the title's lists execute.
    pub(super) fn card_before(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>], submitted: &mut u64) {
        // The previous fence completed, and serial execution consumed
        // every earlier readback. The next sync list signals sent_as.
        let retired = *submitted;
        let sent_as = retired.checked_add(1).expect("card send fence exhausted");
        let dirty: Vec<card_send::Dirty<'_>> = copies.iter().filter_map(|shadow| {
            let card = shadow.card()?;
            let ranges = card.take_dirty(sent_as, retired);
            (!ranges.is_empty()).then_some((shadow, card, ranges))
        }).collect();
        if dirty.is_empty() {
            return;
        }
        let own = card_send::take(self, &dirty);
        let list = self.card_list(CardPhase::Sync);
        if let Some(own) = &own {
            // SAFETY: open sync list, dirty owners and snapshot retained
            // until the same checked fence; buffers start in COMMON.
            unsafe { own.record(&list.list, &dirty) };
        }
        // Reference card_before: record all CPU-to-card copies first, then
        // one resource-barrier batch, then the hashing passes. Independent
        // card buffers no longer serialize a copy and a hash per buffer.
        let mut sent = Vec::new();
        let mut hashing = Vec::new();
        for (shadow, card, ranges) in &dirty {
            let (start, _) = shadow.range();
            let size = shadow.card_hash_size();
            let spans = dirty_hash_spans(start as usize, size as usize, ranges);
            if own.is_none() {
                // SAFETY: ranges lie in both live resources; buffers decayed
                // to COMMON after the preceding fence, promote to COPY_DEST.
                unsafe {
                    for &(from, to) in ranges {
                        list.list.CopyBufferRegion(&shadow.buffer, from as u64,
                            &card.staging, from as u64, (to - from) as u64);
                    }
                }
                if !spans.is_empty() {
                    sent.push(transition(&shadow.buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
                }
            }
            if !spans.is_empty() {
                hashing.push((shadow, card, size, spans));
            }
        }
        if !sent.is_empty() {
            // SAFETY: live distinct buffers, all uploads already recorded.
            // Each transition names its actual COPY_DEST state.
            unsafe { list.list.ResourceBarrier(&sent) };
        }
        if !hashing.is_empty() {
            for (shadow, card, size, spans) in hashing {
                // SAFETY: the source is now in SRV state; each CardWork
                // owns distinct hash buffers and disjoint piece ranges.
                unsafe {
                    card.work.bind(&list.list, &shadow.buffer);
                    for span in &spans {
                        card_pass(&list.list, span, size as u32, 0, 0);
                    }
                }
            }
            compare_barrier(&list.list);
        }
        self.run_card_list(CardPhase::Sync, list, submitted);
        debug_assert_eq!(*submitted, sent_as);
        if let Some(own) = own {
            own.give_back();
        }
    }

    /// Adapt source card_back: compare on the device, read flags after
    /// the fence and land flagged full pieces before the CPU merge.
    /// Full coverage is the default and the conservative fallback.
    pub(super) fn card_back(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>],
        runs: &[&HostList], tail: Option<CardCheck>, relay: &RelayPicked,
        marks: Option<&[crate::shadow::MarkRange]>, submitted: &mut u64)
    {
        if copies.is_empty() {
            return;
        }
        let deferred = deferred_hashes();
        let mut picked = picked_observation(runs, marks);
        let (plans, flag_lease) = if let Some(check) = tail {
            // Host::execute checked the title fence and device status,
            // and its tail was queued before that fence. Assert the
            // completed value here too before recycling the allocator.
            // SAFETY: a plain query under the submission lock.
            if unsafe { self.fence.GetCompletedValue() } < *submitted {
                fail(format_args!("card check tail fence incomplete"));
            }
            let CardCheck { work, commands: _, plans, flags } = check;
            self.recycle_card_list(CardPhase::Check, work);
            (plans, flags)
        } else {
            let scope = declared_scope(runs, marks);
            let plans = card_plans(copies, scope.as_ref());
            if plans.is_empty() {
                return;
            }
            let flags = self.reserve_card_flags(&plans);
            let list = self.record_card_check(copies, &plans, deferred, flags.as_ref());
            self.run_card_list(CardPhase::Check, list, submitted);
            (plans, flags)
        };
        let checking = super::torn::on();
        let mut checks = CheckedRuns::new();
        let mut fetches = Vec::new();
        let mut total = 0u64;
        let mut out_at = 0u32;
        for (copy, spans) in plans {
            let shadow = &copies[copy];
            let card = shadow.card().expect("planned card copy");
            let (start, _) = shadow.range();
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut expected = std::collections::BTreeMap::new();
            for span in spans {
                // SAFETY: this check's fence completed. A packed lease
                // excludes overlapping output ranges until consumption;
                // the fallback mapping is used only by the serial path.
                let flags = unsafe {
                    if let Some(lease) = &flag_lease {
                        let result = lease.read(out_at, span.pieces);
                        out_at += span.pieces;
                        result
                    } else {
                        std::slice::from_raw_parts(
                            (card.work.flags_data as *const u32).add(span.first as usize), span.pieces as usize)
                    }
                };
                for (i, flag) in flags.iter().enumerate() {
                    if super::torn::flagged(*flag) {
                        if checking {
                            expected.insert(span.first as usize + i, super::torn::folded(*flag));
                        }
                        let (from, to) = card_piece_of(start as usize, shadow.card_hash_size() as usize, &span, i as u32);
                        if let Some(picked) = &mut picked {
                            picked.flagged(shadow, from, to);
                        }
                        match ranges.last_mut() {
                            Some(last) if last.1 == from => last.1 = to,
                            _ => ranges.push((from, to)),
                        }
                    }
                }
            }
            for (lo, hi) in ranges {
                let (lo, hi) = if checking {
                    let check = CheckedRun::new(start as usize, shadow.card_hash_size() as usize, lo, hi, &expected);
                    let (lo, hi) = check.extent();
                    checks.insert((copy, lo, hi), check);
                    (lo, hi)
                } else {
                    (lo, hi)
                };
                fetches.push((copy, lo, hi));
                total += (hi - lo) as u64;
            }
        }
        if let Some(lease) = flag_lease {
            assert_eq!(out_at, lease.words);
            // Every flag/hash is now copied into the CPU plan. Reclaim
            // the result interval only after this consumption.
            drop(lease);
        }
        if let Some(picked) = picked {
            picked.finish_observation(copies);
        }
        if fetches.is_empty() {
            return;
        }
        // Source card_back grows fetched to the total changed bytes,
        // rounded to 16 MiB. A failed growth leaves the admission-time
        // allocation intact and pack_fetches divides the work safely.
        let pool = self.card_fetch_at_least(total).unwrap_or_else(||
            fail(format_args!("no admitted GPU readback pool")));
        let worker = if relay.is_empty() { None } else { super::relay::worker() };
        let mut private = worker.and_then(|_| self.relay_fetch());
        let mut pending: Option<super::relay::Receipt> = None;
        let mut relay_title = Vec::new();
        for batch in pack_fetches(&fetches, pool.size) {
            let logical = checking.then(|| batch.iter().map(|piece|
                checks[&(piece.copy, piece.lo, piece.hi)].logical(piece)).collect::<Vec<_>>());
            let mut retry = false;
            let (fetch_fence, completed) = loop {
                // On a copy-queue mismatch, retry on the checked direct
                // queue; its independent allocator/fence remain reusable.
                let mut fetch = private.as_mut().filter(|f| !retry || f.queue.is_none());
                let list = match fetch.as_mut() {
                    Some(fetch) => fetch.take_list(),
                    None => self.card_list(CardPhase::Fetch),
                };
                for piece in &batch {
                    // SAFETY: checked source ranges and disjoint packed
                    // destination ranges, live resources, buffer COPY_SOURCE
                    // promotion from COMMON and readback stays COPY_DEST.
                    unsafe {
                        list.list.CopyBufferRegion(&pool.buffer, piece.at,
                            &copies[piece.copy].buffer, piece.lo as u64, (piece.hi - piece.lo) as u64);
                    }
                }
                if deferred {
                    debug_assert!(fetch.as_ref().is_none_or(|f| f.queue.is_none()));
                    for (copy, ranges) in fetched_hash_runs(logical.as_deref().unwrap_or(&batch)) {
                        let shadow = &copies[copy];
                        let card = shadow.card().expect("fetched card copy");
                        let (start, _) = shadow.range();
                        let spans = compare_spans(start as usize, shadow.card_hash_size() as usize, Some(&ranges));
                        // SAFETY: all copies for this source have been
                        // recorded first; transition its promoted COPY_SOURCE
                        // state to SRV for mode 0. All owners survive the
                        // checked fence. No further title GPU work can run
                        // between fetching and updating the hash.
                        unsafe {
                            list.list.ResourceBarrier(&[transition(&shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                            card.work.bind(&list.list, &shadow.buffer);
                            for span in &spans {
                                card_pass(&list.list, span, shadow.card_hash_size() as u32, 0, 0);
                            }
                        }
                        compare_barrier(&list.list);
                    }
                }
                // Recording this list does not read staging or write the pool.
                // The preceding worker may land while we record, but it
                // must finish before another fetch can overwrite the pool.
                if let Some(receipt) = pending.take() {
                    COPY_BACK[1].fetch_add(receipt.finish(), Ordering::Relaxed);
                }
                let stamp = if let Some(fetch) = fetch {
                    fetch.run(self, list, *submitted)
                } else {
                    self.run_card_list(CardPhase::Fetch, list, submitted);
                    // SAFETY: a plain query of the checked queue fence.
                    (*submitted, unsafe { self.fence.GetCompletedValue() })
                };
                // SAFETY: the fetch fence was checked, and the submission
                // lock plus previous landing receipt exclude reuse.
                if !checking || unsafe { check_fetch_batch(&pool, &batch, &checks) } {
                    break stamp;
                }
                // Unlike the reference's concurrent queue, there is no
                // later flight that can repair a torn piece in this branch.
                // Retry on the same direct queue; never land a mismatch.
                if retry {
                    fail(format_args!("GPU readback failed hash validation after retry"));
                }
                retry = true;
            };
            let batch = logical.unwrap_or(batch);
            // SentRuns uses the main queue's send order, not the
            // private fence domain. This fetch ran after every send
            // up to submitted (no other title work can enter the queue).
            let fetched_after = *submitted;
            if let Some(worker) = worker {
                // Transfer the actual COM and Shadow owners, not just
                // the mapped address. Receipt joins before the next
                // GPU use or any later completion callback.
                let pool = pool.clone();
                let copies = copies.to_vec();
                let relay = relay.clone();
                pending = Some(worker.submit(fetch_fence, completed, move || {
                    let mut title = Vec::new();
                    land_fetch_batch(&pool, &copies, batch, &relay, fetched_after, &mut title)
                }));
            } else {
                let written = land_fetch_batch(&pool, copies, batch, relay, fetched_after, &mut relay_title);
                COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
            }
        }
        // All mapped writes, including the last batch, are complete
        // before finish_picked, marked/declared copy-back or title signals.
        if let Some(receipt) = pending.take() {
            COPY_BACK[1].fetch_add(receipt.finish(), Ordering::Relaxed);
        }
    }
}


/// Verify a real direct-queue private fence and allocator reuse without
/// consuming a main queue generation. The private fence and the send
/// generation are separate domains even if their values coincide.
#[test]
#[ignore = "requires D3D12 adapter; run alone"]
fn private_relay_fetch_has_own_fence_and_reuses_list() {
    let host = host().expect("D3D12 device");
    let submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle_baked_uploads();
    let before = *submitted;
    let mut fetch = host.make_relay_fetch(false).expect("private fetch state");
    let (upload, source) = host.upload_buffer(4).expect("upload");
    let (readback, target) = host.mapped_buffer(4, true).expect("readback");
    let mut handles = None;
    for value in [0x12345678u32, 0xabcdef01] {
        // SAFETY: the previous fetch completed; upload is mapped and
        // contains four bytes. The queue is not using it now.
        unsafe { source.cast::<u32>().write_unaligned(value) };
        let list = fetch.take_list();
        let current = (list.allocator.as_raw(), list.list.as_raw());
        if let Some(handles) = handles {
            assert_eq!(current, handles, "private allocator/list reused");
        }
        handles = Some(current);
        // SAFETY: live distinct upload/readback buffers, valid range,
        // open direct list, both buffers in their fixed heap states.
        unsafe { list.list.CopyBufferRegion(&readback, 0, &upload, 0, 4) };
        let previous = fetch.sent;
        let (sent, completed) = fetch.run(host, list, *submitted);
        assert_eq!(sent, previous + 1);
        assert!(completed >= sent && completed != u64::MAX);
        assert_eq!(*submitted, before);
        // SAFETY: readback is mapped and the private fence completed.
        assert_eq!(unsafe { target.cast::<u32>().read_unaligned() }, value);
    }
    assert_eq!(fetch.sent, 2);
}

/// A real default-heap GPU write on the direct queue, followed by
/// a read on the private copy queue. Both generations and resource
/// ownership remain valid for repeated transfers.
#[test]
#[ignore = "requires D3D12 adapter with high-priority copy queue; run alone"]
fn relay_copy_queue_reads_completed_direct_writes() {
    let host = host().expect("D3D12 device");
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle_baked_uploads();
    let mut fetch = host.make_relay_fetch(true).expect("relay fetch");
    assert!(fetch.queue.is_some(), "copy queue creation fell back");
    let (upload, source) = host.upload_buffer(16).expect("upload");
    let gpu = host.uav_buffer(16).expect("device-local buffer");
    let (readback, target) = host.mapped_buffer(16, true).expect("readback");
    for value in [0x12345678u32, 0xabcdef01] {
        // SAFETY: four mapped words; previous use has completed.
        unsafe { source.cast::<[u32; 4]>().write_unaligned([value; 4]) };
        let direct = host.card_list(CardPhase::Sync);
        // SAFETY: valid distinct buffers, open direct list, COMMON GPU
        // buffer promotes to COPY_DEST and decays after the checked fence.
        unsafe { direct.list.CopyBufferRegion(&gpu, 0, &upload, 0, 16) };
        host.run_card_list(CardPhase::Sync, direct, &mut submitted);
        let before = *submitted;
        let copy = fetch.take_list();
        // SAFETY: the copy buffer is COMMON after the direct fence;
        // it promotes to COPY_SOURCE on the copy queue.
        unsafe { copy.list.CopyBufferRegion(&readback, 0, &gpu, 0, 16) };
        let (sent, completed) = fetch.run(host, copy, before);
        assert!(completed >= sent && completed != u64::MAX);
        assert_eq!(*submitted, before);
        // SAFETY: the private copy fence completed, readback live.
        let got = unsafe { target.cast::<[u32; 4]>().read_unaligned() };
        assert_eq!(got, [value; 4]);
        // The source's folded hash sees the same completed bytes.
        // SAFETY: same checked readback mapping and length.
        let bytes = unsafe { std::slice::from_raw_parts(target, 16) };
        assert_eq!(super::torn::hash(bytes), super::torn::hash(&[value.to_le_bytes(); 4].concat()));
    }
    assert_eq!(fetch.sent, 2);
}

/// Actual scatter/copy commands use the owned snapshot, even when
/// staging changes after take. Also exercise the normal warmup sync.
#[test]
#[ignore = "requires VICTORY_CARD_BUFFERS=1 VICTORY_CARD_BUFFERS_MIN_KB=64 VICTORY_SEND_OWN=1 and D3D12; run alone"]
fn owned_card_send_keeps_original_bytes_after_staging_changes() {
    let host = host().expect("D3D12 device");
    let p = crate::watch::tests::pool(2);
    let len = 2 * PIECE;
    crate::shadow::register(p, p as u64, len as u64);
    let shadow = crate::shadow::find(p as u64, len as u64).expect("registered copy");
    let card = shadow.card().expect("device-local copy");
    // SAFETY: owned title pages, initialize the first 32 bytes.
    unsafe { (p as *mut u8).write_bytes(0, 32) };
    let warm = host.new_list().unwrap();
    shadow.refresh(&warm, p as u64, len as u64);
    warm.close();
    host.execute(&[&warm]);
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle_baked_uploads();
    shadow.write_host(p as u64, &[1, 2, 3, 4]);
    shadow.write_host((p + 8) as u64, &[5, 6, 7, 8]);
    shadow.write_host((p + 17) as u64, &[9, 10, 11]);
    let dirty = [(&shadow, card, vec![(0, 4), (8, 12), (17, 20)])];
    let own = card_send::take(host, &dirty).expect("owned upload and scatter pipeline");
    shadow.write_host(p as u64, &[99; 20]);
    let sync = host.card_list(CardPhase::Sync);
    // SAFETY: the owned bytes and exact destination owners are retained
    // through the checked sync fence; all buffers start in COMMON.
    unsafe { own.record(&sync.list, &dirty) };
    host.run_card_list(CardPhase::Sync, sync, &mut submitted);
    own.give_back();
    let (readback, mapped) = host.mapped_buffer(32, true).unwrap();
    let fetch = host.card_list(CardPhase::Fetch);
    // SAFETY: both resources live, source decayed to COMMON, valid range.
    unsafe { fetch.list.CopyBufferRegion(&readback, 0, &shadow.buffer, 0, 32) };
    host.run_card_list(CardPhase::Fetch, fetch, &mut submitted);
    // SAFETY: mapped bytes from the completed readback.
    let got = unsafe { std::slice::from_raw_parts(mapped, 32) };
    let mut expected = [0u8; 32];
    expected[..4].copy_from_slice(&[1, 2, 3, 4]);
    expected[8..12].copy_from_slice(&[5, 6, 7, 8]);
    expected[17..20].copy_from_slice(&[9, 10, 11]);
    assert_eq!(got, expected);
    drop(submitted);
    crate::shadow::unregister(p, p as u64);
}

/// Exercise the real tail path and verify that the check shares the
/// title fence on both a changed and a clean submission.
#[test]
#[ignore = "requires VICTORY_CARD_BUFFERS=1 VICTORY_CARD_BUFFERS_MIN_KB=64 VICTORY_CARD_CHECK_TAIL=1, ranges/timing off and D3D12; run alone"]
fn card_check_tail_shares_title_fence() {
    let host = host().expect("D3D12 device");
    assert!(check_tail_enabled() && !ranges_enabled());
    #[cfg(feature = "diagnostics")]
    assert!(!timing());
    let p = crate::watch::tests::pool(2);
    let len = 2 * PIECE;
    crate::shadow::register(p, p as u64, len as u64);
    let shadow = crate::shadow::find(p as u64, len as u64).expect("registered copy");
    let card = shadow.card().expect("device-local copy");
    // SAFETY: two mapped title pages belong to this test.
    unsafe { (p as *mut u32).write_unaligned(0) };
    let warm = host.new_list().unwrap();
    shadow.refresh(&warm, p as u64, len as u64);
    warm.close();
    host.execute(&[&warm]);

    let current = || *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    let before = current();
    let (upload, data) = host.upload_buffer(4).unwrap();
    let value = 0x24681357u32;
    // SAFETY: four mapped upload bytes.
    unsafe { data.cast::<u32>().write_unaligned(value) };
    let changed = host.new_list().unwrap();
    changed.keep(&upload);
    // SAFETY: open command list, live buffers, and in-bounds copy.
    unsafe { changed.list.CopyBufferRegion(&shadow.buffer, 0, &upload, 0, 4) };
    changed.close();
    host.execute(&[&changed]);
    assert_eq!(current() - before, 2, "one title/check fence and one fetch fence");
    assert_eq!(shadow.read_host(p as u64, 1), [value]);
    // SAFETY: flags were read back before the completed title fence.
    assert_eq!(unsafe { (card.work.flags_data as *const u32).read() } & 1, 1);

    let clean = host.new_list().unwrap();
    clean.close();
    let before = current();
    host.execute(&[&clean]);
    assert_eq!(current() - before, 1, "no separate check fence or fetch");
    // SAFETY: the next check's flags have completed too.
    assert_eq!(unsafe { (card.work.flags_data as *const u32).read() } & 1, 0);
    assert_eq!(shadow.read_host(p as u64, 1), [value]);
    crate::shadow::unregister(p, p as u64);
}

/// Two checked plans over the same copy retain distinct GPU results.
/// Keep the first result lease live through a second check, even
/// though the per-copy differs scratch is reused by the direct queue.
#[test]
#[ignore = "requires VICTORY_CARD_BUFFERS=1 VICTORY_CARD_BUFFERS_MIN_KB=64 and D3D12; run alone"]
fn packed_card_flags_survive_a_later_check() {
    let host = host().expect("D3D12 device");
    let p = crate::watch::tests::pool(2);
    let len = 2 * PIECE;
    crate::shadow::register(p, p as u64, len as u64);
    let shadow = crate::shadow::find(p as u64, len as u64).expect("registered copy");
    assert!(shadow.card().is_some());
    // SAFETY: this test owns the mapped title range.
    unsafe { (p as *mut u32).write_unaligned(0) };
    let warm = host.new_list().unwrap();
    shadow.refresh(&warm, p as u64, len as u64);
    warm.close();
    host.execute(&[&warm]);

    let arena = FlagArena::make(host).expect("packed flags");
    let copies = [shadow.clone()];
    let plans = card_plans(&copies, None);
    let words = plan_words(&plans).unwrap();
    let first = arena.take(words).unwrap();
    let second = arena.take(words).unwrap();
    assert_eq!(first.at + words, second.at);
    let (upload, data) = host.upload_buffer(4).unwrap();
    // SAFETY: this mapped upload contains four bytes.
    unsafe { data.cast::<u32>().write_unaligned(0x10203040) };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle_baked_uploads();
    let write = host.card_list(CardPhase::Sync);
    // SAFETY: live buffers and valid range; the copy promotes from COMMON.
    unsafe { write.list.CopyBufferRegion(&shadow.buffer, 0, &upload, 0, 4) };
    host.run_card_list(CardPhase::Sync, write, &mut submitted);
    let check = host.record_card_check(&copies, &plans, false, Some(&first));
    host.run_card_list(CardPhase::Check, check, &mut submitted);
    // SAFETY: the first checked fence has completed.
    let changed = unsafe { first.read(0, words) }.to_vec();
    assert_eq!(changed[0] & 1, 1);
    let check = host.record_card_check(&copies, &plans, false, Some(&second));
    host.run_card_list(CardPhase::Check, check, &mut submitted);
    // SAFETY: both checks completed, and the leases are disjoint.
    unsafe {
        assert_eq!(first.read(0, words), changed.as_slice());
        assert!(second.read(0, words).iter().all(|&flag| flag & 1 == 0));
    }
    drop(first);
    assert_eq!(arena.cursor.lock().unwrap().taken, 2 * words);
    drop(second);
    assert_eq!(arena.cursor.lock().unwrap().taken, 0);
    drop(submitted);
    crate::shadow::unregister(p, p as u64);
}

/// Exercise source-length hashing in the production path on a real
/// adapter. The last owned byte changes, but a separate padding-only
/// write must not expand the CPU mirror or produce a changed flag.
#[test]
#[ignore = "requires VICTORY_CARD_BUFFERS=1 VICTORY_CARD_BUFFERS_MIN_KB=64 VICTORY_CARD_HASH_LEN=1 and D3D12; run alone"]
fn logical_card_hash_length_excludes_padding_writes() {
    let host = host().expect("D3D12 device");
    let p = crate::watch::tests::pool(2);
    let len = PIECE + 32;
    crate::shadow::register(p, p as u64, len as u64);
    let shadow = crate::shadow::find(p as u64, len as u64).expect("registered copy");
    let card = shadow.card().expect("device-local copy");
    assert_eq!(shadow.card_hash_size(), len as u64);
    assert_eq!(card.size, (2 * PIECE) as u64);
    let submit = |at: usize, value: u32, refresh: bool| {
        let (upload, data) = host.upload_buffer(4).unwrap();
        // SAFETY: mapped upload contains four bytes.
        unsafe { data.cast::<u32>().write_unaligned(value) };
        let list = host.new_list().unwrap();
        if refresh {
            shadow.refresh(&list, p as u64, len as u64);
        }
        list.keep(&upload);
        // SAFETY: four bytes inside the live allocation and an open list.
        unsafe { list.list.CopyBufferRegion(&shadow.buffer, at as u64, &upload, 0, 4) };
        list.close();
        host.execute(&[&list]);
        // SAFETY: the second flag is initialized by the completed full
        // logical comparison; no GPU work is in flight.
        unsafe { (card.work.flags_data as *const u32).add(1).read() & 1 }
    };
    // SAFETY: owned, mapped title bytes.
    unsafe { ((p + len - 4) as *mut u32).write_unaligned(0) };
    assert_eq!(submit(len - 4, 0x76543210, true), 1);
    assert_eq!(shadow.read_host((p + len - 4) as u64, 1), [0x76543210]);
    assert_eq!(submit(len + 32, 0x11223344, false), 0);
    assert_eq!(shadow.read_host((p + len - 4) as u64, 1), [0x76543210]);
    crate::shadow::unregister(p, p as u64);
}

/// An actual submission changes an undeclared card buffer. Mode 1
/// must fetch it and commit the new hash exactly once; an empty subsequent
/// submission has no refresh/upload to hide a stale baseline.
#[test]
#[ignore = "requires VICTORY_CARD_BUFFERS=1 VICTORY_CARD_BUFFERS_MIN_KB=64 VICTORY_CARD_LOOK_ONLY=1 and D3D12; run alone"]
fn deferred_card_hash_does_not_repeat_a_fetched_change() {
    let host = host().expect("D3D12 device");
    assert!(deferred_hashes());
    let p = crate::watch::tests::pool(2);
    let len = 2 * PIECE;
    crate::shadow::register(p, p as u64, len as u64);
    let shadow = crate::shadow::find(p as u64, len as u64).expect("registered copy");
    let card = shadow.card().expect("device-local copy");
    let value = 0x13572468u32;
    // SAFETY: this test owns the two mapped title pages.
    unsafe { (p as *mut u32).write_unaligned(0) };
    let (upload, data) = host.upload_buffer(4).unwrap();
    // SAFETY: the upload mapping holds four bytes.
    unsafe { data.cast::<u32>().write_unaligned(value) };
    let list = host.new_list().unwrap();
    shadow.refresh(&list, p as u64, len as u64);
    list.keep(&upload);
    // SAFETY: an open list, live distinct resources, valid four-byte range.
    unsafe { list.list.CopyBufferRegion(&shadow.buffer, 0, &upload, 0, 4) };
    list.close();
    host.execute(&[&list]);
    assert_eq!(shadow.read_host(p as u64, 1), [value]);
    // SAFETY: the flag readback is mapped and the check fence has completed.
    assert_eq!(unsafe { (card.work.flags_data as *const u32).read() } & 1, 1);
    let empty = host.new_list().unwrap();
    empty.close();
    host.execute(&[&empty]);
    // SAFETY: the second full comparison has also completed.
    assert_eq!(unsafe { (card.work.flags_data as *const u32).read() } & 1, 0);
    assert_eq!(shadow.read_host(p as u64, 1), [value]);
    crate::shadow::unregister(p, p as u64);
}

/// Reuse the exact three command allocators across two fenced submissions.
/// Requires an actual adapter; the normal shader/range tests remain in CI.
#[test]
#[ignore = "requires D3D12; run alone"]
fn card_preludes_reuse_only_completed_lists() {
    let host = host().expect("D3D12 device");
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle_baked_uploads();
    for phase in [CardPhase::Sync, CardPhase::Check, CardPhase::Fetch] {
        let first = host.card_list(phase);
        let allocator = first.allocator.as_raw();
        let command = first.list.as_raw();
        let before = *submitted;
        host.run_card_list(phase, first, &mut submitted);
        assert_eq!(*submitted, before + 1);
        // SAFETY: a plain fence query; the completed work owns the list.
        assert!(unsafe { host.fence.GetCompletedValue() } >= *submitted);
        let again = host.card_list(phase);
        assert_eq!(again.allocator.as_raw(), allocator);
        assert_eq!(again.list.as_raw(), command);
        host.run_card_list(phase, again, &mut submitted);
    }
}

/// Tests the real reference shader in three modes and fetches only the
/// pieces it marked as changed. Run on a D3D12 adapter, with debug validation.
#[test]
#[ignore = "requires a D3D12 adapter and GPU compute/readback; run alone"]
fn card_compare_detects_and_fetches_changed_pieces() {
    let host = host().expect("D3D12 device");
    let (root, pso) = compare_pipeline(&host.device).unwrap();
    let size = 3 * PIECE;
    let start = 0x1000usize;
    let span = card_span_of(start, 0, size);
    let count = span.pieces as usize;
    let before: Vec<u8> = (0..size).map(|i| (i * 7 + i / 251) as u8).collect();
    let mut after = before.clone();
    after[PIECE] ^= 0x81;
    after[size - 1] ^= 0x31;
    let inputs = [&before, &after].map(|bytes| {
        let (upload, data) = host.upload_buffer(size as u64).unwrap();
        // SAFETY: upload mapping has size bytes, disjoint from the input.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, size) };
        let card = host.default_buffer(size as u64).unwrap();
        (upload, card)
    });
    let hashes = uav_buffer(&host.device, (count * 16) as u64);
    let results = uav_buffer(&host.device, (3 * count * 4) as u64);
    let (back, mapped) = host.mapped_buffer((3 * count * 4) as u64, true).unwrap();
    let list = host.new_list().unwrap();

    // SAFETY: all buffers, pipeline and root signature live across the
    // synchronous submission; resource states are transitioned explicitly.
    unsafe {
        for (upload, card) in &inputs {
            list.list.CopyBufferRegion(card, 0, upload, 0, size as u64);
            list.list.ResourceBarrier(&[transition(card, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        }
        list.list.SetPipelineState(&pso);
        list.list.SetComputeRootSignature(&root);
        list.list.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
        list.list.SetComputeRootUnorderedAccessView(3, results.GetGPUVirtualAddress());
        list.list.SetComputeRootShaderResourceView(1, inputs[0].1.GetGPUVirtualAddress());
        card_pass(&list.list, &span, size as u32, 0, 0);
        compare_barrier(&list.list);
        list.list.SetComputeRootShaderResourceView(1, inputs[1].1.GetGPUVirtualAddress());
        // Compare, compare-and-update, compare again against the new hashes.
        for (i, mode) in [1, 2, 1].into_iter().enumerate() {
            card_pass(&list.list, &span, size as u32, (i * count) as u32, mode);
            compare_barrier(&list.list);
        }
        list.list.ResourceBarrier(&[transition(&results,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.list.CopyBufferRegion(&back, 0, &results, 0, (3 * count * 4) as u64);
        list.list.ResourceBarrier(&[transition(&inputs[1].1,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    }
    list.close();
    host.execute(&[&list]);

    // SAFETY: the readback mapping is live and the checked fence completed.
    let flags = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), 3 * count) };
    let mut fetched = Vec::new();
    for i in 0..count {
        let (lo, hi) = card_hashed_of(start, size, i);
        let changed = before[lo..hi] != after[lo..hi];
        let expected = super::torn::hash(&after[lo..hi]) << 1 | u32::from(changed);
        assert_eq!(flags[i], expected, "first compare, piece {i}");
        assert_eq!(flags[count + i], expected, "update compare, piece {i}");
        assert_eq!(flags[2 * count + i], expected & !1, "updated hash, piece {i}");
        if flags[i] & 1 != 0 {
            fetched.push(card_piece_of(start, size, &span, i as u32));
        }
    }
    assert_eq!(fetched.len(), 2);
    let total: usize = fetched.iter().map(|(lo, hi)| hi - lo).sum();
    assert!(total < size);
    let (fetch, mapped) = host.mapped_buffer(total as u64, true).unwrap();
    let read = host.new_list().unwrap();
    let mut offset = 0;
    for &(lo, hi) in &fetched {
        // SAFETY: checked non-overlapping destinations and a live COPY_SOURCE buffer.
        unsafe {
            read.list.CopyBufferRegion(&fetch, offset as u64, &inputs[1].1,
                lo as u64, (hi - lo) as u64);
        }
        offset += hi - lo;
    }
    read.close();
    host.execute(&[&read]);
    // SAFETY: all requested readback copies have completed.
    let landed = unsafe { std::slice::from_raw_parts(mapped, total) };
    let expected: Vec<u8> = fetched.iter().flat_map(|&(lo, hi)| after[lo..hi].iter().copied()).collect();
    assert_eq!(landed, expected);
}
