//! Synchronous GPU changed-piece compare, adapted from the reference renderer.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, host.rs (CARD_COMPARE, card_pass,
//! make_card_work), shadow.rs (CardSpan, range helpers and tests), host/torn.rs
//! (CPU hash). The optional two-sided Shadow path keeps existing CPU
//! copy-back and fences; relay/async landing are deliberately not imported.

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
#[cfg(test)]
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

#[cfg(test)]
const MUL: [u32; 4] = [0x0100_0193, 0x9E37_79B1, 0x85EB_CA77, 0xC2B2_AE3D];
#[cfg(test)]
const SEED: [u32; 4] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];

#[cfg(test)]
fn step(h: &mut [u32; 4], with: [u32; 4]) {
    let x: [u32; 4] = std::array::from_fn(|i| (h[i] ^ with[i]).wrapping_mul(MUL[i]));
    *h = std::array::from_fn(|i| x[(i + 1) % 4] ^ (x[i] >> 15));
}

/// The 64 threads' hashes as one, folded to 31 bits.
#[cfg(test)]
fn finish(parts: &[[u32; 4]; 64]) -> u32 {
    let mut h = parts[0];
    for part in &parts[1..] {
        step(&mut h, *part);
    }
    let f = h[0] ^ h[1].wrapping_mul(3) ^ h[2].wrapping_mul(5) ^ h[3].wrapping_mul(7);
    f & 0x7FFF_FFFF
}

#[cfg(test)]
fn seeds() -> [[u32; 4]; 64] {
    std::array::from_fn(|thread| SEED.map(|s| s.wrapping_add(thread as u32)))
}

#[cfg(test)]
fn tail(parts: &mut [[u32; 4]; 64], bytes: &[u8]) {
    for (thread, chunk) in bytes.as_chunks::<16>().0.iter().enumerate() {
        let with = std::array::from_fn(|i| u32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]));
        step(&mut parts[thread], with);
    }
}

#[cfg(test)]
fn hash_plain(bytes: &[u8]) -> u32 {
    let mut parts = seeds();
    for row in bytes.chunks(1024) {
        tail(&mut parts, row);
    }
    finish(&parts)
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
        // SAFETY: plain object creation.
        unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
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
        // Only completed work is ever put back. Keep one per phase;
        // it is impossible to overwrite another in-flight allocator.
        let mut lists = self.card_lists.lock().unwrap_or_else(|e| e.into_inner());
        assert!(lists[phase as usize].is_none(), "card prelude slot already occupied");
        lists[phase as usize] = Some(work);
    }
}

impl Host {
    /// Adapt the source card_before: copy CPU-dirty ranges, then hash the
    /// joined pieces on the device before the title's lists execute.
    pub(super) fn card_before(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>], submitted: &mut u64) {
        let dirty: Vec<_> = copies.iter().filter_map(|shadow| {
            let card = shadow.card()?;
            let ranges = card.take_dirty();
            (!ranges.is_empty()).then_some((shadow, card, ranges))
        }).collect();
        if dirty.is_empty() {
            return;
        }
        let list = self.card_list(CardPhase::Sync);
        for (shadow, card, ranges) in dirty {
            let (start, _) = shadow.range();
            let spans = joined_piece_runs(ranges.iter().map(|&(from, to)|
                card_span_of(start as usize, from, to)).collect());
            // SAFETY: ranges are in both live buffers. Buffers have decayed
            // to COMMON after the last separately submitted list; copies
            // promote the destination to COPY_DEST, then transition to SRV.
            unsafe {
                for (from, to) in ranges {
                    list.list.CopyBufferRegion(&shadow.buffer, from as u64,
                        &card.staging, from as u64, (to - from) as u64);
                }
                list.list.ResourceBarrier(&[transition(&shadow.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                card.work.bind(&list.list, &shadow.buffer);
                for span in &spans {
                    card_pass(&list.list, span, card.size as u32, 0, 0);
                }
            }
            compare_barrier(&list.list);
        }
        self.run_card_list(CardPhase::Sync, list, submitted);
    }

    /// Adapt source card_back: compare on the device, read flags after the
    /// fence, fetch flagged pieces, then make their bytes available to the
    /// existing CPU snapshot/marked-page merge. Compare all live card
    /// copies conservatively, including unbounded descriptor writes.
    pub(super) fn card_back(&self, copies: &[std::sync::Arc<crate::shadow::Shadow>], submitted: &mut u64) {
        if copies.is_empty() {
            return;
        }
        let list = self.card_list(CardPhase::Check);
        for shadow in copies {
            let Some(card) = shadow.card() else { continue };
            let (start, _) = shadow.range();
            let span = card_span_of(start as usize, 0, card.size as usize);
            // SAFETY: the title submission has finished. The buffers and
            // UAVs are live, in COMMON, and promoted by their respective use.
            unsafe {
                card.work.bind(&list.list, &shadow.buffer);
                card_pass(&list.list, &span, card.size as u32, 0, 2);
                compare_barrier(&list.list);
                list.list.ResourceBarrier(&[transition(&card.work.differs,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.list.CopyBufferRegion(&card.work.flags, 0, &card.work.differs, 0,
                    u64::from(span.pieces) * 4);
            }
        }
        self.run_card_list(CardPhase::Check, list, submitted);
        let mut fetches = Vec::new();
        let mut total = 0u64;
        for (copy, shadow) in copies.iter().enumerate() {
            let Some(card) = shadow.card() else { continue };
            let (start, _) = shadow.range();
            let span = card_span_of(start as usize, 0, card.size as usize);
            // SAFETY: the persistent flag readback is large enough for
            // this span, and the checked compare fence has completed.
            let flags = unsafe { std::slice::from_raw_parts(card.work.flags_data as *const u32, span.pieces as usize) };
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for (i, flag) in flags.iter().enumerate() {
                if flag & 1 != 0 {
                    let (from, to) = card_piece_of(start as usize, card.size as usize, &span, i as u32);
                    match ranges.last_mut() {
                        Some(last) if last.1 == from => last.1 = to,
                        _ => ranges.push((from, to)),
                    }
                }
            }
            for (lo, hi) in ranges {
                fetches.push((copy, lo, hi));
                total += (hi - lo) as u64;
            }
        }
        if fetches.is_empty() {
            return;
        }
        // Source card_back grows fetched to the total changed bytes,
        // rounded to 16 MiB. A failed growth leaves the admission-time
        // allocation intact and pack_fetches divides the work safely.
        let pool = self.card_fetch_at_least(total).unwrap_or_else(||
            fail(format_args!("no admitted GPU readback pool")));
        for batch in pack_fetches(&fetches, pool.size) {
            let list = self.card_list(CardPhase::Fetch);
            for piece in &batch {
                // SAFETY: checked source ranges and disjoint packed
                // destination ranges, live resources, buffer COPY_SOURCE
                // promotion from COMMON and readback stays COPY_DEST.
                unsafe {
                    list.list.CopyBufferRegion(&pool.buffer, piece.at,
                        &copies[piece.copy].buffer, piece.lo as u64, (piece.hi - piece.lo) as u64);
                }
            }
            self.run_card_list(CardPhase::Fetch, list, submitted);
            for piece in batch {
                // SAFETY: the checked readback fence completed; this
                // owner pins its mapping and the submission lock keeps
                // the pool contents stable until the next batch.
                let bytes = unsafe { std::slice::from_raw_parts(
                    (pool.data as *const u8).add(piece.at as usize), piece.hi - piece.lo) };
                copies[piece.copy].land_card(piece.lo, bytes);
            }
        }
    }
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
        let expected = hash_plain(&after[lo..hi]) << 1 | u32::from(changed);
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
