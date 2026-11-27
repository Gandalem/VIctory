//! Submission-owned CPU-to-GPU sends, adapted from
//! DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e host/send.rs.
//! A submission takes the bytes before recording its GPU list. Short
//! word-aligned runs use the reference scatter shader; other runs use
//! CopyBufferRegion. This branch still waits for every sync list. Ownership
//! is separated now without allowing concurrent title submissions.

use super::*;
use crate::shadow::{Card, Shadow};
use std::sync::Arc;

pub(super) type Dirty<'a> = (&'a Arc<Shadow>, &'a Card, Vec<(usize, usize)>);

pub(super) fn own() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_OWN").is_some_and(|v| v == "1"))
}

/// Runs up to this many bytes go as pairs.
const SHORT: usize = 256;

/// Whether a run of `len` bytes at offset `at` of a copy goes as pairs.
fn short(at: usize, len: usize) -> bool {
    len <= SHORT && at.is_multiple_of(4) && len.is_multiple_of(4)
}

const SCATTER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) { uint at; uint count; };
ByteAddressBuffer pairs : register(t0);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint2 p = pairs.Load2(at + id.x * 8u);
    card.Store(p.x, p.y);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn compile_scatter() -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: static source and names live through the call; output is ours.
    unsafe {
        D3DCompile(SCATTER.as_ptr().cast(), SCATTER.len(), s!("card scatter"),
            None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile_scatter().ok()?;
        // SAFETY: the blob remains live through creation; root pointer is
        // borrowed in the description, just as for the compare pipeline.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).ok()?;
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            }).ok()?;
            name(&pso, || "victory: changed words sent to the card".into());
            Some(Pass { root, pso })
        }
    }).as_ref()
}

/// Source send::Memory, with a lower idle budget for this branch. A spare
/// is available only after its previous sync fence completed.
struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

static SPARE: Mutex<Vec<Memory>> = Mutex::new(Vec::new());
const KEPT: u64 = 128 << 20;
const MAX_ACTIVE: u64 = 512 << 20;

fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if crate::shadow::memory_pressured() {
            spare.clear();
            return None;
        }
        let fit = spare.iter().enumerate().filter(|(_, m)| m.size >= size)
            .min_by_key(|(_, m)| m.size).map(|(i, _)| i);
        if let Some(i) = fit {
            return Some(spare.swap_remove(i));
        }
    }
    let size = size.max(8 << 20).checked_next_power_of_two()?;
    if size > MAX_ACTIVE {
        return None;
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "bytes a submission sends up".into());
    Some(Memory { buffer, data: data as usize, size })
}

fn give_back(memory: Memory) {
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    if crate::shadow::memory_pressured() {
        spare.clear();
        return;
    }
    spare.push(memory);
    spare.sort_by_key(|m| std::cmp::Reverse(m.size));
    while spare.iter().map(|m| m.size).sum::<u64>() > KEPT {
        spare.remove(0);
    }
}

/// Source send::Copy: long byte runs plus packed offset/word pairs.
struct Copy {
    runs: Vec<(usize, u64, usize)>,
    pairs: (u64, u32),
}

pub(super) struct Own {
    memory: Memory,
    copies: Vec<Copy>,
    pass: &'static Pass,
}

/// Source send::take's layout, separated for range-only regression tests.
/// Refuse overlapping, empty or reversed dirty ranges rather than let
/// two scatter threads race. Bound the allocation and 32-bit shader offsets.
fn layout<'a>(all: impl Iterator<Item = &'a [(usize, usize)]>) -> Option<(u64, Vec<Copy>)> {
    let mut at = 0u64;
    let mut copies = Vec::new();
    for ranges in all {
        let mut runs = Vec::new();
        let mut end = 0;
        let mut words = 0u64;
        for &(a, b) in ranges {
            if a < end || a >= b || b > u32::MAX as usize {
                return None;
            }
            end = b;
            if short(a, b - a) {
                words = words.checked_add(((b - a) / 4) as u64)?;
            } else {
                runs.push((a, at, b - a));
                at = at.checked_add(((b - a) as u64).next_multiple_of(16))?;
            }
        }
        copies.push(Copy { runs, pairs: (at, u32::try_from(words).ok()?) });
        at = at.checked_add(words.checked_mul(8)?.next_multiple_of(16))?;
        if at > MAX_ACTIVE {
            return None;
        }
    }
    (at != 0).then_some((at, copies))
}

/// Take an immutable upload snapshot while the submitting thread owns
/// the staging mappings. This is the source's take, without its
/// asynchronous/parallel or BAR allocation dependencies.
pub(super) fn take(host: &Host, dirty: &[Dirty<'_>]) -> Option<Own> {
    if !own() {
        return None;
    }
    let pass = pass(host)?;
    let (size, copies) = layout(dirty.iter().map(|(_, _, ranges)| ranges.as_slice()))?;
    // All validation before allocating or reading CPU-side data.
    if dirty.iter().any(|(_, card, ranges)| ranges.iter().any(|&(_, b)| b as u64 > card.size)) {
        return None;
    }
    let memory = memory(host, size)?;
    for ((shadow, _, ranges), copy) in dirty.iter().zip(&copies) {
        let from = shadow.card_data();
        // SAFETY: validation bounds the source, layout bounds disjoint
        // destinations; under submitted only this thread writes staging.
        unsafe {
            for &(a, to, n) in &copy.runs {
                std::ptr::copy_nonoverlapping(from.add(a), (memory.data as *mut u8).add(to as usize), n);
            }
            let mut to = (memory.data as *mut u8).add(copy.pairs.0 as usize).cast::<[u32; 2]>();
            for &(a, b) in ranges.iter().filter(|&&(a, b)| short(a, b - a)) {
                for offset in (a..b).step_by(4) {
                    to.write([offset as u32, from.add(offset).cast::<u32>().read_unaligned()]);
                    to = to.add(1);
                }
            }
        }
    }
    Some(Own { memory, copies, pass })
}

impl Own {
    /// Source send::Own::record. First long copies, then the short-word
    /// scatter, then the SRV transitions for the existing hash passes.
    /// # Safety
    /// The same dirty owners and ranges passed to take, an open sync list,
    /// card buffers in COMMON after the last fence; keep self and owners
    /// through the new checked fence.
    pub(super) unsafe fn record(&self, list: &ID3D12GraphicsCommandList, dirty: &[Dirty<'_>]) {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let (mut first, mut last) = (Vec::new(), Vec::new());
        for ((shadow, _, _), copy) in dirty.iter().zip(&self.copies) {
            for &(a, from, n) in &copy.runs {
                // SAFETY: caller's live resources and validated offsets.
                unsafe { list.CopyBufferRegion(&shadow.buffer, a as u64, &self.memory.buffer, from, n as u64) };
            }
            let mut state = if copy.runs.is_empty() { D3D12_RESOURCE_STATE_COMMON } else { dest };
            if copy.pairs.1 > 0 {
                first.push(transition(&shadow.buffer, state, write));
                state = write;
            }
            if !copy.runs.is_empty() || copy.pairs.1 > 0 {
                last.push(transition(&shadow.buffer, state, read));
            }
        }
        // SAFETY: caller's open list; all copy/pair ranges are disjoint,
        // barriers match the copy or UAV state, and the PSO and upload
        // owner remain live through completion.
        unsafe {
            if !first.is_empty() {
                list.ResourceBarrier(&first);
                list.SetComputeRootSignature(&self.pass.root);
                list.SetPipelineState(&self.pass.pso);
                list.SetComputeRootShaderResourceView(1, self.memory.buffer.GetGPUVirtualAddress());
                for ((shadow, _, _), copy) in dirty.iter().zip(&self.copies).filter(|(_, c)| c.pairs.1 > 0) {
                    list.SetComputeRootUnorderedAccessView(2, shadow.buffer.GetGPUVirtualAddress());
                    const MOST: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
                    let (at, words) = copy.pairs;
                    let mut done = 0;
                    while done < words {
                        let n = (words - done).min(MOST);
                        let constants = [at as u32 + done * 8, n];
                        list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.Dispatch(n.div_ceil(64), 1, 1);
                        done += n;
                    }
                }
            }
            if !last.is_empty() {
                list.ResourceBarrier(&last);
            }
        }
    }

    /// Only called after a checked completion fence.
    pub(super) fn give_back(self) {
        give_back(self.memory);
    }
}

#[test]
fn reference_card_scatter_compiles() {
    let blob = compile_scatter().expect("compile source scatter shader");
    // SAFETY: live compiled shader.
    assert!(unsafe { blob.GetBufferSize() } > 0);
}

#[test]
fn owned_send_layout_separates_words_and_byte_runs() {
    let a = [(0, 4), (8, 16), (17, 20), (32, 288), (512, 1024)];
    let b = [(4, 8)];
    let (size, copies) = layout([a.as_slice(), b.as_slice()].into_iter()).unwrap();
    assert_eq!(copies[0].runs, [(17, 0, 3), (512, 16, 512)]);
    assert_eq!(copies[0].pairs, (528, 67));
    assert_eq!(copies[1].pairs, (1072, 1));
    assert_eq!(size, 1088);
    assert!(layout([&[(0, 4), (2, 8)][..]].into_iter()).is_none());
    assert!(layout([&[(1, 1)][..]].into_iter()).is_none());
    assert!(layout([&[(0, MAX_ACTIVE as usize + 4)][..]].into_iter()).is_none());
    assert!(layout(std::iter::empty::<&[(usize, usize)]>()).is_none());
}
