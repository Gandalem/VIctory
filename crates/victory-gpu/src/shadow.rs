//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

fn merge_writeback(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = 0;
    for ((title_byte, host_byte), old_byte) in title.iter_mut().zip(host).zip(old) {
        let mask = 0u8.wrapping_sub(u8::from(*host_byte != *old_byte));
        let delta = (*title_byte ^ *host_byte) & mask;
        *title_byte ^= delta;
        *host_byte = *title_byte;
        changed |= delta;
    }
    changed != 0
}

#[test]
fn writeback_merge_matches_bytewise_oracle() {
    for len in [0, 1, 63, 64, 65, 193, 65_536] {
        let old = vec![0x11; len];
        for stride in [1, 5, 67, len + 1] {
            let mut title = old.clone();
            let mut host = old.clone();
            for index in (0..len).step_by(3) {
                title[index] = 0x22;
            }
            for index in (0..len).step_by(stride) {
                host[index] = if index % 2 == 0 { 0x22 } else { 0x33 };
            }
            let expected: Vec<_> = host
                .iter()
                .zip(&old)
                .zip(&title)
                .map(|((&gpu, &previous), &cpu)| if gpu == previous { cpu } else { gpu })
                .collect();
            let changed = expected != title;
            assert_eq!(merge_writeback(&mut title, &mut host, &old), changed);
            assert_eq!(title, expected);
            assert_eq!(host, expected);
        }
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// First bit of the view-written page bitmap for a tracked copy.
    marks: Option<u32>,
    loose: bool,
    // GPU-only changes cannot always be reconstructed from title memory.
    may_have_gpu_writes: AtomicBool,
    used_ms: AtomicU64,
    used_submission: AtomicU64,
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Only returned snapshots may be retained or discarded here. Snapshots
/// still needed by an in-flight copy-back remain owned by their Shadow.
const MIB: usize = 1 << 20;
const MAX_POOL_BYTES: usize = 2048 * MIB;
const MAX_POOL_ENTRIES: usize = 4096;

struct SnapshotPool {
    buffers: Vec<Vec<u8>>,
    bytes: usize,
    limit: usize,
    checked: Option<std::time::Instant>,
    memory: Option<(u64, u64, u64)>,
    pressured: bool,
}

impl SnapshotPool {
    const fn new() -> Self {
        Self {
            buffers: Vec::new(),
            bytes: 0,
            limit: 0,
            checked: None,
            memory: None,
            pressured: false,
        }
    }

    fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
        self.trim();
    }

    fn trim(&mut self) {
        if self.bytes <= self.limit && self.buffers.len() <= MAX_POOL_ENTRIES {
            return;
        }
        let mut remaining = self.limit;
        let mut count = 0;
        self.buffers.retain(|v| {
            let capacity = v.capacity();
            if capacity > remaining || count == MAX_POOL_ENTRIES {
                return false;
            }
            remaining -= capacity;
            count += 1;
            true
        });
        self.bytes = self.limit - remaining;
    }

    fn refresh_limit(&mut self) {
        use std::time::{Duration, Instant};
        if self.checked.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
            return;
        }
        self.checked = Some(Instant::now());
        if let Some(memory) = system_memory() {
            self.memory = Some(memory);
            let (_, physical, commit) = memory;
            let pressure = next_memory_pressure(self.pressured, physical, commit);
            if pressure != self.pressured {
                tracing::info!(pressure, physical_free_mib = physical >> 20,
                    available_commit_mib = commit >> 20, "host memory pressure changed");
            }
            self.pressured = pressure;
        }
        // A failed OS query must retain the last successful observation.
        MEMORY_PRESSURED.store(self.pressured, std::sync::atomic::Ordering::Relaxed);
        let limit = idle_budget(self.memory, configured_pool_ceiling(), self.pressured);
        self.set_limit(limit);
    }

    fn take(&mut self, n: usize) -> Vec<u8> {
        // Largest first, so the last fit is the smallest adequate buffer.
        let Some(i) = self.buffers.iter().rposition(|v| v.capacity() >= n) else {
            return Vec::with_capacity(n);
        };
        let mut v = self.buffers.remove(i);
        self.bytes -= v.capacity();
        v.clear();
        v
    }

    fn put(&mut self, mut v: Vec<u8>) {
        let capacity = v.capacity();
        if capacity == 0 || capacity > self.limit {
            return;
        }
        v.clear();
        let i = self.buffers.partition_point(|b| b.capacity() > capacity);
        self.buffers.insert(i, v);
        self.bytes += capacity;
        self.trim();
    }
}

/// Upper bound for idle snapshot allocations, not a limit on live memory.
/// Under pressure keep at most one eighth of available RAM or commit,
/// with a complete idle-cache release below 1 GiB of either.
fn snapshot_limit(memory: Option<(u64, u64, u64)>, ceiling: usize) -> usize {
    const GIB: u64 = 1 << 30;
    let limit = match memory {
        Some((total, available, commit)) => {
            if available.min(commit) < GIB {
                0
            } else {
                (total / 64).min(available / 8).min(commit / 8)
            }
        }
        None => 512 * MIB as u64,
    };
    usize::try_from(limit).unwrap_or(usize::MAX).min(ceiling)
}

fn configured_pool_ceiling() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_SNAPSHOT_POOL_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map_or(MAX_POOL_BYTES, |mb| mb.min(2048) * MIB)
    })
}

pub(crate) fn system_memory() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: a plain structure filled by GlobalMemoryStatusEx.
    let mut memory: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    memory.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: the structure is writable and dwLength describes its size.
    let ok = unsafe { GlobalMemoryStatusEx(&mut memory) } != 0;
    ok.then_some((memory.ullTotalPhys, memory.ullAvailPhys, memory.ullAvailPageFile))
}

/// Enter pressure if either reserve is low; require both reserves to
/// recover before allowing caches to grow again. Thresholds adapted from
/// cool_fox's residency policy (f4b48ad).
fn next_memory_pressure(was: bool, physical: u64, commit: u64) -> bool {
    const GIB: u64 = 1 << 30;
    if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

fn idle_budget(memory: Option<(u64, u64, u64)>, ceiling: usize, pressure: bool) -> usize {
    let normal = snapshot_limit(memory, ceiling);
    if pressure { normal.min(64 * MIB) } else { normal }
}

static POOL: Mutex<SnapshotPool> = Mutex::new(SnapshotPool::new());
static MEMORY_PRESSURED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn memory_pressured() -> bool {
    MEMORY_PRESSURED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Maintain idle storage even when no snapshot was used in a submission.
pub(crate) fn maintain_snapshot_pool() {
    if let Ok(mut pool) = POOL.try_lock() {
        pool.refresh_limit();
    }
}

pub(crate) fn memory_status() -> Option<(u64, u64, u64)> {
    POOL.lock().unwrap_or_else(|e| e.into_inner()).memory
}

/// Emergency reclamation leaves every live snapshot and texture read
/// intact. Keep the idle limit at zero until the next periodic sample.
pub(crate) fn release_idle_buffers() -> usize {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = pool.bytes;
    pool.set_limit(0);
    pool.checked = Some(std::time::Instant::now());
    bytes
}

/// An empty temporary CPU buffer with room for n bytes. Only returned
/// allocations count against the idle budget; live snapshots and active
/// texture reads remain owned by their callers.
pub(crate) fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.refresh_limit();
    pool.take(n)
}

/// Return a completed temporary buffer to the bounded idle cache.
pub(crate) fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.refresh_limit();
    pool.put(v);
}

/// Logical cache accounting, not physical RAM or D3D12 residency.
#[cfg(feature = "diagnostics")]
pub(crate) fn cache_stats() -> (usize, usize, usize, usize, u64) {
    let (count, bytes, limit) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.buffers.len(), pool.bytes, pool.limit)
    };
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    (count, bytes, limit, loose.live.len(), loose.live.values().map(|s| s.len).sum())
}

/// Merge a completed (waited-for) GPU submission. Compare blocks first:
/// untouched GPU blocks take the CPU's bytes, untouched CPU blocks take
/// the GPU's bytes, and genuinely mixed blocks use the original bytewise
/// rule. GPU changes win on collision. Both buffers finish identical.
/// Inspired by cool_fox's merge_piece; this is the synchronous variant,
/// so writing the host mapping is safe only after the existing fence wait.
fn merge_waited_piece(host: &mut [u8], old: &[u8], title: &mut [u8]) -> bool {
    debug_assert_eq!(host.len(), old.len());
    debug_assert_eq!(host.len(), title.len());
    const BLOCK: usize = 128;
    let mut changed = false;
    for ((h, o), t) in host.chunks_mut(BLOCK).zip(old.chunks(BLOCK)).zip(title.chunks_mut(BLOCK)) {
        if h == t {
            continue;
        }
        if h == o {
            h.copy_from_slice(t);
        } else if t == o {
            t.copy_from_slice(h);
            changed = true;
        } else {
            changed |= merge_writeback(t, h, o);
        }
    }
    changed
}

// SAFETY: D3D12 resources are free-threaded. The persistent mapping has
// one writer per Shadow during a synchronous submission. Batch workers
// operate on distinct Shadow allocations; the submitting thread joins
// them before taking snapshots, submitting GPU work or merging writes.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = pooled(n);
        title.resize(n, 0);
        let different = crate::host::read_into(from as *const u8, &mut title) && title != host;
        give_back(title);
        if !different {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, loose: bool) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let mark_base = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks: mark_base,
            loose,
            may_have_gpu_writes: AtomicBool::new(false),
            used_ms: AtomicU64::new(loose_now_ms()),
            used_submission: AtomicU64::new(LOOSE_SUBMISSIONS.load(Relaxed)),
        });
        if let Some(base) = mark_base {
            marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    pub(crate) fn is_loose(&self) -> bool {
        self.loose
    }

    pub(crate) fn note_possible_gpu_write(&self) {
        if self.loose {
            self.may_have_gpu_writes.store(true, Relaxed);
        }
    }

    fn touch_loose(&self) {
        if self.loose {
            self.used_ms.fetch_max(loose_now_ms(), Relaxed);
            self.used_submission.fetch_max(LOOSE_SUBMISSIONS.load(Relaxed), Relaxed);
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.gpu_writes_marked(list, address, len, false);
    }

    /// Declare writes made through a shader that emits view marks.
    pub(crate) fn gpu_writes_marked(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        self.note_possible_gpu_write();
        let len = len.min(self.left(address));
        list.defer_gpu_write_marked(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.writing_marked(address, len, false);
    }

    /// Marked writes arm pages instead of retaining a full range snapshot.
    /// Unmarked callers continue to use the snapshot path.
    pub(crate) fn writing_marked(&self, address: usize, len: usize, marked: bool) {
        self.note_possible_gpu_write();
        self.touch_loose();
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && self.marked() && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = segments.iter().find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &before[from - taken..to - taken];
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            // Unchanged shader ranges need no temporary CPU buffer at all.
            let mut title: Option<Vec<u8>> = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old {
                    let scratch = title.get_or_insert_with(|| {
                        let mut v = pooled(PIECE);
                        v.resize(PIECE, 0);
                        v
                    });
                    let scratch = &mut scratch[..n];
                    if crate::host::read_into((start + at) as *const u8, scratch) {
                        // Preserve CPU writes where the GPU did not write.
                        // The common unchanged blocks can be merged in bulk.
                        if merge_waited_piece(host, old, scratch) {
                            let _writing = crate::watch::writing(start + at, n);
                            crate::host::write_mapped((start + at) as *mut u8, scratch.as_ptr(), n);
                            written += n;
                        }
                        synced.push((start + at) / PAGE);
                    }
                }
                at += n;
            }
            if let Some(v) = title {
                give_back(v);
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        // Do not save a page for this copy again during its own writeback.
        t.unarm(page / PAGE);
        let saved = t.saved(page / PAGE);
        if saved.is_none() && t.changed(page / PAGE) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
        let mut title = pooled(n);
        title.resize(n, 0);
        if !crate::host::read_into(from as *const u8, &mut title) {
            give_back(title);
            return 0;
        }
        let written = match &saved {
            // The title's bytes are the copy's from before: what differs is
            // the shaders'.
            None => (*host != title[..]).then_some(host.as_ptr()),
            Some(before) => {
                let before = &before[from - page..to - page];
                merge_waited_piece(host, before, &mut title).then_some(title.as_ptr())
            }
        };
        if let Some(bytes) = written {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, bytes, n);
        }
        // Equal to title memory now: taken, unless it holds the page in part.
        if from == page && to == page + PAGE {
            t.taken(&[page / PAGE]);
        }
        let count = written.map_or(0, |_| n);
        give_back(title);
        count
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.touch_loose();
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let runs = t.take(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        let mut sync_before = |a: usize, b: usize| {
            let Some(before) = &mut before else { return };
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            sync_before(a, b);
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

static BITMAP: std::sync::OnceLock<Option<Bitmap>> = std::sync::OnceLock::new();

fn bitmap() -> Option<&'static Bitmap> {
    BITMAP
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::view_heap_capacity();
            let bytes = u64::from(bits / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes.
pub(crate) fn view_writes() -> Option<u64> {
    bitmap().map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let Some(b) = BITMAP.get().and_then(Option::as_ref) else {
        return Vec::new();
    };
    let words = (victory_shader::emit::view_heap_capacity() / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // SAFETY: the mapped bitmap's read marks; the GPU is done with them
    // (one submission runs at a time).
    let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            bits[w] &= !bit;
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, (x, s)) in bits.iter().zip(seen.iter_mut()).enumerate() {
        let mut fresh = *x & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call, whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::view_heap_capacity() / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    if let Some(b) = BITMAP.get().and_then(Option::as_ref) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    if BITMAP.get().and_then(Option::as_ref).is_some() {
        READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
    }
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared); the bitmap is cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)]) {
    // No bitmap allocation for unmarked submissions.
    let Some(b) = BITMAP.get().and_then(Option::as_ref) else {
        return;
    };
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
    // SAFETY: the mapped bitmap; the GPU is done with it (one submission
    // runs at a time).
    let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
    // The final bit alone is reserved for out-of-range offsets; other
    // bits in this word may belong to live copies.
    bits[words - 1] &= !(1u32 << 31);
    let mut marked = Vec::new();
    let used = (marks().end as usize).div_ceil(32).min(words);
    for (w, word) in bits[..used].iter_mut().enumerate() {
        let mut x = std::mem::take(word);
        while x != 0 {
            marked.push((w * 32) as u32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked
            .iter()
            .filter_map(|&n| {
                let (&base, copy) = m.owners.range(..=n).next_back()?;
                let copy = copy.upgrade()?;
                let k = n - base;
                (k < pages(copy.start, copy.len)).then(|| {
                    let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
                    (copy, page)
                })
            })
            .collect()
    };
    for (copy, page) in pages {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n)
        {
            // The page was written through a withheld view. Retake its
            // title bytes at the next refresh (upstream ec9e750e).
            crate::watch::retake(page, PAGE);
            continue;
        }
        let written = copy.copy_back_marked(page);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len, false));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// Read-only idle windows may be released by the opt-in submission sweep.
const WINDOW: u64 = 4 << 20;
const LOOSE_SWEEP_MAX: usize = 128;
static LOOSE_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static LOOSE_NEXT_SWEEP: AtomicU64 = AtomicU64::new(0);

struct LooseCache {
    live: BTreeMap<(u64, u64), Arc<Shadow>>,
    // Address metadata only: the old all-address page table must be able
    // to rediscover a previously seen window, even without a fresh find.
    dormant: BTreeSet<(u64, u64)>,
}

static LOOSE: Mutex<LooseCache> = Mutex::new(LooseCache {
    live: BTreeMap::new(),
    dormant: BTreeSet::new(),
});

fn loose_now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

pub(crate) fn loose_reclaim_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_RECLAIM_LOOSE").as_deref() == Ok("1"))
}

fn loose_idle(owners: usize, age_ms: u64, age_submissions: u64, pressure: bool, writable: bool) -> bool {
    owners == 1 && !writable && age_submissions >= 4 && age_ms >= if pressure { 1000 } else { 10000 }
}

impl LooseCache {
    fn take_idle(&mut self, now: u64, submission: u64, pressure: bool) -> Vec<Arc<Shadow>> {
        let keys: Vec<_> = self.live.iter().filter(|(_, s)| {
            loose_idle(Arc::strong_count(s), now.saturating_sub(s.used_ms.load(Relaxed)),
                submission.saturating_sub(s.used_submission.load(Relaxed)), pressure,
                s.may_have_gpu_writes.load(Relaxed))
                && crate::host::resource_references(&s.buffer) == 1
                && s.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
                && s.before.lock().unwrap_or_else(|e| e.into_inner()).0.is_empty()
        }).take(LOOSE_SWEEP_MAX).map(|(key, _)| *key).collect();
        let mut gone = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(copy) = self.live.remove(&key) {
                self.dormant.insert(key);
                gone.push(copy);
            }
        }
        gone
    }

    fn get_or_make(&mut self, key: (u64, u64)) -> Option<(Arc<Shadow>, bool)> {
        if let Some(shadow) = self.live.get(&key) {
            shadow.touch_loose();
            return Some((shadow.clone(), false));
        }
        let shadow = new_loose(key.0, key.1)?;
        self.dormant.remove(&key);
        self.live.insert(key, shadow.clone());
        Some((shadow, true))
    }
}

/// Adapt cool_fox's loose-window sweep to the older synchronous renderer.
/// The map lock serializes lookup against removal. A descriptor slot or
/// unreset list owns the Shadow; page-table and potentially GPU-written
/// windows are pinned even after their owners disappear. The COM count
/// is an additional conservative veto, not the lifetime proof.
pub(crate) fn reclaim_idle_loose() {
    if !loose_reclaim_enabled() {
        return;
    }
    let submission = LOOSE_SUBMISSIONS.fetch_add(1, Relaxed).saturating_add(1);
    let now = loose_now_ms();
    let next = LOOSE_NEXT_SWEEP.load(Relaxed);
    if now < next || LOOSE_NEXT_SWEEP.compare_exchange(next, now.saturating_add(1000), Relaxed, Relaxed).is_err() {
        return;
    }
    let pressure = memory_pressured() || crate::host::video_memory_pressured();
    let gone = {
        let Ok(mut cache) = LOOSE.try_lock() else { return };
        let gone = cache.take_idle(now, submission, pressure);
        if !gone.is_empty() {
            changed();
        }
        gone
    };
    if !gone.is_empty() {
        let bytes: u64 = gone.iter().map(|s| s.len).sum();
        tracing::debug!(windows = gone.len(), bytes, pressure, "released idle read-only loose shadows");
    }
    drop(gone);
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)?.checked_sub(start)?;
    let key = (start, len);
    let mut cache = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (shadow, made) = cache.get_or_make(key)?;
    if made {
        changed();
    }
    Some(shadow)
}

fn new_loose(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    Some(Shadow::new(buffer, data, gpu, start, len, true))
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, address, len, false);
    map.insert((address, len), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// Every host copy of title memory made so far (buffers and windows).
#[cfg(feature = "diagnostics")]
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).live.values().cloned());
    all
}

/// A shader may compute an address without an explicit CPU-side lookup.
/// Restore any dormant window before rebuilding the all-address table;
/// thereafter pin these windows because its GPU writes are not necessarily
/// reconstructible. An allocation failure must not publish a partial
/// address space as if the reclaimed windows did not exist.
pub(crate) fn all_for_page_table() -> Option<Vec<Arc<Shadow>>> {
    let mut all: Vec<_> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    let mut cache = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let dormant: Vec<_> = cache.dormant.iter().copied().collect();
    for (start, len) in dormant {
        let shadow = new_loose(start, len)?;
        cache.live.insert((start, len), shadow);
        cache.dormant.remove(&(start, len));
        changed();
    }
    for shadow in cache.live.values() {
        shadow.note_possible_gpu_write();
        all.push(shadow.clone());
    }
    Some(all)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .live.values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A shader's read mark; the caller holds the submission lock.
    pub(crate) fn mark_read(slot: u32) {
        let b = bitmap().expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs.
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    #[test]
    fn loose_sweep_requires_idle_unowned_read_only_windows() {
        assert!(!loose_idle(1, 9999, 4, false, false));
        assert!(loose_idle(1, 10000, 4, false, false));
        assert!(!loose_idle(1, 999, 4, true, false));
        assert!(loose_idle(1, 1000, 4, true, false));
        assert!(!loose_idle(1, 10000, 3, true, false));
        assert!(!loose_idle(2, 10000, 4, true, false));
        assert!(!loose_idle(1, 10000, 4, true, true));
    }

    #[test]
    fn loose_cache_recreates_read_only_windows_but_keeps_live_and_writable_ones() {
        use crate::watch::tests as pools;
        let Some(_host) = host() else { return };
        // An isolated cache, so this test never sweeps another test's
        // title memory or depends on a process-global environment flag.
        let p = pools::pool(1);
        let key = (p as u64, PAGE as u64);
        let mut cache = LooseCache { live: BTreeMap::new(), dormant: BTreeSet::new() };
        // SAFETY: the test owns the mapped page.
        unsafe { (p as *mut u8).write_bytes(0x31, PAGE) };
        let (copy, made) = cache.get_or_make(key).unwrap();
        assert!(made);
        assert_eq!(copy.update(p, PAGE), PAGE);
        assert_eq!(copy.read_host(p as u64, 1), [0x31313131]);
        let now = copy.used_ms.load(Relaxed).saturating_add(10000);
        let sub = copy.used_submission.load(Relaxed).saturating_add(4);
        assert!(cache.take_idle(now, sub, false).is_empty());
        let com = copy.buffer.clone();
        drop(copy);
        assert!(cache.take_idle(now, sub, false).is_empty());
        drop(com);
        let gone = cache.take_idle(now, sub, false);
        assert_eq!(gone.len(), 1);
        assert!(cache.dormant.contains(&key));
        drop(gone);
        // SAFETY: the same test-owned page is still live.
        unsafe { (p as *mut u8).write_bytes(0x72, PAGE) };
        let (copy, made) = cache.get_or_make(key).unwrap();
        assert!(made);
        assert!(cache.dormant.is_empty());
        assert_eq!(copy.update(p, PAGE), PAGE);
        assert_eq!(copy.read_host(p as u64, 1), [0x72727272]);
        copy.note_possible_gpu_write();
        let now = copy.used_ms.load(Relaxed).saturating_add(10000);
        let sub = copy.used_submission.load(Relaxed).saturating_add(4);
        drop(copy);
        assert!(cache.take_idle(now, sub, true).is_empty());
        drop(cache);
        pools::free(p);
    }

    #[test]
    fn memory_pressure_requires_both_reserves_to_recover() {
        const GIB: u64 = 1 << 30;
        assert!(next_memory_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_memory_pressure(false, 12 * GIB, 4 * GIB));
        assert!(!next_memory_pressure(false, 4 * GIB, 5 * GIB));
        assert!(next_memory_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_memory_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_memory_pressure(true, 5 * GIB, 6 * GIB));
    }

    #[test]
    fn pressure_reduces_idle_budget_without_restricting_live_allocations() {
        const GIB: u64 = 1 << 30;
        let memory = Some((32 * GIB, 3 * GIB, 4 * GIB));
        assert_eq!(idle_budget(memory, MAX_POOL_BYTES, true), 64 * MIB);
        assert_eq!(idle_budget(memory, 16 * MIB, true), 16 * MIB);
        assert_eq!(idle_budget(Some((32 * GIB, GIB / 2, 4 * GIB)), MAX_POOL_BYTES, true), 0);
        assert_eq!(idle_budget(Some((32 * GIB, 8 * GIB, 8 * GIB)), MAX_POOL_BYTES, false), 512 * MIB);
        let mut pool = SnapshotPool::new();
        pool.set_limit(0);
        let mut live = pool.take(4096);
        live.resize(4096, 1);
        assert_eq!(live[4095], 1);
        pool.put(live);
        assert_eq!(pool.bytes, 0);
    }

    #[test]
    fn snapshot_budget_scales_with_ram_and_commit() {
        const GIB: u64 = 1 << 30;
        assert_eq!(snapshot_limit(Some((32 * GIB, 12 * GIB, 20 * GIB)), MAX_POOL_BYTES), 512 * MIB);
        assert_eq!(snapshot_limit(Some((64 * GIB, 2 * GIB, 8 * GIB)), MAX_POOL_BYTES), 256 * MIB);
        assert_eq!(snapshot_limit(Some((64 * GIB, 8 * GIB, GIB / 2)), MAX_POOL_BYTES), 0);
        assert_eq!(snapshot_limit(Some((64 * GIB, GIB / 2, 8 * GIB)), MAX_POOL_BYTES), 0);
        assert_eq!(snapshot_limit(None, MAX_POOL_BYTES), 512 * MIB);
        assert_eq!(snapshot_limit(Some((32 * GIB, 12 * GIB, 20 * GIB)), 64 * MIB), 64 * MIB);
        assert_eq!(snapshot_limit(None, 0), 0);
    }

    #[test]
    fn returned_snapshots_are_bounded_and_reused_best_fit() {
        let mut pool = SnapshotPool::new();
        pool.set_limit(100);
        pool.put(vec![1; 64]);
        pool.put(vec![2; 32]);
        pool.put(vec![3; 256]); // too large to retain
        assert_eq!(pool.bytes, 96);
        let reused = pool.take(20);
        assert_eq!(reused.capacity(), 32);
        assert!(reused.is_empty());
        pool.put(reused);
        pool.set_limit(40);
        assert_eq!(pool.bytes, 32);
        assert_eq!(pool.buffers.len(), 1);
        pool.set_limit(0);
        assert_eq!(pool.bytes, 0);
        assert!(pool.buffers.is_empty());
        // No cache budget must not prevent a live snapshot from being made.
        assert!(pool.take(128).capacity() >= 128);
    }

    #[test]
    fn snapshot_pool_bounds_entry_count() {
        let mut pool = SnapshotPool::new();
        pool.set_limit(8192);
        for _ in 0..MAX_POOL_ENTRIES + 8 {
            pool.put(vec![0; 1]);
        }
        assert_eq!(pool.buffers.len(), MAX_POOL_ENTRIES);
        assert_eq!(pool.bytes, MAX_POOL_ENTRIES);
    }


    #[test]
    fn waited_copyback_merge_matches_the_original_bytewise_rule() {
        for n in [0, 1, 127, 128, 129, 257, PAGE] {
            for mode in 0..5 {
                let old: Vec<u8> = (0..n).map(|i| (i * 31 + 17) as u8).collect();
                let mut gpu = old.clone();
                let mut cpu = old.clone();
                for i in 0..n {
                    if mode == 1 || (mode >= 3 && i % 37 == 0) {
                        gpu[i] ^= 0x51;
                    }
                    if mode == 2 || (mode >= 3 && i % 43 == 0) {
                        cpu[i] ^= if mode == 4 { 0x51 } else { 0xa2 };
                    }
                }
                let expected: Vec<u8> = (0..n).map(|i| if gpu[i] != old[i] { gpu[i] } else { cpu[i] }).collect();
                let changed = expected != cpu;
                assert_eq!(merge_waited_piece(&mut gpu, &old, &mut cpu), changed);
                assert_eq!(gpu, expected);
                assert_eq!(cpu, expected);
            }
        }
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64);
        copy.writing(p + 3 * PAGE, 100);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE);
        copy.writing(p + PAGE, PAGE);
        copy.writing(p + 80, 8);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }


    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap()) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing_marked(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[]);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing_marked(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p);
        copy.copy_back_marked(p + PAGE);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
}
