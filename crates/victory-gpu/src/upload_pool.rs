//! Idle texture upload buffers. Only an exclusively owned staging
//! allocation, after all recorded lists have released their Arc<Staging>,
//! can enter this pool. Adapted from the spare-buffer idea in
//! soldsawn/victory-nvidia-8gb; no live resources are evicted.

use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const MIB: u64 = 1 << 20;
const ALIGN: u64 = 64 << 10;
const MAX_ENTRY: u64 = 32 * MIB;
const MAX_ENTRIES: usize = 128;
const BLOCK: u64 = 64 * MIB;
const MAX_BLOCKS: usize = 16;

/// Cut staging slices from a shared mapped upload resource, adapted
/// from cool_fox's pool.rs. The recorded-list Arc<Staging> owns the
/// lease until reset; no time-based guess about GPU completion is used.
struct UploadBlock {
    buffer: ID3D12Resource,
    mapped: usize,
    taken: Mutex<Vec<bool>>,
}

pub(crate) struct BlockLease {
    block: Arc<UploadBlock>,
    first: usize,
    count: usize,
}

fn gap(taken: &[bool], count: usize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let mut free = 0;
    for (i, &used) in taken.iter().enumerate() {
        free = if used { 0 } else { free + 1 };
        if free == count {
            return Some(i + 1 - count);
        }
    }
    None
}

impl UploadBlock {
    fn lease(self: &Arc<Self>, capacity: u64) -> Option<BlockLease> {
        let count = (capacity / ALIGN) as usize;
        let mut taken = self.taken.lock().unwrap_or_else(|e| e.into_inner());
        let first = gap(&taken, count)?;
        taken[first..first + count].fill(true);
        Some(BlockLease { block: self.clone(), first, count })
    }
}

impl Drop for BlockLease {
    fn drop(&mut self) {
        let mut taken = self.block.taken.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert!(taken[self.first..self.first + self.count].iter().all(|&used| used));
        taken[self.first..self.first + self.count].fill(false);
    }
}

pub(crate) struct Allocation {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) data: *mut u8,
    pub(crate) capacity: u64,
    pub(crate) offset: u64,
    pub(crate) lease: Option<BlockLease>,
}

impl BlockLease {
    fn allocation(self, size: u64) -> Allocation {
        let offset = self.first as u64 * ALIGN;
        let capacity = self.count as u64 * ALIGN;
        let buffer = self.block.buffer.clone();
        // SAFETY: this lease owns a disjoint, mapped slice and no earlier
        // recorded list can still own it. The slice fits inside BLOCK.
        let data = unsafe { (self.block.mapped as *mut u8).add(offset as usize) };
        unsafe { data.write_bytes(0, size as usize) };
        Allocation { buffer, data, capacity, offset, lease: Some(self) }
    }
}

fn blocks_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_STAGING_BLOCKS").as_deref() == Ok("1"))
        && configured_limit() >= BLOCK
}

struct Spare {
    buffer: ID3D12Resource,
    mapped: usize,
    capacity: u64,
}

#[derive(Default)]
struct Idle {
    entries: Vec<Spare>,
    bytes: u64,
    blocks: Vec<Arc<UploadBlock>>,
    block_pending: usize,
}

static IDLE: Mutex<Idle> = Mutex::new(Idle {
    entries: Vec::new(), bytes: 0, blocks: Vec::new(), block_pending: 0,
});
static RETRY_UNTIL: AtomicU64 = AtomicU64::new(0);

fn elapsed_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis().min(u64::MAX as u128) as u64
}

pub(crate) fn recent_oom() -> bool {
    elapsed_ms() < RETRY_UNTIL.load(Relaxed)
}

/// Bindless heaps may contain speculative references that are not read
/// by this submission. Limit only *new* heap staging allocations under
/// memory pressure; existing staging refreshes and explicit uploads stay
/// on their original path. Deferred descriptors are retried next submit.
pub(crate) struct Admission {
    commit_left: u64,
    batch_left: u64,
    spent: u64,
    limited_ram: bool,
    sampled_at: Instant,
}

fn batch_limit(commit: u64, pressure: bool, limited_ram: bool, staging: u64) -> u64 {
    if commit < 1536 * MIB {
        16 * MIB
    } else if pressure || (limited_ram && staging >= 1024 * MIB) {
        64 * MIB
    } else {
        u64::MAX
    }
}

impl Admission {
    pub(crate) fn new() -> Self {
        let sample = crate::shadow::system_memory().or_else(crate::shadow::memory_status);
        let (total, _, commit) = sample.unwrap_or((u64::MAX, u64::MAX, u64::MAX));
        let pressure = crate::shadow::memory_pressured() || crate::host::video_memory_pressured();
        Self::from_status(commit, pressure, total <= 32 * 1024 * MIB, crate::texture::live_staging_bytes())
    }

    fn from_status(commit: u64, pressure: bool, limited_ram: bool, staging: u64) -> Self {
        Self {
            commit_left: commit,
            batch_left: batch_limit(commit, pressure, limited_ram, staging),
            spent: 0,
            limited_ram,
            sampled_at: Instant::now(),
        }
    }

    fn tighten(&mut self, pressure: bool, staging: u64) {
        let limit = batch_limit(self.commit_left, pressure, self.limited_ram, staging);
        self.batch_left = self.batch_left.min(limit.saturating_sub(self.spent));
    }

    fn fits(&self, bytes: u64, growth: u64) -> bool {
        // Reuse already committed storage without charging system commit.
        // New allocations require a larger reserve, and must still leave
        // 512 MiB for required title and GPU work after the allocation.
        let reserve = if growth != 0 {
            self.commit_left >= 1024 * MIB
                && self.commit_left.saturating_sub(512 * MIB) >= growth
        } else {
            self.commit_left >= 512 * MIB
        };
        reserve && (self.batch_left >= bytes || self.spent == 0)
    }

    fn admit(&mut self, bytes: u64, growth: u64) -> bool {
        if recent_oom() {
            return false;
        }
        // A single heap sweep can outlast the ordinary one-second memory
        // census. Resample commit at most once per 32 ms. On failure keep
        // the previously sampled headroom minus allocations since then.
        if self.sampled_at.elapsed() >= std::time::Duration::from_millis(32) {
            self.sampled_at = Instant::now();
            if let Some((total, _, commit)) = crate::shadow::system_memory() {
                self.commit_left = commit;
                self.limited_ram = total <= 32 * 1024 * MIB;
            }
        }
        let pressure = crate::shadow::memory_pressured() || crate::host::video_memory_pressured();
        self.tighten(pressure, crate::texture::live_staging_bytes());
        self.fits(bytes, growth)
    }

    fn consume(&mut self, bytes: u64, growth: u64) {
        self.commit_left = self.commit_left.saturating_sub(growth);
        self.spent = self.spent.saturating_add(bytes);
        self.batch_left = self.batch_left.saturating_sub(bytes);
    }
}

fn configured_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var("VICTORY_STAGING_POOL").as_deref() == Ok("0") {
            return 0;
        }
        std::env::var("VICTORY_STAGING_POOL_MB").ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(128).min(512) * MIB
    })
}

fn idle_limit() -> u64 {
    if recent_oom() || crate::shadow::memory_pressured() || crate::host::video_memory_pressured() {
        0
    } else {
        configured_limit()
    }
}

fn allocation_size(size: u64, limit: u64) -> u64 {
    let rounded = size.checked_add(ALIGN - 1).map(|n| n & !(ALIGN - 1));
    match rounded {
        Some(n) if n != 0 && n <= MAX_ENTRY && n <= limit => n,
        _ => size,
    }
}

fn idle_blocks(idle: &Idle) -> usize {
    // No new lease can be made without the pool lock. A strong count of
    // one means that the registry is the only owner of this block.
    idle.blocks.iter().filter(|block| Arc::strong_count(block) == 1).count()
}

fn idle_bytes(idle: &Idle) -> u64 {
    idle.bytes + idle_blocks(idle) as u64 * BLOCK
}

fn trim(idle: &mut Idle, limit: u64) {
    // Only idle dedicated buffers and completely unleased blocks can
    // be dropped. The active leases retain both resource and mapping.
    while idle_bytes(idle) > limit || idle.entries.len() > MAX_ENTRIES {
        if let Some(spare) = idle.entries.pop() {
            idle.bytes -= spare.capacity;
        } else if let Some(i) = idle.blocks.iter().position(|b| Arc::strong_count(b) == 1) {
            idle.blocks.swap_remove(i);
        } else {
            break;
        }
    }
}

pub(crate) fn maintain() {
    let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    trim(&mut idle, idle_limit());
}

fn clear() -> u64 {
    let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    let before = idle_bytes(&idle);
    trim(&mut idle, 0);
    before
}

/// An existing slice has no additional system commit cost. A new block
/// charges its complete capacity, not just the first slice. Under
/// pressure never grow blocks; the old dedicated path remains available.
fn take_block(host: &crate::host::Host, size: u64, admission: &mut Option<&mut Admission>) -> Option<Allocation> {
    if !blocks_enabled() || size > MAX_ENTRY {
        return None;
    }
    let capacity = size.next_multiple_of(ALIGN);
    let limit = idle_limit();
    let found = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        trim(&mut idle, limit);
        if let Some(lease) = idle.blocks.iter().find_map(|b| b.lease(capacity)) {
            Some(lease)
        } else {
            if limit < BLOCK || idle.blocks.len() + idle.block_pending >= MAX_BLOCKS
                || admission.as_deref_mut().is_some_and(|a| !a.admit(capacity, BLOCK))
            {
                return None;
            }
            // Reserve the registry slot, but do not hold its lock during
            // D3D12 allocation: the OOM retry may trim this same pool.
            idle.block_pending += 1;
            None
        }
    };
    if let Some(lease) = found {
        if let Some(a) = admission.as_deref_mut() {
            if !a.admit(capacity, 0) {
                return None;
            }
            a.consume(capacity, 0);
        }
        #[cfg(feature = "diagnostics")]
        REUSED.fetch_add(1, Relaxed);
        return Some(lease.allocation(size));
    }

    let result = host.upload_buffer(BLOCK);
    let lease = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        idle.block_pending -= 1;
        let (buffer, mapped) = result?;
        let block = Arc::new(UploadBlock {
            buffer, mapped: mapped as usize, taken: Mutex::new(vec![false; (BLOCK / ALIGN) as usize]),
        });
        let lease = block.lease(capacity)?;
        idle.blocks.push(block);
        lease
    };
    if let Some(a) = admission.as_deref_mut() {
        a.consume(capacity, BLOCK);
    }
    #[cfg(feature = "diagnostics")]
    MADE.fetch_add(1, Relaxed);
    Some(lease.allocation(size))
}

/// Retry a failed required allocation once after releasing only idle
/// caches. The 500 ms hold-down disables idle GPU retention, not required
/// allocations: the older renderer has no deferred admission queue.
pub(crate) fn retry_oom<T>(mut allocate: impl FnMut() -> windows::core::Result<T>) -> windows::core::Result<T> {
    match allocate() {
        Err(e) if e.code() == windows::Win32::Foundation::E_OUTOFMEMORY => {
            RETRY_UNTIL.fetch_max(elapsed_ms().saturating_add(500), Relaxed);
            let gpu = clear();
            let cpu = crate::shadow::release_idle_buffers() as u64;
            if gpu == 0 && cpu == 0 {
                return Err(e);
            }
            tracing::warn!(gpu_idle_bytes = gpu, cpu_idle_bytes = cpu, "reclaimed idle memory; retrying required allocation");
            allocate()
        }
        result => result,
    }
}

pub(crate) fn take(
    host: &crate::host::Host,
    size: u64,
    mut admission: Option<&mut Admission>,
) -> Option<Allocation> {
    if size == 0 || size > isize::MAX as u64 {
        return None;
    }
    if let Some(allocation) = take_block(host, size, &mut admission) {
        return Some(allocation);
    }
    let limit = idle_limit();
    let capacity = allocation_size(size, limit);
    let spare = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        trim(&mut idle, limit);
        let i = idle.entries.iter().position(|s| s.capacity == capacity);
        // Check admission against the actual allocation class and reuse
        // decision while the idle pool is locked. Explicit uploads pass
        // no admission policy and retain their original required path.
        if let Some(limit) = admission.as_deref_mut()
            && !limit.admit(capacity, if i.is_none() { capacity } else { 0 })
        {
            return None;
        }
        i.map(|i| {
            let spare = idle.entries.swap_remove(i);
            idle.bytes -= spare.capacity;
            spare
        })
    };
    if let Some(spare) = spare {
        // SAFETY: the pool owns the entire mapping exclusively; no list
        // holds this staging allocation, and size <= capacity.
        unsafe { (spare.mapped as *mut u8).write_bytes(0, size as usize) };
        #[cfg(feature = "diagnostics")]
        REUSED.fetch_add(1, Relaxed);
        if let Some(limit) = admission {
            limit.consume(capacity, 0);
        }
        return Some(Allocation { buffer: spare.buffer, data: spare.mapped as *mut u8,
            capacity, offset: 0, lease: None });
    }
    #[cfg(feature = "diagnostics")]
    MADE.fetch_add(1, Relaxed);
    let (buffer, data) = host.upload_buffer(capacity)?;
    if let Some(limit) = admission {
        limit.consume(capacity, capacity);
    }
    Some(Allocation { buffer, data, capacity, offset: 0, lease: None })
}

pub(crate) fn put(buffer: ID3D12Resource, mapped: *mut u8, capacity: u64) {
    let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    let limit = idle_limit();
    trim(&mut idle, limit);
    if capacity == 0 || capacity > MAX_ENTRY || !capacity.is_multiple_of(ALIGN)
        || capacity > limit || idle.entries.len() >= MAX_ENTRIES
        || capacity > limit.saturating_sub(idle_bytes(&idle))
    {
        return;
    }
    idle.entries.push(Spare { buffer, mapped: mapped as usize, capacity });
    idle.bytes += capacity;
}

#[cfg(feature = "diagnostics")]
static MADE: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static REUSED: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn stats() -> (u64, usize, u64, u64, bool) {
    let idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    (idle_bytes(&idle), idle.entries.len() + idle_blocks(&idle), MADE.swap(0, Relaxed), REUSED.swap(0, Relaxed), recent_oom())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_preserves_commit_and_allows_oversized_first_texture() {
        let mut limit = Admission::from_status(1024 * MIB, true, false, 0);
        assert!(!limit.fits(513 * MIB, 513 * MIB));
        assert!(limit.fits(32 * MIB, 32 * MIB));
        limit.consume(32 * MIB, 32 * MIB);
        assert!(!limit.fits(MIB, 0));
        assert!(!Admission::from_status(512 * MIB, false, false, 0).fits(1, 1));
        let mut limit = Admission::from_status(4 * 1024 * MIB, true, false, 0);
        assert!(limit.fits(64 * MIB, 64 * MIB));
        limit.consume(64 * MIB, 64 * MIB);
        assert!(!limit.fits(1, 0));
    }

    #[test]
    fn admission_distinguishes_reuse_and_growth() {
        let mut limit = Admission::from_status(768 * MIB, false, false, 0);
        assert!(limit.fits(MIB, 0));
        assert!(!limit.fits(MIB, MIB));
        limit.consume(MIB, 0);
        assert_eq!(limit.commit_left, 768 * MIB);
        assert_eq!(limit.batch_left, 15 * MIB);
        assert!(!Admission::from_status(512 * MIB - 1, false, false, 0).fits(1, 0));
    }

    #[test]
    fn admission_tightens_when_a_long_sweep_crosses_staging_or_commit_limits() {
        let mut limit = Admission::from_status(8 * 1024 * MIB, false, true, 1024 * MIB - 1);
        assert!(limit.fits(48 * MIB, 48 * MIB));
        limit.consume(48 * MIB, 48 * MIB);
        limit.tighten(false, 1024 * MIB);
        assert_eq!(limit.batch_left, 16 * MIB);
        limit.commit_left = 1400 * MIB;
        limit.tighten(false, 1024 * MIB);
        assert!(!limit.fits(1, 0));
        limit.commit_left = 8 * 1024 * MIB;
        limit.tighten(false, 0);
        assert_eq!(limit.batch_left, 0);
        assert_eq!(batch_limit(8 * 1024 * MIB, false, false, 2 * 1024 * MIB), u64::MAX);
        assert_eq!(batch_limit(8 * 1024 * MIB, false, true, 1024 * MIB), 64 * MIB);
    }

    #[test]
    fn block_lease_ranges_find_contiguous_aligned_slices() {
        let mut taken = vec![false; 8];
        assert_eq!(gap(&taken, 3), Some(0));
        taken[..3].fill(true);
        assert_eq!(gap(&taken, 3), Some(3));
        taken[3..6].fill(true);
        assert_eq!(gap(&taken, 3), None);
        taken[..3].fill(false);
        assert_eq!(gap(&taken, 3), Some(0));
        assert_eq!(gap(&taken, 0), None);
    }

    #[test]
    fn first_slice_accounts_for_entire_new_block() {
        let mut limit = Admission::from_status(1024 * MIB, false, false, 0);
        assert!(limit.fits(MIB, BLOCK));
        limit.consume(MIB, BLOCK);
        assert_eq!(limit.commit_left, 960 * MIB);
        assert_eq!(limit.batch_left, 15 * MIB);
        assert!(limit.fits(MIB, 0));
        assert!(!limit.fits(MIB, BLOCK));
        limit.consume(MIB, 0);
        assert_eq!(limit.commit_left, 960 * MIB);
    }

    #[test]
    fn staging_classes_do_not_inflate_large_or_unpooled_allocations() {
        assert_eq!(allocation_size(1, 128 * MIB), ALIGN);
        assert_eq!(allocation_size(ALIGN + 1, 128 * MIB), 2 * ALIGN);
        assert_eq!(allocation_size(33 * MIB + 1, 128 * MIB), 33 * MIB + 1);
        assert_eq!(allocation_size(ALIGN + 1, 0), ALIGN + 1);
        assert_eq!(allocation_size(u64::MAX, 128 * MIB), u64::MAX);
    }

    #[test]
    fn allocation_retries_do_not_mask_non_oom_errors() {
        let mut calls = 0;
        let result: windows::core::Result<()> = retry_oom(|| {
            calls += 1;
            Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}
