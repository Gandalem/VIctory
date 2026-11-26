//! Idle texture upload buffers. Only an exclusively owned staging
//! allocation, after all recorded lists have released their Arc<Staging>,
//! can enter this pool. Adapted from the spare-buffer idea in
//! soldsawn/victory-nvidia-8gb; no live resources are evicted.

use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const MIB: u64 = 1 << 20;
const ALIGN: u64 = 64 << 10;
const MAX_ENTRY: u64 = 32 * MIB;
const MAX_ENTRIES: usize = 128;

struct Spare {
    buffer: ID3D12Resource,
    mapped: usize,
    capacity: u64,
}

#[derive(Default)]
struct Idle {
    entries: Vec<Spare>,
    bytes: u64,
}

static IDLE: Mutex<Idle> = Mutex::new(Idle { entries: Vec::new(), bytes: 0 });
static RETRY_UNTIL: AtomicU64 = AtomicU64::new(0);

fn elapsed_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn recent_oom() -> bool {
    elapsed_ms() < RETRY_UNTIL.load(Relaxed)
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

fn trim(idle: &mut Idle, limit: u64) {
    // The pool only owns idle buffers; dropping them cannot invalidate a
    // recorded or in-flight upload. Prefer keeping the most recent.
    while idle.bytes > limit || idle.entries.len() > MAX_ENTRIES {
        let Some(spare) = idle.entries.pop() else { break };
        idle.bytes -= spare.capacity;
    }
}

pub(crate) fn maintain() {
    let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    trim(&mut idle, idle_limit());
}

fn clear() -> u64 {
    let old = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *idle)
    };
    old.bytes
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

pub(crate) fn take(host: &crate::host::Host, size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    if size == 0 || size > isize::MAX as u64 {
        return None;
    }
    let limit = idle_limit();
    let capacity = allocation_size(size, limit);
    let spare = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        trim(&mut idle, limit);
        let i = idle.entries.iter().position(|s| s.capacity == capacity);
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
        return Some((spare.buffer, spare.mapped as *mut u8, capacity));
    }
    #[cfg(feature = "diagnostics")]
    MADE.fetch_add(1, Relaxed);
    host.upload_buffer(capacity).map(|(buffer, data)| (buffer, data, capacity))
}

pub(crate) fn put(buffer: ID3D12Resource, mapped: *mut u8, capacity: u64) {
    let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
    let limit = idle_limit();
    trim(&mut idle, limit);
    if capacity == 0 || capacity > MAX_ENTRY || !capacity.is_multiple_of(ALIGN)
        || capacity > limit || idle.entries.len() >= MAX_ENTRIES
        || capacity > limit.saturating_sub(idle.bytes)
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
    (idle.bytes, idle.entries.len(), MADE.swap(0, Relaxed), REUSED.swap(0, Relaxed), recent_oom())
}

#[cfg(test)]
mod tests {
    use super::*;

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
