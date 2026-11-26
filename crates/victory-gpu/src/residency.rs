//! Conservative D3D12 residency for committed sampled textures.
//! Never manually evict an allocation once a host pointer or descriptor
//! may have escaped. The eager-resource renderer cannot yet rebuild every
//! descriptor as the lazy Hosted/tiles architecture in cool_fox does.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::{Acquire, Release, Relaxed}};
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use crate::resource::HostTexture;

const OFF: u8 = 0;
const UNPUBLISHED: u8 = 1;
const EVICTED: u8 = 2;
const EXPOSED: u8 = 3;
const MIB: u64 = 1 << 20;
static COPIES: Mutex<BTreeMap<usize, Weak<HostTexture>>> = Mutex::new(BTreeMap::new());

#[derive(Default)]
pub(crate) struct Residency {
    phase: AtomicU8,
    gate: Mutex<()>,
    last: AtomicU64,
    low: AtomicBool,
}

fn now() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TEXTURE_RESIDENCY").as_deref() == Ok("1"))
}

fn may_evict(phase: u8, age: u64, pressure: bool) -> bool {
    phase == UNPUBLISHED && pressure && age >= 2000
}

fn cold_priority(age: u64, pressure: bool) -> bool {
    pressure && age >= 5000
}

/// Register before the texture is published to its title Resource.
/// Never register twins, render targets, depth or UAV allocations.
pub(crate) fn register(copy: &Arc<HostTexture>) {
    if !enabled() || !copy.sampled || copy.bytes < MIB {
        return;
    }
    copy.residency.last.store(now(), Relaxed);
    copy.residency.phase.store(UNPUBLISHED, Release);
    COPIES.lock().unwrap_or_else(|e| e.into_inner())
        .insert(Arc::as_ptr(copy) as usize, Arc::downgrade(copy));
}

impl Residency {
    pub(crate) fn touch(&self) {
        if self.phase.load(Relaxed) != OFF {
            self.last.store(now(), Relaxed);
        }
    }

    /// Called before handing out a host pointer or owning Arc. A texture
    /// whose identity ever escaped may be held by a recorded command or
    /// descriptor that our older renderer cannot enumerate exhaustively.
    /// Therefore it stays explicitly resident for its remaining lifetime.
    pub(crate) fn expose(&self, texture: &ID3D12Resource) {
        let phase = self.phase.load(Acquire);
        if phase == OFF {
            return;
        }
        self.touch();
        if phase == EXPOSED {
            return;
        }
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        if self.phase.load(Acquire) == EVICTED {
            let Some(host) = crate::host::host() else {
                crate::host::fail(format_args!("host missing while restoring an evicted texture"));
            };
            let result = texture.cast::<ID3D12Pageable>().and_then(|pageable| {
                crate::upload_pool::retry_oom(|| {
                    // SAFETY: live committed resource; the gate excludes
                    // eviction, and no descriptor has been published.
                    unsafe { host.device.MakeResident(&[Some(pageable.clone())]) }
                })
            });
            if let Err(e) = result {
                crate::host::fail(format_args!("could not restore sampled texture residency: {e}"));
            }
        }
        self.phase.store(EXPOSED, Release);
        self.touch();
    }
}

/// The submitter calls this with the synchronous submission lock held.
/// At most 128 transitions per sweep; no forced eviction of published
/// resources. Published cold textures receive only a paging priority hint.
pub(crate) fn maintain(host: &crate::host::Host) {
    if !enabled() {
        return;
    }
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let now = now();
    if now < NEXT.load(Relaxed) {
        return;
    }
    NEXT.store(now.saturating_add(1000), Relaxed);
    static BUDGET_PRESSURE: AtomicBool = AtomicBool::new(false);
    let budget_pressure = crate::resource::texture_budget_pressure(host.video_memory(), BUDGET_PRESSURE.load(Relaxed));
    BUDGET_PRESSURE.store(budget_pressure, Relaxed);
    let pressure = crate::host::video_memory_pressured() || budget_pressure;
    let owners: Vec<_> = {
        let mut registry = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        registry.retain(|_, copy| copy.strong_count() > 0);
        registry.values().cloned().collect()
    };
    let device1 = host.device.cast::<ID3D12Device1>().ok();
    let (mut changes, mut evicted, mut bytes, mut lowered, mut raised) = (0, 0, 0u64, 0, 0);
    for owner in owners {
        if changes == 128 { break }
        let Some(copy) = owner.upgrade() else { continue };
        let r = &copy.residency;
        let phase = r.phase.load(Acquire);
        let age = now.saturating_sub(r.last.load(Relaxed));
        let cold = cold_priority(age, pressure);
        if !may_evict(phase, age, pressure) && !(phase == EXPOSED && cold != r.low.load(Relaxed)) {
            continue;
        }
        let Ok(_guard) = r.gate.try_lock() else { continue };
        let phase = r.phase.load(Acquire);
        let age = now.saturating_sub(r.last.load(Relaxed));
        if may_evict(phase, age, pressure) {
            let Ok(pageable) = copy.texture.cast::<ID3D12Pageable>() else { continue };
            // SAFETY: this committed allocation has never been exposed.
            // Every host pointer/Arc publication passes through expose,
            // which shares this gate. No recorded GPU command can name it.
            if unsafe { host.device.Evict(&[Some(pageable)]) }.is_ok() {
                r.phase.store(EVICTED, Release);
                evicted += 1;
                bytes = bytes.saturating_add(copy.bytes);
            } else {
                // Unsupported driver: leave it resident; do not repeat
                // the failing eviction on every sweep.
                r.phase.store(EXPOSED, Release);
            }
            changes += 1;
        } else if phase == EXPOSED && let Some(device) = &device1 {
            let cold = cold_priority(age, pressure);
            if cold == r.low.load(Relaxed) {
                continue;
            }
            let Ok(pageable) = copy.texture.cast::<ID3D12Pageable>() else { continue };
            let objects = [Some(pageable)];
            let priorities = [if cold { D3D12_RESIDENCY_PRIORITY_LOW } else { D3D12_RESIDENCY_PRIORITY_NORMAL }];
            // SAFETY: arrays contain exactly one live pageable and its
            // priority. This does not evict or change its GPU address.
            if unsafe { device.SetResidencyPriority(1, objects.as_ptr(), priorities.as_ptr()) }.is_ok() {
                r.low.store(cold, Relaxed);
                if cold { lowered += 1 } else { raised += 1 }
            }
            changes += 1;
        }
    }
    if evicted != 0 || lowered != 0 || raised != 0 {
        tracing::info!(evicted, logical_evicted_mib = bytes >> 20, lowered, raised, pressure,
            "sampled texture residency sweep");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_unpublished_textures_may_be_explicitly_evicted() {
        assert!(!may_evict(OFF, u64::MAX, true));
        assert!(!may_evict(EXPOSED, u64::MAX, true));
        assert!(!may_evict(EVICTED, u64::MAX, true));
        assert!(!may_evict(UNPUBLISHED, 1999, true));
        assert!(!may_evict(UNPUBLISHED, 2000, false));
        assert!(may_evict(UNPUBLISHED, 2000, true));
        assert!(!cold_priority(4999, true));
        assert!(cold_priority(5000, true));
        assert!(!cold_priority(5000, false));
    }
}
