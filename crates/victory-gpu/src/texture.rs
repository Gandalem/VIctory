//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! Tracked writes refresh the affected retained subresources. Heap
//! uploads copy those subresources; pre-recorded explicit uploads keep
//! their original whole-chain copies.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Logical footprint bytes and allocated upload-buffer capacity.
    size: u64,
    capacity: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Host mips per array slice.
    mips: u32,
    /// Title mip corresponding to host level zero.
    first_mip: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    last: Mutex<Last>,
    used_ms: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked). Temporary tiled reads share the bounded CPU pool
/// instead of retaining a full copy for every resident texture.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    // A failed detile must retry even after consuming dirty pages.
    ready: bool,
    // Accumulate successful staging changes until a heap upload records
    // them. Explicit lists may also refresh staging, so a clean refresh
    // must not discard changes not yet acknowledged by a heap upload.
    pending: Vec<bool>,
}

impl Last {
    fn mark(&mut self, count: usize, indices: impl IntoIterator<Item = usize>) {
        self.pending.resize(count, false);
        for i in indices {
            self.pending[i] = true;
        }
    }

    fn has_pending(&self) -> bool {
        self.ready && self.pending.iter().any(|&dirty| dirty)
    }
}

static STAGING_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn live_staging_bytes() -> u64 {
    STAGING_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

impl Drop for Staging {
    fn drop(&mut self) {
        STAGING_BYTES.fetch_sub(self.size, std::sync::atomic::Ordering::Relaxed);
        // All recorded uploads hold the staging Arc, not only its buffer.
        // The last owner releases an idle allocation, never a live slice.
        crate::upload_pool::put(self.buffer.clone(), self.data, self.capacity);
    }
}

/// Weak owners of staging allocations. Register only on staging creation,
/// so the registry never prolongs a texture's lifetime.
static STAGED: Mutex<std::collections::BTreeMap<usize, std::sync::Weak<crate::resource::HostTexture>>> =
    Mutex::new(std::collections::BTreeMap::new());

fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn idle_staging(owners: usize, age: u64, pressure: bool) -> bool {
    owners == 1 && age >= if pressure { 1000 } else { 10000 }
}

/// Reclaim uploads held only by their texture, not by any recorded list.
/// The next use recreates and fully refreshes the staging contents. Never
/// remove the host texture, its descriptors or GPU-authored contents.
pub(crate) fn reclaim_idle_staging() {
    use std::sync::atomic::Ordering::Relaxed;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let next = NEXT.load(Relaxed);
    if now < next || NEXT.compare_exchange(next, now.saturating_add(1000), Relaxed, Relaxed).is_err() {
        return;
    }
    let pressure = crate::shadow::memory_pressured() || crate::host::video_memory_pressured();
    let owners: Vec<_> = {
        let mut registry = STAGED.lock().unwrap_or_else(|e| e.into_inner());
        registry.retain(|_, owner| owner.strong_count() > 0);
        registry.values().cloned().collect()
    };
    let mut reclaimed = 0u64;
    for owner in owners {
        let Some(copy) = owner.upgrade() else { continue };
        let Ok(mut staging) = copy.staging.try_lock() else { continue };
        let idle = staging.as_ref().is_some_and(|s| {
            idle_staging(Arc::strong_count(s), now.saturating_sub(s.used_ms.load(Relaxed)), pressure)
        });
        if idle {
            if let Some(s) = staging.take() {
                reclaimed = reclaimed.saturating_add(s.capacity);
            }
        }
    }
    if reclaimed != 0 {
        tracing::debug!(bytes = reclaimed, pressure, "released idle texture staging owners");
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Map aliases by their title mip/slice when either host texture is
    // shortened. Otherwise preserve the original prefix-copy behavior.
    let pairs: Vec<(usize, usize)> = if crate::resource::dropped(from) > 0 || crate::resource::dropped(to) > 0 {
        let count = crate::resource::title_subresources(from).min(crate::resource::title_subresources(to));
        (0..count).filter_map(|i| {
            Some((crate::resource::host_subresource(from, i)? as usize,
                  crate::resource::host_subresource(to, i)? as usize))
        }).take_while(|&(i, j)| {
            i < fa.len() && j < fb.len() && fa[i].Footprint.RowPitch == fb[j].Footprint.RowPitch
                && ra[i] == rb[j] && fa[i].Footprint.Depth == fb[j].Footprint.Depth
        }).collect()
    } else {
        (0..fa.len().min(fb.len())).take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        }).map(|i| (i, i)).collect()
    };
    let n = pairs.len() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for &(i, _) in &pairs {
            l.CopyTextureRegion(&place(&buffer, fa[i]), 0, 0, 0, &sub(a, i as u32), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for &(i, j) in &pairs {
            let mut footprint = fb[j];
            footprint.Offset = fa[i].Offset;
            l.CopyTextureRegion(&sub(b, j as u32), 0, 0, 0, &place(&buffer, footprint), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    record(list, resource, texture, &staging, None);
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn prepare_heap_upload(resource: This, admission: &mut crate::upload_pool::Admission) -> Option<Arc<Staging>> {
    if is_host_written(resource) {
        return None;
    }
    staging_with_admission(resource, crate::resource::host_resource(resource)?, Some(admission))
}

/// Record only after all CPU preparation jobs have joined. The heap lock
/// still protects the title resource; command recording stays serial.
pub(crate) fn record_heap_upload(list: &HostList, resource: This, staging: &Arc<Staging>) -> bool {
    let Some(texture) = crate::resource::host_resource(resource) else {
        return false;
    };
    let mut last = staging.last.lock().unwrap_or_else(|e| e.into_inner());
    if !last.has_pending() {
        return false;
    }
    record(list, resource, texture, staging, Some(&last.pending));
    // Recording and execution are serialized by the submission lock.
    // If submission fails, the host stops instead of using stale data.
    last.pending.fill(false);
    list.keep_staging(staging);
    true
}

/// Record selected subresources, or every one for pre-recorded lists.
/// The latter must keep all copies because their future dirty ranges
/// cannot be known at recording time.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging, selected: Option<&[bool]>) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        if selected.is_some_and(|mask| !mask[i]) {
            continue;
        }
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// Bound the simultaneous tiled reads rather than multiplying the size of
/// every live texture by the number of workers. A texture larger than the
/// limit is processed on its own, never silently omitted.
const REFRESH_CHUNK_BYTES: u64 = 128 << 20;
const REFRESH_CHUNK_COUNT: usize = 32;

fn chunk_len(sizes: impl Iterator<Item = u64>, limit: u64, count: usize) -> usize {
    let (mut bytes, mut n) = (0u64, 0usize);
    for size in sizes.take(count) {
        if n != 0 && bytes.saturating_add(size) > limit {
            break;
        }
        bytes = bytes.saturating_add(size);
        n += 1;
    }
    n
}

/// Refresh independent staging buffers in input order (the returned flags
/// correspond to the input). Only tracked source memory enters workers:
/// it is read by the existing checked kernel copy. Untracked memory keeps
/// its existing serial hash path. Each worker owns a distinct staging
/// mapping, and no job survives this call or a submission boundary.
pub(crate) fn refresh_many(stagings: &[Arc<Staging>]) -> Vec<bool> {
    if let [staging] = stagings {
        return vec![staging.refresh()];
    }
    use std::sync::atomic::Ordering::Relaxed;
    let started = std::time::Instant::now();
    let mut flags = Vec::with_capacity(stagings.len());
    let mut read = 0usize;
    let mut at = 0;
    while at < stagings.len() {
        let threads = if crate::shadow::memory_pressured() {
            1
        } else {
            crate::texture_workers::configured_threads()
        };
        if threads < 2 || stagings[at].tracked.is_none() {
            let (bytes, changed) = stagings[at].refresh_now();
            read = read.saturating_add(bytes);
            flags.push(changed);
            at += 1;
            continue;
        }
        let n = chunk_len(stagings[at..].iter().take_while(|s| s.tracked.is_some())
            .map(|s| s.layout.size as u64), REFRESH_CHUNK_BYTES, REFRESH_CHUNK_COUNT);
        debug_assert!(n > 0);
        let results = crate::texture_workers::ordered_map(stagings[at..at + n].to_vec(), threads,
            |staging| staging.refresh_now());
        for (bytes, changed) in results {
            read = read.saturating_add(bytes);
            flags.push(changed);
        }
        at += n;
    }
    // This is elapsed wall time, not the sum of overlapping worker times.
    let preparing = &crate::host::PREPARING;
    preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    preparing[3].fetch_add(read as u64, Relaxed);
    flags
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    /// Returns the bytes of title memory read, and whether it detiled.
    fn refresh_now(&self) -> (usize, bool) {
        self.used_ms.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            let dirty = t.take(start, start + size);
            if dirty.is_empty() && last.ready {
                return (0, false);
            }
            // A new staging buffer has no valid clean subresources.
            // Otherwise retain unchanged mips and array slices.
            let mut spans = std::collections::BTreeMap::<(u64, u64), Vec<usize>>::new();
            let mut supported = true;
            for i in 0..self.footprints.len() {
                let (mip, slice) = (i as u32 % self.mips + self.first_mip, i as u32 / self.mips);
                let Some((a, b)) = self.layout.source_span(mip, slice) else {
                    supported = false;
                    break;
                };
                if !last.ready || dirty.iter().any(|&(from, to)| from < start + b as usize && to > start + a as usize) {
                    spans.entry((a, b)).or_default().push(i);
                }
            }
            let mut read_bytes = 0usize;
            let mut ok = true;
            if !supported {
                // Preserve the previous whole-layout fallback.
                let mut read = crate::shadow::pooled(size);
                read.resize(size, 0);
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                ok = self.detile(&read);
                crate::shadow::give_back(read);
                read_bytes = size;
                if ok {
                    last.mark(self.footprints.len(), 0..self.footprints.len());
                }
            } else {
                // One read per selected source span. Packed tail mips
                // share that read; volume spans conservatively cover
                // the full layout. The kernel copy tolerates unmapped pages.
                for ((a, b), indices) in spans {
                    let len = (b - a) as usize;
                    let mut read = crate::shadow::pooled(len);
                    read.resize(len, 0);
                    crate::host::copy_mapped(read.as_mut_ptr(), self.memory.wrapping_add(a as usize), len, false);
                    let mut span_ok = true;
                    for &i in &indices {
                        span_ok &= self.detile_one(&read, i, true);
                    }
                    if span_ok {
                        last.mark(self.footprints.len(), indices);
                    }
                    ok &= span_ok;
                    crate::shadow::give_back(read);
                    read_bytes += len;
                }
            }
            last.ready = ok;
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, read_bytes as u64);
            return (read_bytes, ok && read_bytes != 0);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false);
        }
        let ok = self.detile(tiled);
        last.ready = ok;
        last.hash = ok.then_some(hash);
        if ok {
            last.mark(self.footprints.len(), 0..self.footprints.len());
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, ok)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) -> bool {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        (0..self.footprints.len()).all(|i| self.detile_one(tiled, i, false))
    }

    fn detile_one(&self, tiled: &[u8], i: usize, span: bool) -> bool {
        let footprint = &self.footprints[i];
        let rows = self.rows[i];
        let (mip, slice) = (i as u32 % self.mips + self.first_mip, i as u32 / self.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
        // SAFETY: the length is clamped to the mapped staging allocation.
        // Only this staging's refresh under its last lock writes it.
        let linear = unsafe {
            std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
        };
        let result = if span {
            self.layout.detile_span(tiled, mip, slice, linear, pitch, pitch * rows as usize)
        } else {
            self.layout.detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
        };
        if let Err(ref e) = result {
            tracing::warn!(?e, mip, slice, "texture not detiled");
        }
        result.is_ok()
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    staging_with_admission(resource, texture, None)
}

fn staging_with_admission(
    resource: This,
    texture: &ID3D12Resource,
    admission: Option<&mut crate::upload_pool::Admission>,
) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        s.used_ms.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data, capacity) = crate::upload_pool::take(host, total, admission)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let s = Arc::new(Staging {
        buffer,
        size: total,
        capacity,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        first_mip: crate::resource::dropped(resource),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        last: Mutex::default(),
        used_ms: std::sync::atomic::AtomicU64::new(now_ms()),
    });
    STAGING_BYTES.fetch_add(s.size, std::sync::atomic::Ordering::Relaxed);
    *held = Some(s.clone());
    STAGED.lock().unwrap_or_else(|e| e.into_inner())
        .insert(Arc::as_ptr(&copy) as usize, Arc::downgrade(&copy));
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_upload_mask_survives_clean_refresh_and_failed_detile() {
        let mut last = Last { ready: true, ..Default::default() };
        last.mark(4, [1, 3]);
        last.mark(4, [2]);
        assert_eq!(last.pending, [false, true, true, true]);
        assert!(last.has_pending());
        last.ready = false;
        assert!(!last.has_pending());
        last.mark(4, 0..4);
        last.ready = true;
        assert_eq!(last.pending, [true; 4]);
        last.pending.fill(false);
        assert!(!last.has_pending());
    }

    #[test]
    fn staging_reclaim_never_drops_recorded_owners() {
        assert!(!idle_staging(2, u64::MAX, true));
        assert!(!idle_staging(1, 999, true));
        assert!(idle_staging(1, 1000, true));
        assert!(!idle_staging(1, 9999, false));
        assert!(idle_staging(1, 10000, false));
    }

    #[test]
    fn refresh_chunks_bound_temporary_reads_without_dropping_large_textures() {
        let mb = 1u64 << 20;
        assert_eq!(chunk_len([64 * mb, 64 * mb, mb].into_iter(), 128 * mb, 32), 2);
        assert_eq!(chunk_len([256 * mb, mb].into_iter(), 128 * mb, 32), 1);
        assert_eq!(chunk_len(std::iter::repeat_n(mb, 100), 128 * mb, 32), 32);
        assert_eq!(chunk_len(std::iter::empty(), 128 * mb, 32), 0);
    }

    #[test]
    fn tracked_texture_batch_keeps_staging_results_and_dirty_tracking() {
        use crate::watch::{PAGE, tests as pools};
        let Some(_host) = host() else { return };
        let words: [u32; 14] = [3, 0, 0, 0, 256, 0, 256, (1 << 16) | 1, 28, 1, 0, 1, 0, 0];
        // SAFETY: the C-layout resource description consists of these words.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc).unwrap();
        let size = layout.planes[0].layout.size as usize;
        let mut images = Vec::new();
        let mut uploads = Vec::new();
        for value in [0x23u8, 0x57] {
            let p = pools::pool(size.div_ceil(PAGE));
            // SAFETY: this test owns the writable pool of at least size bytes.
            unsafe { (p as *mut u8).write_bytes(value, size) };
            let mut image = std::ptr::null();
            assert_eq!(crate::resource::create_at_address(
                std::ptr::null(), p as u64, &desc, 0, std::ptr::null(),
                std::ptr::from_ref(&ID3D12Resource::IID).cast(), &mut image
            ), crate::object::S_OK);
            let upload = staging(image, crate::resource::host_resource(image).unwrap()).unwrap();
            assert!(upload.tracked.is_some());
            images.push((p, image));
            uploads.push(upload);
        }
        assert_eq!(refresh_many(&uploads), vec![true, true]);
        for (s, value) in uploads.iter().zip([0x23u8, 0x57]) {
            // SAFETY: both jobs have finished and the staging mappings are live.
            assert_eq!(unsafe { *s.data }, value);
        }
        assert_eq!(refresh_many(&uploads), vec![false, false]);
        // SAFETY: writing our tracked pool marks the second texture dirty.
        unsafe { (images[1].0 as *mut u8).write_volatile(0x91) };
        assert_eq!(refresh_many(&uploads), vec![false, true]);
        drop(uploads);
        for (p, image) in images {
            assert_eq!(crate::object::slot::<extern "system" fn(This) -> u32>(image, 0x10)(image), 0);
            pools::free(p);
        }
    }

    #[test]
    fn tracked_texture_reads_only_dirty_array_slice() {
        use crate::watch::{PAGE, tests as pools};
        let Some(_host) = host() else { return };
        let words: [u32; 14] = [3, 0, 0, 0, 256, 0, 256, (1 << 16) | 2, 28, 1, 0, 1, 0, 0];
        // SAFETY: C-layout resource description.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc).unwrap();
        let layout = &layout.planes[0].layout;
        let size = layout.size as usize;
        let p = pools::pool(size.div_ceil(PAGE));
        let (a, b) = layout.source_span(0, 1).unwrap();
        assert!((a as usize).is_multiple_of(PAGE));
        // SAFETY: test owns the entire mapped allocation.
        unsafe { (p as *mut u8).write_bytes(0x37, size) };
        let mut image = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(
            std::ptr::null(), p as u64, &desc, 0, std::ptr::null(),
            std::ptr::from_ref(&ID3D12Resource::IID).cast(), &mut image
        ), crate::object::S_OK);
        let upload = staging(image, crate::resource::host_resource(image).unwrap()).unwrap();
        assert!(upload.tracked.is_some());
        assert_eq!(upload.refresh_now(), (size, true));
        assert_eq!(upload.refresh_now(), (0, false));
        {
            let mut last = upload.last.lock().unwrap();
            assert_eq!(last.pending, [true, true]);
            // Acknowledge the initial full upload for this mask test.
            last.pending.fill(false);
        }
        // SAFETY: the second array slice belongs to the test allocation.
        unsafe { ((p + a as usize) as *mut u8).write_volatile(0x91) };
        assert_eq!(upload.refresh_now(), ((b - a) as usize, true));
        assert_eq!(upload.refresh_now(), (0, false));
        assert_eq!(upload.last.lock().unwrap().pending, [false, true]);
        // SAFETY: both staging subresources remain live.
        unsafe {
            assert_eq!(*upload.data.add(upload.footprints[0].Offset as usize), 0x37);
            assert_eq!(*upload.data.add(upload.footprints[1].Offset as usize), 0x91);
        }
        drop(upload);
        assert_eq!(crate::object::slot::<extern "system" fn(This) -> u32>(image, 0x10)(image), 0);
        pools::free(p);
    }

    #[test]
    fn retained_texture_staging_reads_original_mip_and_slice() {
        use windows::core::Interface;
        let Some(_host) = host() else { return };
        let words: [u32; 14] = [3, 0, 0, 0, 512, 0, 512, (5 << 16) | 2, 28, 1, 0, 1, 0, 0];
        // SAFETY: the C-layout resource description consists of these words.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let title = &crate::resource::texture_layout(&desc).unwrap().planes[0].layout;
        let bytes: Vec<u8> = (0..title.size).map(|i| (i ^ (i >> 8) ^ (i >> 16)) as u8).collect();
        let mut image = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(
            std::ptr::null(), bytes.as_ptr() as u64, &desc, 0, std::ptr::null(),
            std::ptr::from_ref(&ID3D12Resource::IID).cast(), &mut image
        ), crate::object::S_OK);
        let upload = staging(image, crate::resource::host_resource(image).unwrap()).unwrap();
        let drop = crate::resource::dropped(image);
        assert_eq!(upload.first_mip, drop);
        assert_eq!(upload.mips, 5 - drop);
        assert!(upload.detile(&bytes));
        for slice in 0..2 {
            let index = (slice * upload.mips) as usize;
            let f = upload.footprints[index];
            let pitch = f.Footprint.RowPitch as usize;
            let rows = upload.rows[index] as usize;
            let len = pitch * rows;
            let mut expected = vec![0u8; len];
            title.detile(&bytes, drop, slice, &mut expected, pitch, len).unwrap();
            // This first retained level is large enough to fill whole rows.
            // SAFETY: this subresource is within the live staging allocation.
            let got = unsafe { std::slice::from_raw_parts(upload.data.add(f.Offset as usize), len) };
            assert_eq!(got, expected.as_slice(), "retained mip {drop}, slice {slice}");
        }
        std::mem::drop(upload);
        assert_eq!(crate::object::slot::<extern "system" fn(This) -> u32>(image, 0x10)(image), 0);
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
