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
//! Tracked textures detile retained source spans touched by dirty pages;
//! untracked textures may opt into per-span content hashes.
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

/// A texture's upload state: how a staging buffer lays out the host
/// texture's subresources, and what was last detiled. The buffer itself is
/// taken for each upload from a small pool shared by all textures
/// ([`staging_buffer`]; kept for each texture's life, the buffers took its
/// size again in system memory: 2.6-2.8 GB with placed textures made on
/// read, run integ2).
pub(crate) struct Staging {
    /// A staging buffer's bytes.
    size: u64,
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
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// Retry a failed detile even when the dirty tracker is now clean.
    retry: std::sync::atomic::AtomicBool,
    /// Uploads [`record_upload`] recorded, and the subresources they
    /// copied.
    uploads: std::sync::atomic::AtomicU64,
    copied: std::sync::atomic::AtomicU64,
}

/// What was last detiled: the hash of the memory (not tracked), or of each
/// subresource's bytes (tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    hashes: Vec<Option<blake3::Hash>>,
    // Retained source-span hashes for the opt-in untracked path.
    span_hashes: SpanHashes,
    // A layout that rejected bounded detiling uses the full-source fallback.
    full_source: bool,
}

type SourceGroups = std::collections::BTreeMap<(usize, usize), Vec<usize>>;
type SpanHashes = std::collections::BTreeMap<(usize, usize), blake3::Hash>;

/// Exact conservative source interval from the format layout. Packed tails
/// share an interval; volumes keep the full layout. No corner-based
/// approximation is used for swizzled layouts.
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32, first_mip: u32) -> Vec<(usize, usize)> {
    if mips == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips + first_mip, i / mips);
            let (a, b) = layout.source_span(mip, slice)?;
            Some((usize::try_from(a).ok()?, usize::try_from(b).ok()?))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Group mips that share packed source bytes, as in the original fork's
/// tracked and untracked source-span refreshes. Unsupported intervals
/// are rejected before touching any GPU staging memory.
fn source_groups(spans: &[(usize, usize)], count: usize, size: usize) -> Option<SourceGroups> {
    if spans.len() != count {
        return None;
    }
    let mut groups = SourceGroups::new();
    for (i, &(a, b)) in spans.iter().enumerate() {
        if a >= b || b > size {
            return None;
        }
        groups.entry((a, b)).or_default().push(i);
    }
    Some(groups)
}

fn span_dirty((a, b): (usize, usize), start: usize, dirty: &[(usize, usize)]) -> bool {
    dirty.iter().any(|&(from, to)| from < start + b && to > start + a)
}

/// The old fork's per-span hash census, independent of mutable descriptors.
/// Return the retained subresources whose source changed, once per span.
fn changed_source_spans(groups: &SourceGroups, previous: &SpanHashes, tiled: &[u8]) -> (SpanHashes, Vec<u32>) {
    let mut next = SpanHashes::new();
    let mut fresh = Vec::new();
    for (&(a, b), indices) in groups {
        let digest = blake3::hash(&tiled[a..b]);
        if previous.get(&(a, b)) != Some(&digest) {
            fresh.extend(indices.iter().map(|&i| i as u32));
        }
        next.insert((a, b), digest);
    }
    fresh.sort_unstable();
    (next, fresh)
}

/// Staging buffers for detiles to reuse, by size: the menu movie rewrites
/// the same textures every frame, and a fresh 12.7 MB of upload buffers a
/// frame cost ~5 ms (run textures8). Uploads run only inside
/// `Host::execute`, which waits for each submission, so the buffers of one
/// submission are free from the next on.
struct Buffers {
    /// (bytes, buffer, mapping), oldest first.
    free: std::collections::VecDeque<(u64, ID3D12Resource, usize)>,
    free_bytes: u64,
    /// Taken by the submission being prepared.
    used: Vec<(u64, ID3D12Resource, usize)>,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Buffers {}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    free: std::collections::VecDeque::new(),
    free_bytes: 0,
    used: Vec::new(),
});

/// Free staging buffers kept at most.
const KEPT_BYTES: u64 = 64 << 20;

/// Called before a submission's uploads: the buffers of the last one are
/// free again (the oldest dropped past `KEPT_BYTES`).
pub(crate) fn recycle() {
    // The previous submission has completed; only idle allocations are reclaimed.
    // Follow the former upload_pool idle-limit policy on RAM or DXGI pressure.
    let pressure = crate::shadow::memory_pressured() || crate::host::video_memory_pressured();
    let (kept_bytes, chunks_kept) = if pressure { (0, 0) } else { (KEPT_BYTES, CHUNKS_KEPT) };
    {
        let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
        (c.current, c.used) = (0, 0);
        c.all.truncate(chunks_kept);
    }
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let b = &mut *b;
    for e in b.used.drain(..) {
        b.free_bytes += e.0;
        b.free.push_back(e);
    }
    while b.free_bytes > kept_bytes
        && let Some((n, ..)) = b.free.pop_front()
    {
        b.free_bytes -= n;
    }
}

/// A free staging buffer of `size` bytes, or a new one.
/// Sizes are taken in powers of two (at least 64 KiB), so a buffer serves
/// textures of other sizes too: a new shot's textures came in new sizes and
/// made buffers 10 % of the submitting thread (run evict5).
fn staging_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let class = size.max(1 << 16).next_power_of_two();
    let taken = {
        let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.free.iter().position(|e| e.0 == class);
        i.and_then(|i| b.free.remove(i)).inspect(|_| b.free_bytes -= class)
    };
    // Made without the pool's lock (refreshes run on several threads).
    let e = match taken {
        Some(e) => e,
        None => {
            let (buffer, data) = host()?.upload_buffer(class)?;
            crate::host::name(&buffer, || "texture staging".into());
            (class, buffer, data as usize)
        }
    };
    let made = (e.1.clone(), e.2 as *mut u8);
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).used.push(e);
    Some(made)
}

/// Staging space carved from large upload buffers, front to back: most
/// textures take a part of one instead of a buffer of their own (making
/// upload buffers was 10 % of the submitting thread in a new shot, run
/// evict5). Rewound at each submission ([`recycle`]): the last one has run.
struct Chunks {
    /// (buffer, mapping), [`CHUNK`] bytes each.
    all: Vec<(ID3D12Resource, usize)>,
    /// The chunk being carved and the bytes taken from it.
    current: usize,
    used: u64,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Chunks {}

static CHUNKS: Mutex<Chunks> = Mutex::new(Chunks { all: Vec::new(), current: 0, used: 0 });

const CHUNK: u64 = 32 << 20;
/// Chunks kept between submissions (more are made while one needs them).
const CHUNKS_KEPT: usize = 16;

/// `size` bytes of staging space: a buffer, the offset of the space in it
/// (a multiple of 512, as texture copies need) and its mapping there.
/// Space larger than a chunk is a buffer of its own ([`staging_buffer`]).
fn staging_space(size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    if size > CHUNK {
        return staging_buffer(size).map(|(b, data)| (b, 0, data));
    }
    let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some((buffer, data)) = c.all.get(c.current) {
            let at = c.used.next_multiple_of(512);
            if at + size <= CHUNK {
                let made = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                c.used = at + size;
                return Some(made);
            }
            (c.current, c.used) = (c.current + 1, 0);
            continue;
        }
        let (buffer, data) = host()?.upload_buffer(CHUNK)?;
        crate::host::name(&buffer, || "texture staging chunk".into());
        c.all.push((buffer, data as usize));
    }
}

// SAFETY: the title memory is only read, under `last`'s lock.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve that `list` records writes `resource`: from the
/// submission that runs `list` on, its host texture's contents are the
/// host's, not title memory's, which stays as it was. Marked when that
/// submission is made, not when `list` is recorded: a list recorded before
/// it but run after it must not upload title memory over the host's texels
/// (submissions run in the order they are made, which need not be the
/// order their lists were recorded in), and one run before it must still
/// get its upload.
pub(crate) fn host_writes(list: &HostList, resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        list.defer_host(move || copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed));
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
    let pairs: Vec<(usize, usize)> = if crate::resource::dropped(from, &a) > 0 || crate::resource::dropped(to, &b) > 0 {
        let count = crate::resource::title_subresources(from).min(crate::resource::title_subresources(to));
        (0..count).filter_map(|i| {
            Some((crate::resource::host_subresource(from, &a, i)? as usize,
                  crate::resource::host_subresource(to, &b, i)? as usize))
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
    crate::texture::host_writes(list, to);
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
            l.CopyTextureRegion(&place(&buffer, fa[i]), 0, 0, 0, &sub(&a, i as u32), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for &(i, j) in &pairs {
            let mut footprint = fb[j];
            footprint.Offset = fa[i].Offset;
            l.CopyTextureRegion(&sub(&b, j as u32), 0, 0, 0, &place(&buffer, footprint), None);
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

/// Has `resource`'s host copy brought up to date at the submission that
/// runs `list`: its staging buffer detiled again if the memory changed, and
/// then copied into the host texture by the submission's own upload list,
/// once for all its lists (`crate::bindless::uploads`).
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(resource as usize) {
                let memory = crate::resource::texture_memory(resource).map(|(m, _)| m);
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                tracing::debug!(?resource, ?memory, ?desc, "a sampled texture keeps what the host copied into it: not uploaded from title memory");
            }
        }
        return;
    }
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (staging(resource, texture), crate::resource::host_copy(resource)) else {
        return;
    };
    // Each list recorded the copy before (run perf7: 149 textures uploaded
    // 3,195 times in a frame, 17 ms of GPU time), changed or not.
    list.defer_texture(staging, copy);
}

/// Records into `list` the copy of `buffer` (laid out as `staging` says)
/// into host texture `copy`. Returns the barrier that leaves it readable by
/// shaders, for the caller to record with the others after every copy: one
/// barrier per texture between the copies kept the GPU from overlapping
/// them (new shots copy hundreds of textures, run evict6).
#[must_use]
pub(crate) fn record_upload(
    list: &HostList,
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    (buffer, at, subresources): &Upload,
) -> Option<D3D12_RESOURCE_BARRIER> {
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    if let Some(b) = list.move_texture(copy, COPY_DEST) {
        // SAFETY: an open list; the barrier names a live texture.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
    copies(list, &copy.texture, staging, (buffer, *at), subresources, String::new);
    list.keep(buffer);
    staging.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    staging.copied.fetch_add(subresources.len() as u64, std::sync::atomic::Ordering::Relaxed);
    list.move_texture(copy, SHADER_RESOURCE)
}

/// The uploads [`record_upload`] recorded into `resource`'s host texture
/// (tests: counts of their own texture, whatever other tests upload
/// meanwhile).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// The subresources those uploads copied (tests, as [`uploads`]).
#[cfg(test)]
pub(crate) fn copied(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.copied.load(std::sync::atomic::Ordering::Relaxed))
}


/// Records the copy of `subresources` of `buffer` (laid out as `staging`
/// says) into `texture` (in the copy destination state); `name` names it
/// in diagnostics notes.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
fn copies(
    list: &HostList,
    texture: &ID3D12Resource,
    staging: &Staging,
    (buffer, at): (&ID3D12Resource, u64),
    subresources: &[u32],
    name: impl Fn() -> String,
) {
    for &i in subresources {
        let i = i as usize;
        let mut footprint = staging.footprints[i];
        footprint.Offset += at;
        let footprint = &footprint;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        name(),
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
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// What a refresh gives: a staging buffer, where its space starts in it,
/// and the subresources detiled into it.
pub(crate) type Upload = (ID3D12Resource, u64, Vec<u32>);

/// Runs `f(i)` for each `i` below `n` on up to a quarter of the CPUs (four
/// at most: the title's own threads need the rest), here when that is
/// fewer than two.
pub(crate) fn in_parallel(n: usize, f: impl Fn(usize) + Sync) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(1, 4).min(n);
    if threads < 2 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                }
            });
        }
    });
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

fn parallel_untracked() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VICTORY_PARALLEL_UNTRACKED").as_deref() == Ok("1"))
}

/// Prepare only changed textures; results remain in the original order.
/// Apply the old fork's bounded worker and chunk policy to the upstream
/// submission-owned upload buffers. No worker outlives this call.
pub(crate) fn refresh_all(work: &[(Arc<Staging>, Arc<crate::resource::HostTexture>)]) -> Vec<Option<Upload>> {
    use std::sync::atomic::Ordering::Relaxed;
    let changed: Vec<usize> = (0..work.len()).filter(|&i| work[i].0.maybe_changed()).collect();
    let mut out: Vec<Option<Upload>> = vec![None; work.len()];
    if let [i] = changed.as_slice() {
        out[*i] = work[*i].0.refresh();
        return out;
    }
    let started = std::time::Instant::now();
    let mut read = 0usize;
    let mut at = 0;
    let allow_untracked = parallel_untracked();
    while at < changed.len() {
        let threads = if crate::shadow::memory_pressured() {
            1
        } else {
            crate::texture_workers::configured_threads()
        };
        let first = &work[changed[at]].0;
        if threads < 2 || (!allow_untracked && first.tracked.is_none()) {
            let (bytes, upload) = first.refresh_now();
            read = read.saturating_add(bytes);
            out[changed[at]] = upload;
            at += 1;
            continue;
        }
        let n = chunk_len(changed[at..].iter()
            .map(|&i| &work[i].0)
            .take_while(|s| allow_untracked || s.tracked.is_some())
            .map(|s| s.layout.size as u64), REFRESH_CHUNK_BYTES, REFRESH_CHUNK_COUNT);
        debug_assert!(n > 0);
        let results = if n == 1 {
            vec![first.refresh_now()]
        } else {
            let jobs = changed[at..at + n].iter().map(|&i| work[i].0.clone()).collect();
            crate::texture_workers::ordered_map(jobs, threads, |staging: Arc<Staging>| staging.refresh_now_with(true))
        };
        for (k, (bytes, upload)) in results.into_iter().enumerate() {
            read = read.saturating_add(bytes);
            out[changed[at + k]] = upload;
        }
        at += n;
    }
    let preparing = &crate::host::PREPARING;
    preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    preparing[3].fetch_add(read as u64, Relaxed);
    out
}

impl Staging {
    /// Whether its memory may have changed since it was last detiled (an
    /// untracked texture always may).
    pub(crate) fn maybe_changed(&self) -> bool {
        self.retry.load(std::sync::atomic::Ordering::Relaxed)
            || self.tracked.as_ref().is_none_or(|t| !t.clean())
    }

    /// A staging buffer with the texture memory detiled into it if it
    /// changed since the last time (and is mapped at all), and the
    /// subresources that changed (only those are detiled and to be copied).
    pub(crate) fn refresh(&self) -> Option<Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, buffer) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        buffer
    }

    /// Returns the bytes of title memory read, and the buffer it detiled
    /// into with the subresources detiled.
    fn refresh_now(&self) -> (usize, Option<Upload>) {
        self.refresh_now_with(false)
    }

    fn refresh_now_with(&self, checked_untracked: bool) -> (usize, Option<Upload>) {
        let all = || (0..self.footprints.len() as u32).collect::<Vec<_>>();
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            // Most tracked refreshes remain a single clean-bit check.
            use std::sync::atomic::Ordering::Relaxed;
            let retry = self.retry.load(Relaxed);
            if t.clean() && !retry {
                return (0, None);
            }
            let ranges = t.take(start, start + size);
            if ranges.is_empty() && !retry {
                return (0, None);
            }
            let n = self.footprints.len();
            // Port bd1c79f7's bounded per-span scratch reads, retaining
            // the current renderer's per-submission staging allocation.
            let groups = (!last.full_source).then(|| source_groups(&self.spans, n, size)).flatten();
            let bounded = groups.is_some();
            let groups = groups.unwrap_or_else(|| {
                std::iter::once(((0, size), (0..n).collect())).collect()
            });
            let mut bytes = 0usize;
            let mut fresh = Vec::new();
            let mut staging = None;
            let mut failed = false;
            last.hashes.resize(n, None);
            let page = crate::watch::PAGE;
            'groups: for ((a, b), indices) in groups {
                if !retry && !span_dirty((a, b), start, &ranges) {
                    continue;
                }
                // Only the selected span is read into CPU scratch. Packed
                // tails share this read; an unsupported layout reads whole.
                let mut scratch = Scratch::take(b - a);
                crate::host::copy_mapped(scratch.0.as_mut_ptr(), self.memory.wrapping_add(a), b - a, false);
                bytes = bytes.saturating_add(b - a);
                // Preserve the upstream whole-dirty-page fast path and
                // hash shared pages only once per distinct source span.
                let (a0, b0) = (start + a, start + b);
                let whole_page = ranges.iter().any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0));
                let digest = (!whole_page).then(|| blake3::hash(&scratch.0));
                let changed: Vec<usize> = indices.into_iter().filter(|&i| {
                    let changed = retry || whole_page || last.hashes[i] != digest;
                    last.hashes[i] = digest;
                    changed
                }).collect();
                if changed.is_empty() {
                    continue;
                }
                if staging.is_none() {
                    staging = staging_space(self.size);
                }
                let Some((_, _, data)) = &staging else {
                    failed = true;
                    break;
                };
                for i in changed {
                    if !self.detile_one(&scratch.0, *data, i, bounded) {
                        // Retry using the original full-layout path if the
                        // bounded format walk rejected this layout.
                        last.full_source |= bounded;
                        failed = true;
                        break 'groups;
                    }
                    fresh.push(i as u32);
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, bytes as u64);
            if failed {
                // Consumed dirty pages alone cannot schedule a retry.
                // Never publish a partially prepared upload.
                last.hashes.clear();
                self.retry.store(true, Relaxed);
                return (bytes, None);
            }
            self.retry.store(false, Relaxed);
            fresh.sort_unstable();
            return (bytes, staging.map(|(b, at, _)| (b, at, fresh)));
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            // Never uploaded: zeros, as a committed texture starts (one in
            // a heap of textures starts with whatever the range held).
            if last.hash.is_some() {
                return (0, None);
            }
            let zeros = vec![0u8; size];
            let every = all();
            let buffer = self.detile(&zeros, &every);
            last.hash = buffer.is_some().then(|| blake3::hash(&zeros));
            return (0, buffer.map(|(b, at)| (b, at, every)));
        }
        if checked_untracked {
            // The old fork's checked kernel snapshot for untracked worker
            // reads, using this integration's bounded RAII scratch pool.
            let mut snapshot = Scratch::take(size);
            if crate::host::read_into(self.memory, &mut snapshot.0) {
                return self.refresh_untracked(last, &snapshot.0);
            }
            return (0, None);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        self.refresh_untracked(last, tiled)
    }

    fn refresh_untracked(&self, last: &mut Last, tiled: &[u8]) -> (usize, Option<Upload>) {
        static PARTIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let partial = *PARTIAL.get_or_init(|| std::env::var("VICTORY_PARTIAL_UNTRACKED_UPLOADS").as_deref() == Ok("1"));
        let size = tiled.len();
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, None);
        }
        // 2c9e2a13: retain the whole-image digest, but skip detiling and
        // GPU copies for source spans whose contents did not change.
        if partial && self.footprints.len() > 1 && self.footprints.len() <= 256
            && let Some(groups) = source_groups(&self.spans, self.footprints.len(), size)
        {
            let (next, fresh) = changed_source_spans(&groups, &last.span_hashes, tiled);
            if fresh.is_empty() {
                last.hash = Some(hash);
                last.span_hashes = next;
                return (size, None);
            }
            let buffer = self.detile(tiled, &fresh);
            if buffer.is_some() {
                last.hash = Some(hash);
                last.span_hashes = next;
            } else {
                // The upstream staging allocation is per submission. On a
                // failure retry all retained spans and publish no upload.
                last.hash = None;
                last.span_hashes.clear();
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            return (size, buffer.map(|(b, at)| (b, at, fresh)));
        }
        last.span_hashes.clear();
        let every: Vec<u32> = (0..self.footprints.len() as u32).collect();
        let buffer = self.detile(tiled, &every);
        last.hash = buffer.is_some().then_some(hash);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, buffer.map(|(b, at)| (b, at, every)))
    }

    /// Detiles `subresources` of `tiled` (the texture's memory) into a
    /// staging buffer (`None` if none could be had or a subresource could
    /// not be read).
    fn detile(&self, tiled: &[u8], subresources: &[u32]) -> Option<(ID3D12Resource, u64)> {
        let (buffer, at, data) = staging_space(self.size)?;
        #[cfg(feature = "diagnostics")]
        // Scanned only when its message is wanted: a pass over every
        // texture read, in timing runs too.
        if tracing::enabled!(tracing::Level::DEBUG) {
            if tiled.iter().all(|&b| b == 0) {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            } else {
                tracing::debug!(memory = ?self.memory, desc = ?self.layout.desc, "a sampled texture detiled");
            }
        }
        for &i in subresources {
            if !self.detile_one(tiled, data, i as usize, false) {
                return None;
            }
        }
        Some((buffer, at))
    }

    /// Port the original fork's one-subresource format walk, targeting a
    /// range owned by this submission instead of a permanent mapping.
    fn detile_one(&self, tiled: &[u8], data: *mut u8, i: usize, span: bool) -> bool {
        let footprint = &self.footprints[i];
        let rows = self.rows[i];
        let (mip, slice) = (i as u32 % self.mips + self.first_mip, i as u32 / self.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
        // SAFETY: the length is clamped to the staging allocation; this
        // worker owns that range until the submission is prepared.
        let linear = unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
        let result = if span {
            self.layout.detile_span(tiled, mip, slice, linear, pitch, pitch * rows as usize)
        } else {
            self.layout.detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
        };
        if let Err(e) = &result {
            tracing::warn!(?e, mip, slice, "texture not detiled");
        }
        result.is_ok()
    }
}

/// A temporary read buffer from the bounded shared CPU pool. The active
/// read remains owned by this refresh until detiling finishes.
struct Scratch(Vec<u8>);

impl Scratch {
    fn take(size: usize) -> Self {
        let mut v = crate::shadow::pooled(size);
        v.resize(size, 0);
        Scratch(v)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        crate::shadow::give_back(std::mem::take(&mut self.0));
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
pub(crate) fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
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
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let first_mip = crate::resource::dropped(resource, texture);
    let spans = spans(&layout, count, u32::from(d.MipLevels), first_mip);
    let s = Arc::new(Staging {
        size: total,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        first_mip,
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        spans,
        last: Mutex::default(),
        retry: true.into(),
        uploads: Default::default(),
        copied: Default::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_chunks_bound_temporary_reads_without_dropping_large_textures() {
        let mb = 1u64 << 20;
        assert_eq!(chunk_len([64 * mb, 64 * mb, mb].into_iter(), 128 * mb, 32), 2);
        assert_eq!(chunk_len([256 * mb, mb].into_iter(), 128 * mb, 32), 1);
        assert_eq!(chunk_len(std::iter::repeat_n(mb, 100), 128 * mb, 32), 32);
        assert_eq!(chunk_len(std::iter::empty(), 128 * mb, 32), 0);
    }

    #[test]
    fn tracked_source_groups_select_only_intersecting_array_slices() {
        let groups = source_groups(&[(0, 64), (64, 128), (64, 128)], 3, 128).unwrap();
        assert_eq!(groups.get(&(64, 128)), Some(&vec![1, 2]));
        let dirty = [(1000 + 64, 1000 + 128)];
        let touched: Vec<_> = groups.iter()
            .filter(|(span, _)| span_dirty(**span, 1000, &dirty))
            .flat_map(|(_, indices)| indices.iter().copied()).collect();
        assert_eq!(touched, [1, 2]);
        assert!(source_groups(&[(0, 129)], 1, 128).is_none());
        assert!(source_groups(&[(64, 64)], 1, 128).is_none());
        assert!(source_groups(&[(0, 64)], 2, 128).is_none());
    }

    #[test]
    fn untracked_span_hashes_skip_unchanged_slices_and_share_packed_tails() {
        let groups = source_groups(&[(0, 64), (64, 128), (64, 128)], 3, 128).unwrap();
        let mut bytes = vec![0x37u8; 128];
        let (first, changed) = changed_source_spans(&groups, &SpanHashes::new(), &bytes);
        assert_eq!(changed, [0, 1, 2]);
        assert_eq!(first.len(), 2);
        assert!(changed_source_spans(&groups, &first, &bytes).1.is_empty());
        bytes[64] = 0x91;
        let (second, changed) = changed_source_spans(&groups, &first, &bytes);
        assert_eq!(changed, [1, 2]);
        assert!(changed_source_spans(&groups, &second, &bytes).1.is_empty());
        bytes[0] = 0x72;
        assert_eq!(changed_source_spans(&groups, &second, &bytes).1, [0]);
        // A failed upload clears the census, causing every span to retry.
        assert_eq!(changed_source_spans(&groups, &SpanHashes::new(), &bytes).1, [0, 1, 2]);
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
        let texture = crate::resource::host_resource(image).unwrap();
        let upload = staging(image, &texture).unwrap();
        let drop = crate::resource::dropped(image, &texture);
        assert_eq!(upload.first_mip, drop);
        assert_eq!(upload.mips, 5 - drop);
        let full = spans(title, 10, 5, 0);
        assert!(!full.is_empty());
        for slice in 0..2 {
            for mip in 0..upload.mips {
                let host_index = (slice * upload.mips + mip) as usize;
                let title_index = (slice * 5 + mip + drop) as usize;
                assert_eq!(upload.spans[host_index], full[title_index]);
                let f = upload.footprints[host_index];
                let pitch = f.Footprint.RowPitch as usize;
                let rows = upload.rows[host_index] as usize;
                let len = pitch * rows;
                let mut expected = vec![0u8; len];
                title.detile(&bytes, mip + drop, slice, &mut expected, pitch, len).unwrap();
                let (a, b) = upload.spans[host_index];
                let mut actual = vec![0u8; len];
                title.detile_span(&bytes[a..b], mip + drop, slice, &mut actual, pitch, len).unwrap();
                assert_eq!(actual, expected);
            }
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
