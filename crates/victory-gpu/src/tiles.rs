//! Reserved sampled textures and a shared tile heap pool.
//!
//! Adapted from DoubleGhosty/cool_fox tiles.rs (f4b48ad): 64 KiB tiles,
//! 64 MiB heaps, free runs, standard MIP groups and atomic packed tails.
//! Track and extend indivisible mapping groups, retaining their leases
//! through the resource's COM lifetime. The title path still backs every
//! retained level before publication. Partial expansion is exercised by
//! an isolated GPU test; live descriptor and GPU-write coordination remain
//! prerequisites for exposing sparse resources to title commands.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::{AcqRel, Relaxed}};
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface};
use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

const TILE: u64 = 64 << 10;
const HEAP_TILES: u32 = 1024;
const MAX_HEAPS: usize = 16;
const MIN_BYTES: u64 = 1 << 20;
const MAX_BYTES: u64 = HEAP_TILES as u64 * TILE;

type Run = (u32, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Group {
    first: u32,
    levels: u32,
    tiles: u32,
}

fn groups(standard: &[u32], packed_levels: u32, packed_tiles: u32, total: u32) -> Option<Vec<Group>> {
    let levels = u32::try_from(standard.len()).ok()?.checked_add(packed_levels)?;
    if levels == 0 || levels > 16 || total == 0 || total > HEAP_TILES
        || (packed_levels == 0) != (packed_tiles == 0)
        || standard.iter().any(|&n| n == 0 || n > HEAP_TILES)
    {
        return None;
    }
    let sum = standard.iter().try_fold(packed_tiles, |a, &b| a.checked_add(b))?;
    if sum != total {
        return None;
    }
    let mut result: Vec<_> = standard.iter().enumerate().map(|(i, &tiles)|
        Group { first: i as u32, levels: 1, tiles }).collect();
    if packed_levels != 0 {
        result.push(Group { first: standard.len() as u32, levels: packed_levels, tiles: packed_tiles });
    }
    Some(result)
}

impl Group {
    fn mask(self) -> u32 {
        ((1u32 << self.levels) - 1) << self.first
    }
}

fn selected(groups: &[Group], mask: u32) -> impl Iterator<Item = Group> + '_ {
    groups.iter().copied().filter(move |g| g.mask() & mask != 0)
}

/// cool_fox Tiled::with_level/without/give_locked: requesting one packed
/// level allocates the entire tail, and previously mapped groups stay put.
fn group_mask(groups: &[Group], mask: u32) -> u32 {
    selected(groups, mask).fold(0, |bits, group| bits | group.mask())
}

fn missing_groups(groups: &[Group], mapped: u32, wanted: u32) -> impl Iterator<Item = Group> + '_ {
    selected(groups, wanted).filter(move |g| g.mask() & mapped == 0)
}

fn free_count(runs: &[Run]) -> u32 {
    runs.iter().map(|r| r.1).sum()
}

/// All-or-nothing allocation, consuming the lowest free runs first.
fn cut(runs: &mut Vec<Run>, count: u32) -> Option<Vec<Run>> {
    if count == 0 || free_count(runs) < count {
        return None;
    }
    let mut remaining = count;
    let mut result = Vec::new();
    while remaining != 0 {
        let (start, len) = runs[0];
        let n = remaining.min(len);
        result.push((start, n));
        if n == len {
            runs.remove(0);
        } else {
            runs[0] = (start + n, len - n);
        }
        remaining -= n;
    }
    Some(result)
}

fn join(free: &mut Vec<Run>, returned: &[Run]) {
    free.extend_from_slice(returned);
    free.sort_unstable();
    let mut result: Vec<Run> = Vec::with_capacity(free.len());
    for &(start, len) in free.iter() {
        if let Some(last) = result.last_mut() {
            debug_assert!(last.0 + last.1 <= start);
            if last.0 + last.1 == start {
                last.1 += len;
                continue;
            }
        }
        result.push((start, len));
    }
    *free = result;
}

struct Heap {
    heap: ID3D12Heap,
    free: Mutex<Vec<Run>>,
}

// SAFETY: D3D12 heaps are free-threaded and the allocator state is locked.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

static POOL: Mutex<Vec<Arc<Heap>>> = Mutex::new(Vec::new());
static MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

struct Lease {
    heap: Arc<Heap>,
    runs: Vec<Run>,
    count: u32,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut free = self.heap.free.lock().unwrap_or_else(|e| e.into_inner());
        join(&mut free, &self.runs);
        MAPPED_BYTES.fetch_sub(u64::from(self.count) * TILE, Relaxed);
    }
}

fn pressured() -> bool {
    crate::shadow::memory_pressured()
        || crate::host::video_memory_pressured()
}

fn take(device: &ID3D12Device, count: u32) -> Option<Lease> {
    if count == 0 || count > HEAP_TILES {
        return None;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    for heap in pool.iter() {
        let mut free = heap.free.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(runs) = cut(&mut free, count) {
            MAPPED_BYTES.fetch_add(u64::from(count) * TILE, Relaxed);
            return Some(Lease { heap: heap.clone(), runs, count });
        }
    }
    if pool.len() >= MAX_HEAPS || pressured() {
        return None;
    }
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: MAX_BYTES,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        ..Default::default()
    };
    let mut made: Option<ID3D12Heap> = None;
    // SAFETY: the description and output remain live during the call.
    unsafe { device.CreateHeap::<ID3D12Heap>(&desc, &mut made) }.ok()?;
    let mut free = vec![(0, HEAP_TILES)];
    let runs = cut(&mut free, count)?;
    let heap = Arc::new(Heap { heap: made?, free: Mutex::new(free) });
    pool.push(heap.clone());
    MAPPED_BYTES.fetch_add(u64::from(count) * TILE, Relaxed);
    Some(Lease { heap, runs, count })
}

/// Only completely free heaps are trimmed. Active allocations have
/// their own Arc<Heap>, including resources retained by descriptor slots.
pub(crate) fn maintain() {
    let pressure = pressured();
    let mut idle = 0usize;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.retain(|heap| {
        if Arc::strong_count(heap) != 1 {
            return true;
        }
        let keep = !pressure && idle == 0;
        idle += 1;
        keep
    });
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn usage() -> (u64, u64) {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    (MAPPED_BYTES.load(Relaxed), pool.len() as u64 * MAX_BYTES)
}

/// Per-resource group residency, adapted from cool_fox's Tiled/Given.
/// The mask is published only after a successful initialization fence.
/// Each group owns its leases; existing groups are never remapped.
pub(crate) struct Mapping {
    groups: Vec<Group>,
    mapped: u32,
    leases: Vec<Lease>,
    /// Levels held by title data and levels actually written by the card.
    /// The production full-chain path treats every initialized level as held.
    held: u32,
    /// Advisory held-level discovery. Never used to remove backing or
    /// change the shown level until the sparse title path is validated.
    observed_held: Option<u32>,
    /// Title-source bytes actually copied by a completed GPU upload.
    /// Full-chain zero initialization alone does not set these bits.
    source_uploaded: u32,
    /// The reference checks waiting source levels about once a second.
    /// No prepared upload allocation is retained across that wait.
    source_retry_at: Option<std::time::Instant>,
    card: u32,
    /// Packed-group claims are acquired at submission, never recording.
    /// Production initializes every group before a list records a write.
    claimed: u32,
}

impl Mapping {
    fn new(groups: Vec<Group>) -> Self {
        Self { groups, mapped: 0, leases: Vec::new(), held: 0, observed_held: None, source_uploaded: 0, source_retry_at: None, card: 0, claimed: 0 }
    }

    fn every_level(&self) -> u32 {
        group_mask(&self.groups, u32::MAX)
    }

    /// Reference Tiled::without/claim/unclaim semantics, synchronized by
    /// this mapping's mutex. Claim whole packed groups, not just one MIP.
    fn claim(&mut self, levels: u32) -> u32 {
        let missing = group_mask(&self.groups, levels) & !self.mapped;
        let fresh = missing & !self.claimed;
        self.claimed |= fresh;
        fresh
    }

    fn unclaim(&mut self, levels: u32) {
        self.claimed &= !group_mask(&self.groups, levels);
    }

    /// Reference card_wrote's successful-backed-level mask. Called only
    /// after the real list's completion fence. It never maps tiles or
    /// acknowledges unbacked writes. Preserve exact authored subresources,
    /// rather than marking the entire packed tail as GPU-written.
    pub(crate) fn card_wrote(&mut self, levels: u32) -> bool {
        let previous = self.visible_mask();
        self.card |= levels & self.mapped;
        self.unclaim(levels);
        previous != self.visible_mask()
    }

    /// Finish the reference held-level upload only after the actual
    /// GPU copy has run. Mapping or preparing a staging buffer does not
    /// publish newly backed MIPs. Preserve the separate GPU-write mask.
    pub(crate) fn source_wrote(&mut self, levels: u32) -> bool {
        let previous = self.visible_mask();
        let completed = levels & self.mapped;
        self.source_uploaded |= completed;
        self.held |= completed;
        previous != self.visible_mask()
    }

    pub(crate) fn observe_held(&mut self, levels: u32) -> bool {
        let levels = levels & self.every_level();
        let changed = self.observed_held != Some(levels);
        self.observed_held = Some(levels);
        changed
    }

    /// Unlike the reference's single first_shown value, retain every
    /// shown bit so a hole filling behind the first level can also
    /// invalidate a descriptor. The normal full-chain path is unchanged.
    pub(crate) fn visible_mask(&self) -> u32 {
        (self.held | self.card) & self.mapped
    }

    pub(crate) fn fully_backed(&self) -> bool {
        self.mapped == self.every_level()
    }

    /// Reference first_shown, retained for isolated policy/readback tests.
    #[cfg(test)]
    fn first_shown(&self) -> Option<u32> {
        let shown = self.visible_mask();
        (shown != 0).then(|| shown.trailing_zeros())
    }

    fn given_bytes(&self) -> u64 {
        self.leases.iter().map(|lease| u64::from(lease.count) * TILE).sum()
    }

    /// Port the reference Tiled::uploaded_on admission policy: title-held
    /// MIPs may wait for memory room; GPU-authored MIPs never wait.
    /// Existing backing is kept even if a later title judgement shrinks.
    /// Returns whether any named levels still lack physical backing.
    /// This is allocation only; do not treat zero-filled packed siblings
    /// as source content or publish them before the real upload fence.
    ///
    /// SAFETY: same queue/state exclusion as map_more. Newly admitted
    /// groups are zero-initialized, not yet filled from title memory.
    unsafe fn admit_held(
        &mut self, host: &crate::host::Host, texture: &ID3D12Resource,
        desc: &D3D12_RESOURCE_DESC, held: u32, room: bool,
        before: D3D12_RESOURCE_STATES,
    ) -> Option<bool> {
        let held = held & self.every_level();
        let named = held | self.card;
        let wanted = if room { named } else { self.card };
        // SAFETY: inherited caller's queue and state exclusion.
        let mapped = unsafe { self.map_more(host, texture, desc, wanted, before) }?;
        Some(named & !mapped != 0)
    }

    /// Add only the missing groups. Allocation is transactional until the
    /// queue is touched; a failed admission returns every tentative run.
    /// Once the queue is touched initialize() either finishes its fence
    /// or terminates the run, so no in-flight leases can be reused.
    ///
    /// SAFETY: the caller must own a live resource matching these groups
    /// and desc, with `before` the actual state, and exclude concurrent
    /// GPU access until the mapper fence. The caller either has an
    /// unpublished resource or holds the submission lock before preludes.
    unsafe fn map_more(&mut self, host: &crate::host::Host, texture: &ID3D12Resource,
        desc: &D3D12_RESOURCE_DESC, wanted: u32, before: D3D12_RESOURCE_STATES) -> Option<u32> {
        let additions = missing_groups(&self.groups, self.mapped, wanted)
            .map(|group| Some((group, take(&host.device, group.tiles)?)))
            .collect::<Option<Vec<_>>>()?;
        if additions.is_empty() {
            return Some(self.mapped);
        }
        initialize(host, texture, desc, &additions, before)?;
        let mapped = additions.iter().fold(self.mapped, |bits, (group, _)| bits | group.mask());
        self.leases.extend(additions.into_iter().map(|(_, lease)| lease));
        self.mapped = mapped;
        Some(mapped)
    }
}

/// Adapt cool_fox uploaded_on/card_wrote_on admission to the synchronous renderer:
/// initialize missing groups in a fenced mapper pass, not in each title
/// list. Recording or resetting a list cannot clear another list's write.
///
/// SAFETY: the submission lock is held before any upload or state
/// prelude; all earlier GPU work has completed.
pub(crate) unsafe fn prepare_backing(
    host: &crate::host::Host,
    copy: &crate::resource::HostTexture,
    levels: u32,
) -> bool {
    let Some(mapping) = &copy.tiled else { return true };
    let mut state = mapping.lock().unwrap_or_else(|e| e.into_inner());
    let requested = group_mask(&state.groups, levels);
    let missing = requested & !state.mapped;
    if missing == 0 { return true; }
    state.claim(missing);
    // SAFETY: the copy is live; the mapper restores its actual state
    // and completes its fence before returning.
    let desc = unsafe { copy.texture.GetDesc() };
    let before = D3D12_RESOURCE_STATES(copy.state.load(Relaxed) as i32);
    // SAFETY: the submission lock excludes GPU access, and before is
    // the resource's actual state from the last completed submission.
    let result = unsafe { state.map_more(host, &copy.texture, &desc, missing, before) };
    state.unclaim(missing);
    result.is_some_and(|mapped| mapped & requested == requested)
}

/// Adapt cool_fox's one-second waiting-source retry. Only ordinary
/// nonmandatory heap refreshes that have not been detiled or recorded
/// may use this gate. Required copies and GPU writes remain mandatory.
/// Returns Some(true) when new backing needs a fresh source read;
/// None defers the refresh, preserving its dirty state.
fn retry_due(now: std::time::Instant, deadline: Option<std::time::Instant>) -> bool {
    deadline.is_none_or(|until| now >= until)
}

/// SAFETY: same submission-lock and state-prelude exclusion as
/// prepare_backing. No GPU command has been recorded for this source.
pub(crate) unsafe fn prepare_optional_source(
    host: &crate::host::Host,
    copy: &crate::resource::HostTexture,
    levels: u32,
) -> Option<bool> {
    let Some(mapping) = &copy.tiled else { return Some(false) };
    let now = std::time::Instant::now();
    {
        let mut state = mapping.lock().unwrap_or_else(|e| e.into_inner());
        if group_mask(&state.groups, levels) & !state.mapped == 0 {
            state.source_retry_at = None;
            return Some(false);
        }
        if !retry_due(now, state.source_retry_at) {
            return None;
        }
        // Bound pressure and allocation-failure retries alike, without
        // consuming a tracker bit or keeping a stale upload buffer.
        state.source_retry_at = Some(now + std::time::Duration::from_secs(1));
    }
    if crate::shadow::memory_pressured() || crate::host::video_memory_pressured() {
        return None;
    }
    // SAFETY: the caller holds the submission lock and no prelude ran.
    if !unsafe { prepare_backing(host, copy, levels) } {
        return None;
    }
    mapping.lock().unwrap_or_else(|e| e.into_inner()).source_retry_at = None;
    Some(true)
}

/// The resource itself retains the shared mapping state through
/// SetPrivateDataInterface, including any subsequently added groups.
/// This is not tied to HostTexture's Arc: raw COM owners in slots and
/// retired descriptors can outlive it. No cycle: the state owns heaps.
#[repr(C)]
struct Owner {
    vtable: *const IUnknown_Vtbl,
    refs: AtomicU32,
    _mapping: Arc<Mutex<Mapping>>,
}

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return HRESULT(0x80004003u32 as i32);
    }
    // SAFETY: COM provides a writable interface output.
    unsafe { out.write(std::ptr::null_mut()) };
    if iid.is_null() || unsafe { *iid } != IUnknown::IID {
        return HRESULT(0x80004002u32 as i32);
    }
    // SAFETY: this is a live Owner's canonical IUnknown.
    unsafe {
        add_ref(this);
        out.write(this);
    }
    HRESULT(0)
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: IUnknown methods are called on a live Owner.
    let owner = unsafe { &*this.cast::<Owner>() };
    owner.refs.fetch_add(1, Relaxed) + 1
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: the caller holds a reference.
    let owner = unsafe { &*this.cast::<Owner>() };
    let old = owner.refs.fetch_sub(1, AcqRel);
    if old == 1 {
        // SAFETY: the last reference owns the original Box.
        drop(unsafe { Box::from_raw(this.cast::<Owner>()) });
        0
    } else {
        old - 1
    }
}

static VTABLE: IUnknown_Vtbl = IUnknown_Vtbl { QueryInterface: query, AddRef: add_ref, Release: release };
const OWNER_KEY: GUID = GUID::from_u128(0xf36b88b5_a067_4c2d_9fec_f11eaa6b0a72);

fn attach(texture: &ID3D12Resource, mapping: Arc<Mutex<Mapping>>) -> windows::core::Result<()> {
    let raw = Box::into_raw(Box::new(Owner { vtable: &VTABLE, refs: AtomicU32::new(1), _mapping: mapping }));
    // SAFETY: Owner begins with a valid IUnknown vtable and owns one ref.
    let unknown = unsafe { IUnknown::from_raw(raw.cast()) };
    // SAFETY: the resource stores its own reference to the owner.
    unsafe { texture.SetPrivateDataInterface(&OWNER_KEY, &unknown) }
}

fn tier2(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: the output has the required size and alignment.
    unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut options).cast(), size_of_val(&options) as u32)
    }.is_ok() && options.TiledResourcesTier.0 >= 2
}

/// Observation is separately opt-in: checking title page commitment
/// adds CPU work and is not yet used as an allocation decision.
pub(crate) fn judgement_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TILED_TEXTURE_POOL").as_deref() == Ok("1")
        && std::env::var("VICTORY_TILED_MIP_JUDGEMENT").as_deref() == Ok("1"))
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TILED_TEXTURE_POOL").as_deref() == Ok("1")
        && crate::host::host().is_some_and(|host| tier2(&host.device)))
}

fn candidate(d: &D3D12_RESOURCE_DESC) -> bool {
    // Deliberately limited to single-plane RGBA8, one array slice,
    // ordinary sampled textures. Every GPU-written/castable fallback
    // remains legal because this initial path maps the whole chain.
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.DepthOrArraySize == 1 && (2..=16).contains(&d.MipLevels)
        && d.SampleDesc.Count == 1 && d.Flags == D3D12_RESOURCE_FLAG_NONE
        && matches!(d.Format.0, 27..=29)
}

fn query_groups(device: &ID3D12Device, texture: &ID3D12Resource, levels: u32) -> Option<Vec<Group>> {
    let mut total = 0;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut count = levels;
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); levels as usize];
    // SAFETY: all outputs and the tiling array are live and sized.
    unsafe {
        device.GetResourceTiling(texture, Some(&mut total), Some(&mut packed), None,
            Some(&mut count), 0, tilings.as_mut_ptr());
    }
    let standard = u32::from(packed.NumStandardMips);
    if standard > count || standard > levels || standard + u32::from(packed.NumPackedMips) != levels {
        return None;
    }
    let sizes = tilings[..standard as usize].iter().map(|t| {
        if t.StartTileIndexInOverallResource == D3D12_PACKED_TILE {
            return None;
        }
        t.WidthInTiles.checked_mul(u32::from(t.HeightInTiles))?
            .checked_mul(u32::from(t.DepthInTiles))
    }).collect::<Option<Vec<_>>>()?;
    groups(&sizes, u32::from(packed.NumPackedMips), packed.NumTilesForPackedMips, total)
}

/// A separate fenced initialization queue. The caller must exclude other
/// GPU uses until mapping and zero initialization have completed.
/// Never use the title submission mutex recursively.
struct Mapper {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: HANDLE,
    value: u64,
}

// SAFETY: the queue and fence are free-threaded, and the event is used
// only while holding MAPPER's mutex.
unsafe impl Send for Mapper {}

fn mapper(device: &ID3D12Device) -> Option<&'static Mutex<Mapper>> {
    static MAPPER: OnceLock<Option<Mutex<Mapper>>> = OnceLock::new();
    MAPPER.get_or_init(|| {
        // SAFETY: plain D3D12 object and event creation.
        unsafe {
            let queue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default()
            }).ok()?;
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(Mutex::new(Mapper { queue, fence, event, value: 0 }))
        }
    }).as_ref()
}

fn initialize(host: &crate::host::Host, texture: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC,
    mappings: &[(Group, Lease)], before: D3D12_RESOURCE_STATES) -> Option<()> {
    let mut footprints = Vec::with_capacity(desc.MipLevels as usize);
    let mut largest = 0u64;
    // Whole standard MIPs or every subresource in an added packed tail.
    // Do not zero levels which already held mapped texture contents.
    let added = mappings.iter().fold(0, |bits, (group, _)| bits | group.mask());
    if added == 0 { return Some(()); }
    for level in (0..u32::from(desc.MipLevels)).filter(|&level| added & (1 << level) != 0) {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: one footprint output for one valid subresource.
        unsafe {
            host.device.GetCopyableFootprints(desc, level, 1, 0,
                Some(&mut footprint), None, None, Some(&mut bytes));
        }
        if bytes == 0 || bytes > MAX_BYTES {
            return None;
        }
        largest = largest.max(bytes);
        footprints.push((level, footprint));
    }
    let (zero, data) = host.upload_buffer(largest)?;
    // SAFETY: the upload mapping is at least largest bytes.
    unsafe { data.write_bytes(0, largest as usize) };
    let (allocator, list): (ID3D12CommandAllocator, ID3D12GraphicsCommandList) = unsafe {
        // SAFETY: ordinary direct-list creation.
        let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
        let list = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
        (allocator, list)
    };
    let alias = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: ManuallyDrop::new(None),
                // SAFETY: borrowed pointer, kept alive through the fence.
                pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
            }),
        },
    };
    // SAFETY: the resources and command allocator are live. The mapping
    // commands will be placed on this queue before the closed copy list.
    unsafe {
        list.ResourceBarrier(&[alias]);
        if before != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[crate::host::transition(texture,
                before, D3D12_RESOURCE_STATE_COPY_DEST)]);
        }
        for &(level, footprint) in &footprints {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&zero),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        if before != D3D12_RESOURCE_STATE_COPY_DEST {
            list.ResourceBarrier(&[crate::host::transition(texture,
                D3D12_RESOURCE_STATE_COPY_DEST, before)]);
        }
        list.Close().ok()?;
    }
    let command: ID3D12CommandList = list.cast().ok()?;
    let mut mapper = mapper(&host.device)?.lock().unwrap_or_else(|e| e.into_inner());
    let value = mapper.value.checked_add(1)?;
    // After the first queue mapping, failure is fatal: no lease or upload
    // may return to its pool until the initialization fence has completed.
    for (group, lease) in mappings {
        let at = D3D12_TILED_RESOURCE_COORDINATE { Subresource: group.first, ..Default::default() };
        let size = D3D12_TILE_REGION_SIZE { NumTiles: group.tiles, ..Default::default() };
        let starts: Vec<_> = lease.runs.iter().map(|r| r.0).collect();
        let counts: Vec<_> = lease.runs.iter().map(|r| r.1).collect();
        let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; lease.runs.len()];
        // SAFETY: the whole standard level or complete packed tail maps
        // to disjoint leased heap runs with the same tile count.
        unsafe {
            mapper.queue.UpdateTileMappings(texture, 1, Some(&at), Some(&size),
                &lease.heap.heap, starts.len() as u32, Some(flags.as_ptr()),
                Some(starts.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
        }
    }
    // SAFETY: all references, mappings and uploads are held to completion.
    unsafe {
        mapper.queue.ExecuteCommandLists(&[Some(command)]);
        if let Err(e) = mapper.queue.Signal(&mapper.fence, value)
            .and_then(|()| mapper.fence.SetEventOnCompletion(value, mapper.event))
        {
            crate::host::fail(format_args!("tile initialization fence failed: {e}"));
        }
        if WaitForSingleObject(mapper.event, INFINITE) != WAIT_OBJECT_0
            || mapper.fence.GetCompletedValue() < value
            || host.device.GetDeviceRemovedReason().is_err()
        {
            crate::host::fail(format_args!("tile initialization did not complete"));
        }
    }
    mapper.value = value;
    drop(allocator);
    Some(())
}

/// Create an unpublished reserved resource with no physical groups
/// yet assigned. Keep the same format/tier/budget admission policy.
fn reserve(host: &crate::host::Host, desc: &D3D12_RESOURCE_DESC,
    formats: &[DXGI_FORMAT]) -> Option<(ID3D12Resource, D3D12_RESOURCE_DESC, Arc<Mutex<Mapping>>)> {
    if !candidate(desc) || !tier2(&host.device) || pressured() {
        return None;
    }
    let mut desc = *desc;
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description is valid for a reserved 2D resource.
    unsafe {
        if formats.is_empty() {
            host.device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture).ok()?;
        } else {
            let device = host.device.cast::<ID3D12Device10>().ok()?;
            device.CreateReservedResource2(&desc, D3D12_BARRIER_LAYOUT_COMMON, None,
                None::<&ID3D12ProtectedResourceSession>, Some(formats), &mut texture).ok()?;
        }
    }
    let texture = texture?;
    let groups = query_groups(&host.device, &texture, u32::from(desc.MipLevels))?;
    let bytes = groups.iter().map(|g| u64::from(g.tiles) * TILE).sum::<u64>();
    if !(MIN_BYTES..=MAX_BYTES).contains(&bytes) {
        return None;
    }
    let mapping = Arc::new(Mutex::new(Mapping::new(groups)));
    // Attach before allocating; a failed attachment cannot leave any
    // queued mappings without a resource-lifetime owner.
    attach(&texture, mapping.clone()).ok()?;
    Some((texture, desc, mapping))
}

/// Reference cool_fox show_from, keeping the original 2D/2D-array
/// range and LOD adjustment semantics.
pub(crate) fn show_from(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, first: u32, levels: u32) {
    if first >= levels {
        return;
    }
    let clamp = |most: &mut u32, mips: &mut u32, min_lod: &mut f32| {
        if first <= *most {
            return;
        }
        let fewer = first - *most;
        *most = first;
        if *mips != u32::MAX {
            *mips = mips.saturating_sub(fewer).clamp(1, levels - first);
        }
        if *min_lod > 0.0 {
            *min_lod = (*min_lod - fewer as f32).max(0.0);
        }
    };
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: `ViewDimension` selects `Texture2D`.
            let view = unsafe { &mut desc.Anonymous.Texture2D };
            clamp(&mut view.MostDetailedMip, &mut view.MipLevels, &mut view.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: `ViewDimension` selects `Texture2DArray`.
            let view = unsafe { &mut desc.Anonymous.Texture2DArray };
            clamp(&mut view.MostDetailedMip, &mut view.MipLevels, &mut view.ResourceMinLODClamp);
        }
        _ => {}
    }
}

/// A sparse SRV must not span holes or expand beyond the title's
/// requested MIP range. Start with the most detailed visible level in
/// that range, then end at the first hole. None means a null view.
fn visible_run(mask: u32, most: u32, mips: u32, levels: u32) -> Option<(u32, u32)> {
    let levels = levels.min(32);
    let end = most.saturating_add(mips).min(levels);
    if most >= end { return None; }
    let first = (most..end).find(|&i| mask & (1u32 << i) != 0)?;
    let until = (first..end).find(|&i| mask & (1u32 << i) == 0).unwrap_or(end);
    Some((first, until - first))
}

/// Keep the reference's LOD adjustment, then cap its range at the end
/// of the safe run. The full-chain case preserves the original view.
pub(crate) fn clamp_visible(
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, mask: u32, levels: u32,
) -> bool {
    let all = if levels >= 32 { u32::MAX } else { (1u32 << levels) - 1 };
    if levels != 0 && mask & all == all { return true; }
    let (most, mips) = match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: the dimension selects this union member.
            let view = unsafe { desc.Anonymous.Texture2D };
            (view.MostDetailedMip, view.MipLevels)
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: the dimension selects this union member.
            let view = unsafe { desc.Anonymous.Texture2DArray };
            (view.MostDetailedMip, view.MipLevels)
        }
        _ => return false,
    };
    let Some((first, count)) = visible_run(mask, most, mips, levels) else { return false };
    show_from(desc, first, levels);
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            desc.Anonymous.Texture2D.MipLevels = count;
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            desc.Anonymous.Texture2DArray.MipLevels = count;
        }
        _ => unreachable!(),
    }
    true
}

/// Production remains fully mapped before first publication. The
/// incremental allocator is shared with the isolated sparse GPU test;
/// live partial mapping is NOT enabled until descriptor/write tracking
/// and readback verification are complete.
pub(crate) fn make(host: &crate::host::Host, desc: &D3D12_RESOURCE_DESC,
    formats: &[DXGI_FORMAT]) -> Option<(ID3D12Resource, u64, Arc<Mutex<Mapping>>)> {
    let (texture, desc, mapping) = reserve(host, desc, formats)?;
    let mut state = mapping.lock().unwrap_or_else(|e| e.into_inner());
    let all = state.every_level();
    // SAFETY: reserve returned an unpublished COMMON-state resource.
    let waiting = unsafe {
        state.admit_held(host, &texture, &desc, all, true, D3D12_RESOURCE_STATE_COMMON)
    }?;
    if waiting || state.mapped != all {
        return None;
    }
    // Production deliberately treats the completely zero-initialized
    // chain as visible, matching the committed-resource baseline. The
    // partial admission path itself must never publish source bytes.
    state.held = all;
    let bytes = state.given_bytes();
    drop(state);
    Some((texture, bytes, mapping))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_and_packed_levels_are_indivisible_mapping_groups() {
        let g = groups(&[16, 4, 1], 4, 2, 23).unwrap();
        assert_eq!(selected(&g, 1 << 1).collect::<Vec<_>>(), [Group { first: 1, levels: 1, tiles: 4 }]);
        assert_eq!(selected(&g, 1 << 5).collect::<Vec<_>>(), [Group { first: 3, levels: 4, tiles: 2 }]);
        assert_eq!(selected(&g, 0).count(), 0);
        assert_eq!(selected(&g, u32::MAX).map(|g| g.tiles).sum::<u32>(), 23);
        assert!(groups(&[16, 4, 1], 4, 2, 22).is_none());
        assert!(groups(&[0], 1, 1, 1).is_none());
        assert!(groups(&[1], 0, 1, 2).is_none());
    }

    #[test]
    fn incremental_mapping_expands_packed_tail_and_skips_existing_groups() {
        let groups = groups(&[16, 4, 1], 4, 2, 23).unwrap();
        let tail = group_mask(&groups, 1 << 5);
        assert_eq!(tail, 0b1111000);
        assert_eq!(group_mask(&groups, 1 << 1), 1 << 1);
        assert_eq!(group_mask(&groups, u32::MAX), 0b1111111);
        assert_eq!(missing_groups(&groups, tail, 1 << 4).count(), 0);
        assert_eq!(missing_groups(&groups, tail, tail | 1).collect::<Vec<_>>(),
            [Group { first: 0, levels: 1, tiles: 16 }]);
        assert_eq!(missing_groups(&groups, 0, 0).count(), 0);
    }

    #[test]
    fn claims_recover_and_only_backed_gpu_levels_are_published() {
        let mut state = Mapping::new(groups(&[16, 4, 1], 4, 2, 23).unwrap());
        let tail = 0b1111000;
        assert_eq!(state.claim(1 << 5), tail);
        assert_eq!(state.claim(1 << 6), 0);
        // Recording was discarded or allocation failed.
        state.unclaim(1 << 5);
        assert_eq!(state.claim(1 << 6), tail);
        assert!(!state.card_wrote(1 << 6));
        assert_eq!(state.card, 0);
        assert_eq!(state.first_shown(), None);
        assert_eq!(state.claim(1 << 6), tail);
        state.mapped = tail;
        assert!(state.card_wrote(1 << 6));
        assert_eq!(state.card, 1 << 6);
        assert_eq!(state.claimed, 0);
        assert_eq!(state.first_shown(), Some(6));
        assert_eq!(state.claim(1 << 4), 0);
        state.held = 1 << 3;
        assert_eq!(state.first_shown(), Some(3));
        assert!(state.observe_held(1 << 6));
        assert!(!state.observe_held(1 << 6));
        assert_eq!(state.first_shown(), Some(3), "observation cannot change published levels");
    }

    #[test]
    fn waiting_source_retries_only_after_the_interval() {
        let now = std::time::Instant::now();
        let until = now + std::time::Duration::from_secs(1);
        assert!(retry_due(now, None));
        assert!(!retry_due(now, Some(until)));
        assert!(retry_due(until, Some(until)));
    }

    #[test]
    fn visible_runs_never_span_holes_or_leave_the_requested_range() {
        assert_eq!(visible_run(0b100110, 0, u32::MAX, 6), Some((1, 2)));
        assert_eq!(visible_run(0b100110, 3, 3, 6), Some((5, 1)));
        assert_eq!(visible_run(0b100110, 3, 2, 6), None);
        assert_eq!(visible_run(u32::MAX, 31, u32::MAX, 32), Some((31, 1)));
        assert_eq!(visible_run(0, 0, 4, 4), None);
        assert_eq!(visible_run(0b1111, 0, 0, 4), None);

        let mut view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0, MipLevels: u32::MAX,
                    ResourceMinLODClamp: 2.0, PlaneSlice: 0,
                },
            },
            ..Default::default()
        };
        assert!(clamp_visible(&mut view, 0b100110, 6));
        // SAFETY: the dimension selects Texture2D.
        let v = unsafe { view.Anonymous.Texture2D };
        assert_eq!((v.MostDetailedMip, v.MipLevels), (1, 2));
        assert_eq!(v.ResourceMinLODClamp, 1.0);
        assert!(!clamp_visible(&mut view, 0b100000, 6));
    }

    #[test]
    fn a_filled_hole_changes_visibility_even_with_the_same_first_mip() {
        let mut state = Mapping::new(groups(&[16, 4, 1], 4, 2, 23).unwrap());
        state.mapped = state.every_level();
        assert!(state.source_wrote(1));
        assert!(state.source_wrote(1 << 2));
        assert_eq!(state.first_shown(), Some(0));
        assert!(state.card_wrote(1 << 1));
        assert_eq!(state.visible_mask(), 0b111);
        assert_eq!(state.first_shown(), Some(0));
    }

    #[test]
    fn source_completion_publishes_only_backed_mips_and_preserves_card_mask() {
        let mut state = Mapping::new(groups(&[16, 4, 1], 4, 2, 23).unwrap());
        assert!(!state.source_wrote(1 << 6));
        assert_eq!(state.source_uploaded, 0);
        assert_eq!(state.first_shown(), None);
        state.mapped = 0b1111000;
        assert!(state.card_wrote(1 << 6));
        assert!(state.source_wrote((1 << 5) | 1));
        assert_eq!(state.source_uploaded, 1 << 5);
        assert_eq!(state.card, 1 << 6);
        assert_eq!(state.first_shown(), Some(5));
        assert!(!state.source_wrote(1 << 5));
    }

    #[test]
    fn shown_srv_preserves_reference_range_and_lod_clamp() {
        let mut view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 1, MipLevels: 7,
                    ResourceMinLODClamp: 4.5, PlaneSlice: 0,
                },
            },
            ..Default::default()
        };
        show_from(&mut view, 3, 8);
        // SAFETY: Texture2D is the selected union member.
        let v = unsafe { view.Anonymous.Texture2D };
        assert_eq!((v.MostDetailedMip, v.MipLevels), (3, 5));
        assert_eq!(v.ResourceMinLODClamp, 2.5);
        let mut array = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0, MipLevels: u32::MAX,
                    FirstArraySlice: 2, ArraySize: 3, PlaneSlice: 0,
                    ResourceMinLODClamp: 1.0,
                },
            },
            ..Default::default()
        };
        show_from(&mut array, 4, 8);
        // SAFETY: Texture2DArray is the selected union member.
        let v = unsafe { array.Anonymous.Texture2DArray };
        assert_eq!((v.MostDetailedMip, v.MipLevels), (4, u32::MAX));
        assert_eq!((v.FirstArraySlice, v.ArraySize), (2, 3));
        assert_eq!(v.ResourceMinLODClamp, 0.0);
    }

    #[test]
    fn tile_runs_reuse_holes_and_return_to_one_free_range() {
        let mut free = vec![(0, 4), (8, 4), (20, 12)];
        let initial = free.clone();
        assert!(cut(&mut free, 21).is_none());
        assert_eq!(free, initial);
        let first = cut(&mut free, 6).unwrap();
        assert_eq!(first, [(0, 4), (8, 2)]);
        let second = cut(&mut free, 14).unwrap();
        assert!(free.is_empty());
        join(&mut free, &first);
        join(&mut free, &second);
        assert_eq!(free, initial);
        let mut full = vec![(0, HEAP_TILES)];
        let a = cut(&mut full, 32).unwrap();
        let b = cut(&mut full, 64).unwrap();
        join(&mut full, &b);
        join(&mut full, &a);
        assert_eq!(full, [(0, HEAP_TILES)]);
    }

    #[test]
    #[ignore = "requires a tiled-resource tier-2 D3D12 adapter; run alone"]
    fn reserved_texture_zero_initialization_and_com_owner_lifetime() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        let host = crate::host::host().expect("D3D12 device");
        assert!(tier2(&host.device));
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024, Height: 1024, DepthOrArraySize: 1, MipLevels: 8,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let before = usage().0;
        let (texture, bytes, mapping) = make(host, &desc, &[]).expect("reserved texture");
        assert_eq!(mapping.lock().unwrap().first_shown(), Some(0));
        drop(mapping);
        assert_eq!(usage().0, before + bytes);
        let retained = texture.clone();
        drop(texture);
        assert_eq!(usage().0, before + bytes);
        // A recycled heap must not leak old texture contents.
        assert_eq!(read_texels(host, &retained, &desc, [0, 7]), [0, 0]);
        drop(retained);
        assert_eq!(usage().0, before);
    }

    #[test]
    #[ignore = "requires a tiled-resource tier-2 D3D12 adapter; run alone"]
    fn sparse_group_expansion_preserves_existing_texels_and_leases() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        let host = crate::host::host().expect("D3D12 device");
        assert!(tier2(&host.device));
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024, Height: 1024, DepthOrArraySize: 1, MipLevels: 8,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let before = usage().0;
        let (texture, desc, mapping) = reserve(host, &desc, &[]).expect("reserved texture");
        let (tail, full_bytes) = {
            let map = mapping.lock().unwrap();
            let tail = group_mask(&map.groups, 1 << 7);
            assert_ne!(tail, map.every_level());
            (tail, map.groups.iter().map(|g| u64::from(g.tiles) * TILE).sum())
        };
        let copy = Arc::new(crate::resource::HostTexture::new_with_size(
            texture.clone(), 0, Vec::new(), Some((full_bytes, mapping.clone())),
        ));
        let abandoned = record_texel(host, &copy, &desc, 0, 0xdeadbeef, false);
        abandoned.reset();
        drop(abandoned);
        assert_eq!(mapping.lock().unwrap().mapped, 0);
        assert_eq!(usage().0, before);
        // Same new group, reverse recording order: initialize exactly once.
        let marker = 0x11223344;
        let first = record_texel(host, &copy, &desc, 7, marker, false);
        let second = record_texel(host, &copy, &desc, 7, 0xaabbccdd, false);
        host.execute(&[&second, &first]);
        drop(first);
        drop(second);
        let tail_bytes = {
            let map = mapping.lock().unwrap();
            assert_eq!(map.mapped, tail);
            assert_eq!(map.card, 1 << 7);
            assert_eq!(map.claimed, 0);
            assert_eq!(map.first_shown(), Some(7));
            assert_eq!(usage().0, before + map.given_bytes());
            map.given_bytes()
        };
        let fine = 0x55667788;
        let list = record_texel(host, &copy, &desc, 0, fine, false);
        host.execute(&[&list]);
        drop(list);
        {
            let map = mapping.lock().unwrap();
            assert_eq!(map.mapped, tail | 1);
            assert_eq!(map.card, (1 << 7) | 1);
            assert_eq!(map.claimed, 0);
            assert_eq!(map.first_shown(), Some(0));
            assert!(map.given_bytes() > tail_bytes);
            assert_eq!(usage().0, before + map.given_bytes());
        }
        assert_eq!(read_texels(host, &texture, &desc, [0, 7]), [fine, marker]);
        let untouched = tail & !(1 << 7);
        if untouched != 0 {
            assert_eq!(read_texels(host, &texture, &desc, [untouched.trailing_zeros(), 7]), [0, marker]);
        }
        let retained = texture.clone();
        drop(copy);
        drop(mapping);
        drop(texture);
        assert!(usage().0 > before);
        drop(retained);
        assert_eq!(usage().0, before);
    }

    #[test]
    #[ignore = "requires a tiled-resource tier-2 D3D12 adapter; run alone"]
    fn held_mip_admission_waits_for_room_and_keeps_existing_backing() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        let host = crate::host::host().expect("D3D12 device");
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024, Height: 1024, DepthOrArraySize: 1, MipLevels: 8,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let before = usage().0;
        let (texture, desc, mapping) = reserve(host, &desc, &[]).expect("reserved texture");
        {
            let mut state = mapping.lock().unwrap();
            let tail = group_mask(&state.groups, 1 << 7);
            let all = state.every_level();
            assert_ne!(tail, all);
            // SAFETY: this resource is unpublished, in COMMON, and no
            // other list accesses it. Each mapper pass waits for its fence.
            assert_eq!(unsafe { state.admit_held(host, &texture, &desc,
                1 << 7, false, D3D12_RESOURCE_STATE_COMMON) }, Some(true));
            assert_eq!(state.mapped, 0);
            assert_eq!(usage().0, before);
            // SAFETY: the resource remains unpublished, in COMMON.
            assert_eq!(unsafe { state.admit_held(host, &texture, &desc,
                1 << 7, true, D3D12_RESOURCE_STATE_COMMON) }, Some(false));
            assert_eq!(state.mapped, tail);
            assert_eq!(state.first_shown(), None);
            // Simulate completion of a source upload of the coarse MIP.
            assert!(state.source_wrote(1 << 7));
            assert_eq!(state.first_shown(), Some(7));
            let tail_bytes = state.given_bytes();
            // SAFETY: as above; a new held level waits without discarding
            // the tail or exposing its not-yet-uploaded siblings.
            assert_eq!(unsafe { state.admit_held(host, &texture, &desc,
                all, false, D3D12_RESOURCE_STATE_COMMON) }, Some(true));
            assert_eq!(state.given_bytes(), tail_bytes);
            assert_eq!(state.visible_mask(), 1 << 7);
            // SAFETY: the resource remains unpublished, in COMMON.
            assert_eq!(unsafe { state.admit_held(host, &texture, &desc,
                all, true, D3D12_RESOURCE_STATE_COMMON) }, Some(false));
            assert_eq!(state.mapped, all);
            assert_eq!(state.first_shown(), Some(7));
            assert!(state.source_wrote(all));
            assert_eq!(state.first_shown(), Some(0));
            let bytes = state.given_bytes();
            assert!(bytes > tail_bytes);
            // SAFETY: as above. The reference keeps backing and this
            // allocation-only pass does not hide completed source levels.
            assert_eq!(unsafe { state.admit_held(host, &texture, &desc,
                1 << 7, true, D3D12_RESOURCE_STATE_COMMON) }, Some(false));
            assert_eq!(state.mapped, all);
            assert_eq!(state.visible_mask(), all);
            assert_eq!(state.given_bytes(), bytes);
        }
        assert_eq!(read_texels(host, &texture, &desc, [0, 7]), [0, 0]);
        drop(mapping);
        drop(texture);
        assert_eq!(usage().0, before);
    }

    #[test]
    #[ignore = "requires a tiled-resource tier-2 D3D12 adapter; run alone"]
    fn sparse_source_upload_is_published_only_after_the_copy_fence() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        let host = crate::host::host().expect("D3D12 device");
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024, Height: 1024, DepthOrArraySize: 1, MipLevels: 8,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let before = usage().0;
        let (texture, desc, mapping) = reserve(host, &desc, &[]).expect("reserved texture");
        let full_bytes = mapping.lock().unwrap().groups.iter()
            .map(|g| u64::from(g.tiles) * TILE).sum();
        let copy = Arc::new(crate::resource::HostTexture::new_with_size(
            texture.clone(), 0, Vec::new(), Some((full_bytes, mapping.clone())),
        ));
        let abandoned = record_texel(host, &copy, &desc, 0, 0xdeadbeef, true);
        abandoned.reset();
        drop(abandoned);
        let coarse = record_texel(host, &copy, &desc, 7, 0x11223344, true);
        {
            let state = mapping.lock().unwrap();
            assert_eq!(state.first_shown(), None);
            assert_eq!(state.source_uploaded, 0);
            assert_eq!(state.mapped, 0);
        }
        host.execute(&[&coarse]);
        drop(coarse);
        {
            let state = mapping.lock().unwrap();
            assert_eq!(state.source_uploaded, 1 << 7);
            assert_eq!(state.card, 0);
            assert_eq!(state.first_shown(), Some(7));
        }
        let fine = record_texel(host, &copy, &desc, 0, 0x55667788, true);
        host.execute(&[&fine]);
        drop(fine);
        {
            let state = mapping.lock().unwrap();
            assert_eq!(state.source_uploaded, (1 << 7) | 1);
            assert_eq!(state.card, 0);
            assert_eq!(state.first_shown(), Some(0));
            assert_eq!(usage().0, before + state.given_bytes());
        }
        assert_eq!(read_texels(host, &texture, &desc, [0, 7]), [0x55667788, 0x11223344]);
        drop(copy);
        drop(mapping);
        drop(texture);
        assert_eq!(usage().0, before);
    }

    fn record_texel(host: &crate::host::Host, copy: &Arc<crate::resource::HostTexture>,
        desc: &D3D12_RESOURCE_DESC, level: u32, value: u32, source: bool) -> crate::host::HostList {
        let texture = &copy.texture;
        let (upload, data) = host.upload_buffer(512).unwrap();
        // SAFETY: mapped upload contains at least one RGBA8 texel.
        unsafe { data.cast::<u32>().write_unaligned(value) };
        let list = host.new_list().unwrap();
        list.keep(&upload);
        if source {
            list.defer_tile_source(copy, 1 << level);
        } else {
            list.defer_tile_write(copy, 1 << level);
        }
        // SAFETY: the list retains its destination and upload; submission
        // admits missing backing before either command runs.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(texture,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: desc.Format, Width: 1, Height: 1, Depth: 1, RowPitch: 256,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.list.ResourceBarrier(&[crate::host::transition(texture,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
        }
        list.close();
        list
    }

    fn read_texels(host: &crate::host::Host, texture: &ID3D12Resource,
        desc: &D3D12_RESOURCE_DESC, levels: [u32; 2]) -> [u32; 2] {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let mut back: Option<ID3D12Resource> = None;
        let properties = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 1024, Height: 1,
            DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
        };
        // SAFETY: valid readback allocation and output.
        unsafe {
            host.device.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).unwrap();
        }
        let back = back.unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: live resources, a closed/fenced list below, and two
        // non-overlapping 512-byte-aligned single-texel footprints.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(texture,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (i, level) in levels.into_iter().enumerate() {
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
                };
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&back),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: i as u64 * 512,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: desc.Format, Width: 1, Height: 1, Depth: 1, RowPitch: 256,
                            },
                        },
                    },
                };
                let area = D3D12_BOX { left: 0, top: 0, front: 0, right: 1, bottom: 1, back: 1 };
                list.list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&area));
            }
            list.list.ResourceBarrier(&[crate::host::transition(texture,
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        }
        list.close();
        host.execute(&[&list]);
        let mut data = std::ptr::null_mut();
        // SAFETY: readback follows successful GPU completion.
        unsafe {
            back.Map(0, None, Some(&mut data)).unwrap();
            let values = [*data.cast::<u32>(), *data.cast::<u8>().add(512).cast::<u32>()];
            back.Unmap(0, None);
            values
        }
    }
}
