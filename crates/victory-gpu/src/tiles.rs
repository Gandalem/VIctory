//! Reserved sampled textures and a shared tile heap pool.
//!
//! Adapted from DoubleGhosty/cool_fox tiles.rs (f4b48ad): 64 KiB tiles,
//! 64 MiB heaps, free runs, standard MIP groups and atomic packed tails.
//! Back and initialize every retained level before publishing the resource.
//! The current renderer can give up whole textures and rebuild descriptors;
//! sparse live MIP mappings still require separate descriptor coordination.

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

fn selected(groups: &[Group], mask: u32) -> impl Iterator<Item = Group> + '_ {
    groups.iter().copied().filter(move |g| {
        let bits = ((1u32 << g.levels) - 1) << g.first;
        bits & mask != 0
    })
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

pub(crate) fn usage() -> (u64, u64) {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    (MAPPED_BYTES.load(Relaxed), pool.len() as u64 * MAX_BYTES)
}

/// The resource itself retains the lease through SetPrivateDataInterface.
/// This is deliberately not tied to HostTexture's Arc or a COM reference
/// count query: the eager renderer also keeps raw COM owners in slots and
/// in the retired-descriptor queue. No cycle: the owner holds only heaps.
#[repr(C)]
struct Owner {
    vtable: *const IUnknown_Vtbl,
    refs: AtomicU32,
    _leases: Vec<Lease>,
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

fn attach(texture: &ID3D12Resource, leases: Vec<Lease>) -> windows::core::Result<()> {
    let raw = Box::into_raw(Box::new(Owner { vtable: &VTABLE, refs: AtomicU32::new(1), _leases: leases }));
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

/// A separate fenced initialization queue. No title GPU command can
/// name the reserved resource until its mappings and zero initialization
/// have completed. Never use the title submission mutex recursively.
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
    mappings: &[(Group, Lease)]) -> Option<()> {
    let mut footprints = Vec::with_capacity(desc.MipLevels as usize);
    let mut largest = 0u64;
    for level in 0..u32::from(desc.MipLevels) {
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
        footprints.push(footprint);
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
        list.ResourceBarrier(&[alias, crate::host::transition(texture,
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        for (level, &footprint) in footprints.iter().enumerate() {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&zero),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level as u32 },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.ResourceBarrier(&[crate::host::transition(texture,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
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

/// A pooled reserved texture, initialized before publication. Preserve
/// the existing castable format list on Device10 or decline the path.
/// A failure before queue publication falls back to committed creation.
pub(crate) fn make(host: &crate::host::Host, desc: &D3D12_RESOURCE_DESC,
    formats: &[DXGI_FORMAT]) -> Option<(ID3D12Resource, u64)> {
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
    let all = (1u32 << desc.MipLevels) - 1;
    let mappings = selected(&groups, all).map(|group| Some((group, take(&host.device, group.tiles)?)))
        .collect::<Option<Vec<_>>>()?;
    initialize(host, &texture, &desc, &mappings)?;
    let leases = mappings.into_iter().map(|(_, lease)| lease).collect();
    // Initialization has completed. If attaching the COM owner fails,
    // this resource is unpublished and can safely fall back.
    attach(&texture, leases).ok()?;
    Some((texture, bytes))
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
        let (texture, bytes) = make(host, &desc, &[]).expect("reserved texture");
        assert_eq!(usage().0, before + bytes);
        let retained = texture.clone();
        drop(texture);
        assert_eq!(usage().0, before + bytes);
        // Read a texel from the first and packed-tail MIP after the
        // zero-initialization fence. A recycled heap must not leak old
        // texture contents.
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
            list.list.ResourceBarrier(&[crate::host::transition(&retained,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (i, level) in [0, 7].into_iter().enumerate() {
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&retained),
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
            list.list.ResourceBarrier(&[crate::host::transition(&retained,
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        }
        list.close();
        host.execute(&[&list]);
        let mut data = std::ptr::null_mut();
        // SAFETY: readback follows successful GPU completion.
        unsafe {
            back.Map(0, None, Some(&mut data)).unwrap();
            assert_eq!(*data.cast::<u32>(), 0);
            assert_eq!(*data.cast::<u8>().add(512).cast::<u32>(), 0);
            back.Unmap(0, None);
        }
        drop(list);
        drop(retained);
        assert_eq!(usage().0, before);
    }
}
