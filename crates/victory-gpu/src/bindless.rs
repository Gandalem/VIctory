//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). Grand Theft Auto VI is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize, u32),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
#[derive(Default)]
pub(crate) struct Viewed {
    inner: Mutex<Inner>,
    ahead_queued: AtomicBool,
    ahead_cursor: Mutex<Option<usize>>,
}

type Range = (usize, u64, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

#[derive(Default)]
struct Inner {
    slots: HashMap<u32, Seen>,
    ranges: BTreeMap<Range, (u32, Arc<Shadow>)>,
    textures: BTreeMap<usize, u32>,
    /// Reference counts of restricted SRV masks; full views are implicit
    /// in the difference from the total texture count.
    texture_masks: BTreeMap<usize, BTreeMap<u32, u32>>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4).
    hulls: HashMap<(usize, bool), Hull>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: HashMap<(usize, bool), Ends>,
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    fn requested_mips(&self, resource: usize) -> u32 {
        let Some(masks) = self.texture_masks.get(&resource) else { return u32::MAX };
        if masks.values().sum::<u32>() != self.textures.get(&resource).copied().unwrap_or(0) {
            return u32::MAX;
        }
        masks.keys().fold(0, |mask, &m| mask | m)
    }

    fn count(&mut self, seen: &Seen, add: bool) {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry(key).or_insert_with(|| (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let ends = self
                        .ends
                        .entry((key.0, key.3))
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let hull = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => Some((shadow.clone(), from, to - from, key.3, shadow.dirty_bit())),
                        _ => None,
                    };
                    match hull {
                        Some(hull) => {
                            self.hulls.insert((key.0, key.3), hull);
                        }
                        None => {
                            self.ends.remove(&(key.0, key.3));
                            self.hulls.remove(&(key.0, key.3));
                        }
                    }
                }
            }
            Seen::Texture(r, mask) => {
                let n = self.textures.entry(*r).or_insert(0);
                step(n);
                if *n == 0 {
                    self.textures.remove(r);
                    self.texture_masks.remove(r);
                } else if *mask != u32::MAX {
                    let masks = self.texture_masks.entry(*r).or_default();
                    let count = masks.entry(*mask).or_insert(0);
                    step(count);
                    if *count == 0 {
                        masks.remove(mask);
                    }
                    if masks.is_empty() {
                        self.texture_masks.remove(r);
                    }
                }
            }
            Seen::Left(_) => {}
        }
    }
}

impl Viewed {
    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        let seen = view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) });
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = inner.slots.remove(&index) {
            inner.count(&old, false);
        }
        if let Some(seen) = seen {
            inner.count(&seen, true);
            inner.slots.insert(index, seen);
        }
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).hulls.values().cloned().collect()
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, desc) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        let mask = crate::descriptor::host::srv_mip_mask(*resource as This, desc.as_ref(), &texture.texture);
        return crate::texture::title_written(&texture.texture).then_some(Seen::Texture(*resource, mask));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

// Reverse-port ac64cdb1 and e6be9a0c. Unlike the older renderer,
// the current staging pool is recycled every submission: ahead jobs
// own private upload buffers, and never acknowledge a discarded upload.
const AHEAD_COUNT: usize = 32;
const AHEAD_BYTES: u64 = 16 << 20;
const AHEAD_MS: u128 = 8;
const READY_LIMIT: usize = 8;
const READY_AGE_MS: u128 = 250;

struct Ready {
    viewed: Weak<Viewed>,
    made: std::time::Instant,
    list: Option<HostList>,
    entries: Vec<crate::texture::AheadUpload>,
}

static READY: Mutex<Vec<Ready>> = Mutex::new(Vec::new());

fn worker_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_HEAP_WORKER").as_deref() == Ok("1"))
}

fn ready_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // ff33360: the speculative full dirty list cannot know the actual
    // submission's SRV union. Keep CPU preparation in range mode.
    worker_enabled() && !crate::texture::srv_ranges_enabled()
        && *ON.get_or_init(|| std::env::var("VICTORY_HEAP_READY_LISTS").as_deref() == Ok("1"))
}

fn ahead_pressured() -> bool {
    crate::shadow::memory_pressured() || crate::host::video_memory_pressured()
}

fn ready_live(age_ms: u128, heap_alive: bool, pressure: bool) -> bool {
    !pressure && heap_alive && age_ms <= READY_AGE_MS
}

/// Called under the submission lock. Dropped entries leave their
/// staging retry flag set until mandatory preparation replaces them.
pub(crate) fn maintain_ready() {
    if !worker_enabled() {
        return;
    }
    let pressure = ahead_pressured();
    READY.lock().unwrap_or_else(|e| e.into_inner())
        .retain(|r| ready_live(r.made.elapsed().as_millis(), r.viewed.strong_count() != 0, pressure));
}

fn publish_ready(ready: Ready) {
    let mut queue = READY.lock().unwrap_or_else(|e| e.into_inner());
    queue.retain(|r| !Weak::ptr_eq(&r.viewed, &ready.viewed)
        && ready_live(r.made.elapsed().as_millis(), r.viewed.strong_count() != 0, false));
    if queue.len() >= READY_LIMIT {
        queue.remove(0);
    }
    queue.push(ready);
}

fn take_ready(viewed: &Arc<Viewed>) -> Option<Ready> {
    let key = Arc::downgrade(viewed);
    let mut queue = READY.lock().unwrap_or_else(|e| e.into_inner());
    let i = queue.iter().position(|r| Weak::ptr_eq(&r.viewed, &key))?;
    Some(queue.remove(i))
}

struct AheadJob {
    viewed: Weak<Viewed>,
}

impl Drop for AheadJob {
    fn drop(&mut self) {
        if let Some(viewed) = self.viewed.upgrade() {
            viewed.ahead_queued.store(false, Relaxed);
        }
    }
}

/// Never queue a raw title pointer or wait for room in the queue.
pub(crate) fn kick(viewed: &Arc<Viewed>) {
    if !worker_enabled() || ahead_pressured() {
        return;
    }
    static WORKER: OnceLock<Option<mpsc::SyncSender<AheadJob>>> = OnceLock::new();
    let Some(sender) = WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel::<AheadJob>(8);
        match std::thread::Builder::new().name("victory-heap-ahead".into()).spawn(move || {
            for job in rx {
                if let Some(viewed) = job.viewed.upgrade()
                    && let Some(host) = crate::host::host()
                {
                    host.prepare_heap_ahead(&viewed);
                }
                drop(job);
            }
        }) {
            Ok(_) => Some(tx),
            Err(e) => {
                tracing::warn!(error = %e, "heap preparation worker unavailable");
                None
            }
        }
    }) else { return };
    if viewed.ahead_queued.swap(true, Relaxed) {
        return;
    }
    let _ = sender.try_send(AheadJob { viewed: Arc::downgrade(viewed) });
}

fn ahead_keys(textures: &BTreeMap<usize, u32>, cursor: Option<usize>, count: usize) -> Vec<usize> {
    use std::ops::Bound::{Excluded, Unbounded};
    match cursor {
        Some(last) => textures.range((Excluded(last), Unbounded))
            .chain(textures.range(..=last)).take(count).map(|(&key, _)| key).collect(),
        None => textures.keys().take(count).copied().collect(),
    }
}

/// The host holds its submission lock. Only made, recent, title-fed
/// copies qualify. The heap lock keeps the title object alive while
/// reading it, but published entries own only host objects and staging.
pub(crate) fn prepare_ahead(viewed: &Arc<Viewed>) {
    maintain_ready();
    if ahead_pressured() {
        return;
    }
    let Ok(inner) = viewed.inner.try_lock() else { return };
    let Ok(mut cursor) = viewed.ahead_cursor.try_lock() else { return };
    let keys = ahead_keys(&inner.textures, *cursor, AHEAD_COUNT);
    let started = std::time::Instant::now();
    let mut left = AHEAD_BYTES;
    let mut entries: Vec<crate::texture::AheadUpload> = Vec::new();
    for key in keys {
        if left == 0 || started.elapsed().as_millis() >= AHEAD_MS || ahead_pressured() {
            break;
        }
        *cursor = Some(key);
        let Some(copy) = crate::resource::made_copy(key as This) else { continue };
        if !copy.used_recently() || copy.host_written.load(Relaxed)
            || entries.iter().any(|e| Arc::ptr_eq(e.copy(), &copy))
        {
            continue;
        }
        if let Some((cost, prepared)) = crate::texture::prepare_ahead(key as This, copy, left) {
            left = left.saturating_sub(cost);
            if let Some(prepared) = prepared {
                entries.push(prepared);
            }
        }
    }
    drop(inner);
    drop(cursor);
    if entries.is_empty() || ahead_pressured() {
        return;
    }
    let list = if ready_enabled() && entries.iter().all(crate::texture::AheadUpload::can_record) {
        crate::host::host().and_then(|host| host.new_list().ok()).inspect(|list| {
            list.set_ours();
            let readable: Vec<_> = entries.iter().filter_map(|e| e.record(list, false)).collect();
            if !readable.is_empty() {
                // SAFETY: an open list naming live, owned textures.
                unsafe { list.list.ResourceBarrier(&readable) };
            }
            list.close();
        })
    } else {
        None
    };
    publish_ready(Ready { viewed: Arc::downgrade(viewed), made: started, list, entries });
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[crate::texture::RefreshWork],
) -> Option<Uploads> {
    static LIST: Mutex<Holder> = Mutex::new(Holder(None));
    let mut guard = LIST.lock().unwrap_or_else(|e| e.into_inner());
    crate::texture::recycle();
    match &guard.0 {
        // Closed by the last submission, which has run.
        Some(list) => list.reset(),
        // Made open.
        None => {
            let list = crate::host::host()?.new_list().ok()?;
            list.set_ours();
            guard.0 = Some(list);
        }
    }
    let list = guard.0.as_ref()?;
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Collected under the heaps'
    // locks and brought up without them, several at a time: reading title
    // memory, detiling and making staging buffers took half the submitting
    // thread in a new shot, and title threads setting views waited on the
    // lock meanwhile (run evict5).
    let mut work: Vec<crate::texture::RefreshWork> = Vec::new();
    let mut staged: HashMap<*const crate::texture::Staging, usize> = HashMap::new();
    let range_mode = crate::texture::srv_ranges_enabled();
    for (staging, copy, requested) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded).
        if copy.host_written.load(Relaxed) {
            continue;
        }
        let mask = if range_mode { *requested } else { u32::MAX };
        match staged.entry(Arc::as_ptr(staging)) {
            std::collections::hash_map::Entry::Occupied(e) => work[*e.get()].2 |= mask,
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(work.len());
                work.push((staging.clone(), copy.clone(), mask));
            }
        }
    }
    let mandatory = work.len();
    let mut done = std::collections::HashSet::new();
    for viewed in heaps {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        for &r in inner.textures.keys() {
            if !range_mode && !done.insert(r) {
                continue;
            }
            // Not made (given up, or never read): nothing to bring up.
            let Some(texture) = crate::resource::made_copy(r as This) else {
                continue;
            };
            // Only what shaders read lately (the eviction pass's stamps):
            // the heaps view thousands of textures a frame never reads, and
            // bringing each up whenever its pages change re-uploaded about
            // 4 GB in 30-46 world frames (runs evict2, noevict1).
            // shortcut: a texture read again after a gap is brought up after
            // the submission that read it, which can see bytes one refresh
            // old (a slot the title rewrites is made fresh anyway); upload on
            // the mark itself if that shows.
            if !texture.used_recently() {
                continue;
            }
            // Its contents are the host's since a host copy wrote it.
            if texture.host_written.load(Relaxed) {
                continue;
            }
            if let Some(staging) = crate::texture::staging(r as This, &texture.texture) {
                let mask = if range_mode { inner.requested_mips(r) } else { u32::MAX };
                // Union across heaps and title aliases before reading once.
                // Explicit copy requests carry full masks; table reads
                // carry their requested range when the option is enabled.
                match staged.entry(Arc::as_ptr(&staging)) {
                    std::collections::hash_map::Entry::Occupied(e) => work[*e.get()].2 |= mask,
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(work.len());
                        work.push((staging, texture, mask));
                    }
                }
            }
        }
    }
    // Observe title-held levels independently of dirty hashes and ahead
    // reuse. Do not narrow allocations, uploads or published SRVs.
    if crate::tiles::judgement_enabled() && work.iter().any(|(_, copy, _)| copy.tiled.is_some()) {
        let hints = crate::descriptor::placed_judgements();
        for (staging, copy, _) in &work {
            staging.observe_held(copy, &hints);
        }
    }
    // The reference's waiting-source path is applied before detiling,
    // only to opportunistic heap reads. Explicit table/copy requests
    // retain mandatory admission and the existing full-chain fallback.
    // In the production full-map path this makes no allocation.
    if work.iter().skip(mandatory).any(|(_, copy, _)| copy.tiled.is_some()) {
        let gpu = crate::host::host()?;
        let mut index = 0;
        work.retain(|(staging, copy, mips)| {
            let explicit = index < mandatory;
            index += 1;
            if explicit || copy.tiled.is_none() { return true; }
            // SAFETY: this upload pass runs under the submission lock,
            // before preludes or any detiling for this batch.
            match unsafe { crate::tiles::prepare_optional_source(gpu, copy, staging.source_backing_mask(*mips)) } {
                Some(added) => {
                    if added { staging.retry_after_backing(); }
                    true
                }
                None => false, // dirty state untouched; retry when due
            }
        });
        // The ready-list validator indexes work by staging identity.
        // Filtering waiting work may have moved those indices.
        staged.clear();
        staged.extend(work.iter().enumerate().map(|(i, (staging, _, _))| (Arc::as_ptr(staging), i)));
    }
    let mut recorded = false;
    let mut readable = Vec::new();
    let mut accepted = Vec::new();
    if worker_enabled() && !ahead_pressured() {
        let mut used = std::collections::HashSet::new();
        for viewed in heaps {
            let Some(ready) = take_ready(viewed) else { continue };
            // All or nothing for a recorded list. The actual submission
            // work set contains current owners, including explicit reads.
            if !ready_live(ready.made.elapsed().as_millis(), true, false)
                || !ready.entries.iter().all(|entry| {
                    let key = Arc::as_ptr(entry.staging());
                    !used.contains(&key) && staged.get(&key)
                        .is_some_and(|&i| entry.revalidate(&work[i]))
                })
            {
                continue;
            }
            for entry in &ready.entries {
                let key = Arc::as_ptr(entry.staging());
                let mips = work[staged[&key]].2;
                let (copied, barrier) = entry.accept(ready.list.is_none().then_some(list), mips);
                used.insert(key);
                readable.extend(barrier);
                if copied {
                    recorded |= ready.list.is_none();
                    #[cfg(feature = "diagnostics")]
                    UPLOADED.fetch_add(1, Relaxed);
                }
            }
            accepted.push(ready);
        }
        work.retain(|(staging, _, _)| !used.contains(&Arc::as_ptr(staging)));
    }
    // The textures copied into stay alive in `work` until recorded.
    for ((staging, copy, _), upload) in work.iter().zip(crate::texture::refresh_all(&work)) {
        if let Some(upload) = upload {
            readable.extend(crate::texture::record_upload(list, copy, staging, &upload));
            recorded = true;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    if !readable.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&readable) };
    }
    list.close();
    // A list recording nothing is not run (its reset next time is fine).
    if recorded || !accepted.is_empty() {
        Some(Uploads { ready: accepted, fallback: recorded, guard })
    } else {
        None
    }
}

/// The upload list, used under its lock only.
pub(crate) struct Holder(Option<HostList>);
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, under the lock.
unsafe impl Send for Holder {}

/// All private ahead buffers and accepted lists remain owned through
/// the submission fence. State preludes run in actual execution order.
pub(crate) struct Uploads {
    ready: Vec<Ready>,
    fallback: bool,
    guard: std::sync::MutexGuard<'static, Holder>,
}

impl Uploads {
    pub(crate) fn lists(&self) -> impl Iterator<Item = &HostList> {
        self.ready.iter().filter_map(|r| r.list.as_ref())
            .chain(self.guard.0.as_ref().filter(|_| self.fallback))
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, address, len, uav, bit) in inner.hulls.values() {
        if !crate::watch::clean_bit(*bit) {
            let (address, len) = if *uav && shadow.marked() {
                shadow.page_range(*address, *len)
            } else {
                (*address, *len)
            };
            add(shadow, address as usize, len as usize);
        }
    }
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
    }
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.textures.remove(&resource).is_some() {
            inner.texture_masks.remove(&resource);
            inner.slots.retain(|_, seen| !matches!(seen, Seen::Texture(r, _) if *r == resource));
        }
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = HashMap::new();
        for seen in inner.slots.values() {
            match seen {
                Seen::Buffer(..) => buffers += 1,
                Seen::Texture(..) => textures += 1,
                Seen::Left((k, address, bytes)) => {
                    left[*k] += 1;
                    if *k == 2 {
                        let b = placed.entry(*address).or_insert(0u64);
                        *b = (*b).max(*bytes);
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let bytes: u64 = inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum();
        drop(inner);
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let copies = {
            let mut c: Vec<usize> = inner.ranges.keys().map(|k| k.0).collect();
            c.dedup();
            c.len()
        };
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame; slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            inner.ranges.len(),
            bytes as f64 / 1e6,
            inner.textures.len(),
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ahead_cursor_wraps_without_repeating_entries() {
        let textures = BTreeMap::from([(1, 1), (3, 1), (7, 1), (9, 1)]);
        assert_eq!(ahead_keys(&textures, None, 2), [1, 3]);
        assert_eq!(ahead_keys(&textures, Some(3), 3), [7, 9, 1]);
        assert_eq!(ahead_keys(&textures, Some(6), 8), [7, 9, 1, 3]);
        assert!(ahead_keys(&BTreeMap::new(), Some(9), 4).is_empty());
    }

    #[test]
    fn rejected_or_finished_ahead_job_clears_queue_flag() {
        let viewed = Arc::new(Viewed::default());
        viewed.ahead_queued.store(true, Relaxed);
        let (tx, _rx) = mpsc::sync_channel::<AheadJob>(0);
        assert!(tx.try_send(AheadJob { viewed: Arc::downgrade(&viewed) }).is_err());
        assert!(!viewed.ahead_queued.load(Relaxed));
        viewed.ahead_queued.store(true, Relaxed);
        drop(AheadJob { viewed: Arc::downgrade(&viewed) });
        assert!(!viewed.ahead_queued.load(Relaxed));
    }

    #[test]
    fn ready_lists_expire_or_are_dropped_under_pressure() {
        assert!(ready_live(0, true, false));
        assert!(ready_live(READY_AGE_MS, true, false));
        assert!(!ready_live(READY_AGE_MS + 1, true, false));
        assert!(!ready_live(0, false, false));
        assert!(!ready_live(0, true, true));
    }

    #[test]
    fn texture_srv_union_tracks_duplicate_rewrites_and_full_views() {
        let mut inner = Inner::default();
        let a = Seen::Texture(11, 0b0011);
        let b = Seen::Texture(11, 0b1100);
        let full = Seen::Texture(11, u32::MAX);
        inner.count(&a, true);
        inner.count(&a, true);
        inner.count(&b, true);
        assert_eq!(inner.requested_mips(11), 0b1111);
        inner.count(&a, false);
        assert_eq!(inner.requested_mips(11), 0b1111);
        inner.count(&a, false);
        assert_eq!(inner.requested_mips(11), 0b1100);
        inner.count(&full, true);
        assert_eq!(inner.requested_mips(11), u32::MAX);
        inner.count(&full, false);
        assert_eq!(inner.requested_mips(11), 0b1100);
        inner.count(&b, false);
        assert!(inner.textures.is_empty());
        assert!(inner.texture_masks.is_empty());
    }
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0), UAV(u0), SRV(t1))"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[RootSignature(RS)]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("heap"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a compiled shader with its root signature inside.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            let mut inner = viewed.inner.lock().unwrap();
            if let Some(old) = inner.slots.remove(&slot) {
                inner.count(&old, false);
            }
            if let Some((from, len)) = range {
                let seen = Seen::Buffer(copy.clone(), at + from, len, false);
                inner.count(&seen, true);
                inner.slots.insert(slot, seen);
            }
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }
}
