//! Reference picked-view shader classification and write discovery.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, picked.rs.
//! This branch already compares all carded buffers for heap-indexed writes.
//! Reuse those flags to observe possible undeclared writes. Stage 2
//! registers previously found live UAV ranges before a future submission,
//! never retroactively lands a discovery, and keeps full GPU coverage.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use crate::pipeline::Shader;
use crate::shadow::Shadow;

pub(crate) fn on() -> bool {
    stage() > 0
}

/// Reference stage selection: off, observation, or automatic landing.
/// Stage 1 retains this branch's observation-only behavior.
fn stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| match std::env::var("VICTORY_PICKED_WRITES").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

pub(crate) fn lands() -> bool {
    stage() > 1
}

// Source picked.rs UNKNOWN / ASKED / NO / YES, stored on Shader here
// instead of Translated so cached and test translations need no new fields.
const UNKNOWN: u8 = 0;
const ASKED: u8 = 1;
const NO: u8 = 2;
const YES: u8 = 3;

/// As in the source, run descriptor tracing away from the draw thread.
/// Pending or unanalyzable programs are candidates, not proof of a write.
pub(crate) fn picks(shader: &Arc<Shader>) -> bool {
    match shader.picked.load(Relaxed) {
        NO => false,
        UNKNOWN => {
            if shader.picked.compare_exchange(UNKNOWN, ASKED, Relaxed, Relaxed).is_ok() {
                ask(shader.clone());
            }
            true
        }
        _ => true,
    }
}

fn ask(shader: Arc<Shader>) {
    use std::sync::mpsc::{SyncSender, sync_channel};
    static ASKS: OnceLock<SyncSender<Arc<Shader>>> = OnceLock::new();
    let asks = ASKS.get_or_init(|| {
        // Bound outstanding diagnostics; a full queue retries at next bind.
        let (send, asks) = sync_channel::<Arc<Shader>>(256);
        let _ = std::thread::Builder::new().name("victory-picked".into()).spawn(move || {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
            // SAFETY: this thread's own priority.
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
            for shader in asks {
                let yes = shader.programs().is_none_or(|all| all.iter().any(stores_picked));
                shader.picked.store(if yes { YES } else { NO }, Relaxed);
                if yes {
                    tracing::debug!(hash = %shader.hash, "possible picked-view buffer store");
                }
            }
        });
        send
    });
    if asks.try_send(shader.clone()).is_err() {
        shader.picked.store(UNKNOWN, Relaxed);
    }
}

/// Whether the code of `program` stores to a buffer through a descriptor
/// that its tables do not name: loaded from the view heap, or from a table at
/// an offset it computes (programs with the view heap among their parameters). Yes where that cannot be told.
fn stores_picked(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::HwStage;
    use victory_shader::gfx10::Fields;
    use victory_shader::native::RootType;
    use victory_shader::resource::{Role, Source};
    // (The programs that are translated.)
    if !matches!(program.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms) {
        return false;
    }
    let heap = |param: usize| program.layout.get(param).is_some_and(|p| p.root == RootType::ResourceHeap);
    if !(0..program.layout.len()).any(heap) {
        return false;
    }
    let Ok(p) = victory_shader::ir::build(&program.patched_code(), program.wave64, program.vgprs) else {
        return true;
    };
    // The structure is what takes time, and only per-lane descriptors need
    // it.
    let plain = victory_shader::resource::track(&p, &program.layout, program.stage, None);
    let res = plain.or_else(|_| {
        let st = victory_shader::structure::structure(&p);
        victory_shader::resource::track(&p, &program.layout, program.stage, st.as_ref().ok())
    });
    let Ok(res) = res else {
        return true;
    };
    res.descriptors.iter().filter(|d| d.role == Role::Resource).any(|d| {
        let inst = &p.blocks[d.site.0].insts[d.site.1];
        let stores = matches!(inst.fields, Fields::Buffer { .. }) && (inst.name().contains("store") || inst.name().contains("atomic"));
        stores
            && d.source.leaves().into_iter().any(|s| match s {
                Source::Loaded { load, .. } | Source::Fetched { load, .. } => res
                    .scalar
                    .iter()
                    .find(|(site, _)| site == load)
                    .is_some_and(|(_, a)| heap(a.param) || a.dynamic.is_some()),
                _ => false,
            })
    })
}

/// The reference's found registry, keyed by a weak exact owner rather than
/// its source-only monotonic Shadow id. Weak keeps the address from being
/// reused while tracked; prune dead entries before recording a new one.
static FOUND: Mutex<BTreeMap<usize, Weak<Shadow>>> = Mutex::new(BTreeMap::new());

pub(crate) fn found(copy: &Arc<Shadow>, at: usize) {
    let mut found = FOUND.lock().unwrap_or_else(|e| e.into_inner());
    found.retain(|_, owner| owner.strong_count() != 0);
    let id = Arc::as_ptr(copy) as usize;
    if let std::collections::btree_map::Entry::Vacant(entry) = found.entry(id) {
        entry.insert(Arc::downgrade(copy));
        let (start, len) = copy.range();
        tracing::info!(
            "possible undeclared picked-view write: {start:#x}+{len:#x}, piece +{at:#x}; found for future submissions (VICTORY_PICKED_WRITES)"
        );
    }
}

/// A live UAV range, in title addresses. Adapt source picked::declare
/// and picked::with to this branch's weak exact-owner registry.
pub(crate) type Range = (Arc<Shadow>, usize, usize);

fn key(range: &Range) -> (usize, usize, usize) {
    (Arc::as_ptr(&range.0) as usize, range.1, range.2)
}

/// Existing declarations or marks may own a snapshot of the entire page.
/// Exclude the same copy even for disjoint ranges, and overlapping title
/// pages of distinct copies. This avoids double merge and alias conflicts.
fn conflicts(a: (usize, usize, usize), b: (usize, usize, usize)) -> bool {
    const PAGE: usize = 1 << 16;
    if a.2 == 0 || b.2 == 0 {
        return false;
    }
    a.0 == b.0
        || (a.1 / PAGE <= b.1.saturating_add(b.2).saturating_sub(1) / PAGE
            && b.1 / PAGE <= a.1.saturating_add(a.2).saturating_sub(1) / PAGE)
}

pub(crate) fn overlaps(a: &Range, b: &Range) -> bool {
    conflicts(key(a), key(b))
}

/// Reference found -> shown intersection. Merge live hulls per exact copy,
/// then exclude ambiguous ownership. No stale descriptor or found pointer
/// alone is sufficient to create a write declaration.
pub(crate) fn known_ranges(shown: Vec<Range>, declared: &[Range]) -> Vec<Range> {
    let mut found = FOUND.lock().unwrap_or_else(|e| e.into_inner());
    found.retain(|_, owner| owner.strong_count() != 0);
    let mut hulls = BTreeMap::<usize, Range>::new();
    for (copy, address, len) in shown {
        let id = Arc::as_ptr(&copy) as usize;
        if len == 0 || copy.card().is_none() || !copy.marked()
            || !found.get(&id).is_some_and(|owner| owner.ptr_eq(&Arc::downgrade(&copy)))
        {
            continue;
        }
        let (start, size) = copy.range();
        let from = address.max(start as usize);
        let to = address.saturating_add(len).min(start.saturating_add(size) as usize);
        if from >= to {
            continue;
        }
        hulls.entry(id).and_modify(|h| {
            let lo = h.1.min(from);
            let hi = h.1.saturating_add(h.2).max(to);
            h.1 = lo;
            h.2 = hi - lo;
        }).or_insert((copy, from, to - from));
    }
    drop(found);
    let candidates: Vec<_> = hulls.into_values().collect();
    let conflicts_with = |i: usize, range: &Range| {
        declared.iter().any(|d| overlaps(range, d))
            || candidates.iter().enumerate().any(|(j, other)| i != j && overlaps(range, other))
    };
    candidates.iter().enumerate().filter(|(i, range)| !conflicts_with(*i, range))
        .map(|(_, range)| range.clone()).collect()
}

#[test]
fn picked_declarations_exclude_same_copy_and_overlapping_title_pages() {
    let page = 1 << 16;
    assert!(conflicts((1, page, 4), (1, 4 * page, 4)));
    assert!(conflicts((1, page, 4), (2, page + 4096, 4)));
    assert!(conflicts((1, 2 * page - 1, 2), (2, 2 * page, 4)));
    assert!(!conflicts((1, page, page), (2, 2 * page, page)));
    assert!(!conflicts((1, page, 0), (1, page, 4)));
}

/// A hash flags a whole piece, not individual bytes. Any declaration or
/// mark intersecting that piece can explain the change, so do not claim
/// undeclared writes in it, even if another UAV overlaps the piece.
pub(crate) fn unexplained(piece: (usize, usize), shown: &[(usize, usize)], declared: &[(usize, usize)]) -> bool {
    let overlaps = |&(lo, hi): &(usize, usize)| lo < piece.1 && piece.0 < hi;
    piece.0 < piece.1 && shown.iter().any(overlaps) && !declared.iter().any(overlaps)
}

#[test]
fn picked_discovery_requires_a_uav_and_no_explaining_write() {
    let page = 1 << 16;
    let shown = [(page + 20, 3 * page)];
    assert!(!unexplained((0, page), &shown, &[]));
    assert!(unexplained((page, 2 * page), &shown, &[(0, page)]));
    assert!(!unexplained((page, 2 * page), &shown, &[(2 * page - 1, 2 * page)]));
    assert!(!unexplained((page, 2 * page), &shown, &[(page, page + 1)]));
    assert!(!unexplained((3 * page, 4 * page), &shown, &[]));
    assert!(!unexplained((page, page), &shown, &[]));
}
