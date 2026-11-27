//! Reference picked-view shader classification and write discovery.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, picked.rs.
//! This branch already compares all carded buffers for heap-indexed writes.
//! Reuse those flags to observe possible undeclared writes. Stage 2
//! registers previously found live UAV ranges before a future submission,
//! never retroactively lands a discovery, and keeps full GPU coverage.

use std::collections::{BTreeMap, BTreeSet};
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

/// Opt-in scheduling of discovery from completed comparison flags.
/// The reference reduces GPU work by delaying hashes and using mode 1
/// look-only compares. This branch shares the hash baseline with CPU
/// merge, so do not delay its GPU reads or modify its landing semantics.
pub(crate) fn rotates() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PICKED_ROTATE").is_some_and(|v| v == "1"))
}

/// Reference picked::slice, with a checked/saturating byte conversion.
fn slice() -> u64 {
    static SLICE: OnceLock<u64> = OnceLock::new();
    *SLICE.get_or_init(|| {
        let mb = std::env::var("VICTORY_PICKED_MB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        mb.unwrap_or(32).max(1).saturating_mul(1 << 20)
    })
}

/// A buffer the heap shows unordered views of: copy offsets, from the
/// first to the last.
struct Shown {
    copy: Weak<Shadow>,
    from: u64,
    to: u64,
}

#[derive(Default)]
struct State {
    shown: BTreeMap<u64, Shown>,
    /// The copies (by number) found written through picked views.
    found: BTreeSet<u64>,
    /// Where the next slice begins: copy, offset from its first address.
    next: (u64, u64),
    /// The frame of the last slice; frame zero is also eligible.
    frame: Option<u64>,
    /// One unexplained flagged piece per not-yet-found copy. The original
    /// GPU comparison stays full, so evidence must survive skipped turns.
    pending: BTreeMap<u64, (u64, u64)>,
}

static PROBES: Mutex<Option<State>> = Mutex::new(None);

impl State {
    /// The ranges of the next `most` bytes in turn, at most all once.
    fn turn(&mut self, most: u64) -> Vec<(u64, u64, u64)> {
        let mut out: Vec<(u64, u64, u64)> = Vec::new();
        let mut left = most;
        // (The one it begins in may come twice: its end first.)
        for _ in 0..=self.shown.len() {
            let (copy, offset) = self.next;
            let Some((&id, s)) = self.shown.range(copy..).next().or_else(|| self.shown.iter().next()) else {
                break;
            };
            let offset = if id == copy { offset } else { 0 };
            let len = s.to - s.from;
            let n = len.saturating_sub(offset).min(left);
            if n > 0 && !self.found.contains(&id) {
                if out.iter().any(|r| r.0 == id) {
                    break;
                }
                out.push((id, s.from + offset, n));
                left -= n;
                if offset + n < len {
                    self.next = (id, offset + n);
                    break;
                }
            }
            self.next = (id + 1, 0);
            if left == 0 {
                break;
            }
        }
        out
    }
}

impl State {
    /// Consume evidence only when its bytes fall inside the current turn.
    /// No re-hash or late write-back: the evidence was captured from the
    /// completed submission where it occurred. A found copy needs no
    /// further rotating discovery.
    fn take_seen(&mut self, most: u64) -> Vec<(u64, u64)> {
        let mut found = Vec::new();
        for (id, at, len) in self.turn(most) {
            let hit = self.pending.get(&id).is_some_and(|&(lo, hi)|
                lo < at.saturating_add(len) && at < hi);
            if hit {
                if let Some((lo, _)) = self.pending.remove(&id) {
                    self.found.insert(id);
                    found.push((id, lo));
                }
            }
        }
        found
    }

    fn take_frame(&mut self, frame: u64, most: u64) -> Vec<(u64, u64)> {
        if self.frame == Some(frame) {
            return Vec::new();
        }
        self.frame = Some(frame);
        self.take_seen(most)
    }
}

/// Both vectors name exact live owners and copy-relative byte ranges.
/// Keep pending evidence even if a view disappears before its turn,
/// but never use an old view to declare writes: known_ranges still
/// requires the UAV to be present in the current submission.
pub(crate) fn rotate(shown: Vec<Range>, evidence: Vec<Range>) {
    if !rotates() {
        return;
    }
    let known = {
        let mut found = FOUND.lock().unwrap_or_else(|e| e.into_inner());
        found.retain(|_, owner| owner.strong_count() != 0);
        found.keys().map(|&id| id as u64).collect::<BTreeSet<_>>()
    };
    let ready = {
        let mut state = PROBES.lock().unwrap_or_else(|e| e.into_inner());
        let state = state.get_or_insert_with(State::default);
        state.found = known;
        state.shown.retain(|id, view| view.copy.strong_count() != 0
            && state.pending.contains_key(id) && !state.found.contains(id));
        state.pending.retain(|id, _| state.shown.contains_key(id));
        for (copy, from, len) in shown {
            let id = Arc::as_ptr(&copy) as u64;
            if len == 0 || state.found.contains(&id) {
                continue;
            }
            let from = from as u64;
            let to = from.saturating_add(len as u64);
            let e = state.shown.entry(id).or_insert_with(|| Shown {
                copy: Arc::downgrade(&copy), from, to,
            });
            (e.from, e.to) = (e.from.min(from), e.to.max(to));
        }
        for (copy, from, len) in evidence {
            let id = Arc::as_ptr(&copy) as u64;
            if let Some(view) = state.shown.get_mut(&id)
                && !state.found.contains(&id) && len != 0
            {
                let lo = from as u64;
                let hi = lo.saturating_add(len as u64);
                // A hash covers a whole piece; preserve its intersection
                // with the known UAV hull, even for an unaligned view.
                let (lo, hi) = (lo.max(view.from), hi.min(view.to));
                if lo < hi {
                    state.pending.entry(id).or_insert((lo, hi));
                }
            }
        }
        state.take_frame(crate::frame::current(), slice()).into_iter()
            .filter_map(|(id, at)| Some((state.shown.get(&id)?.copy.upgrade()?, at as usize)))
            .collect::<Vec<_>>()
    };
    // Do not hold the probe lock while updating the found registry.
    for (copy, at) in ready {
        found(&copy, at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(sizes: &[u64]) -> State {
        let mut state = State::default();
        for (i, &n) in sizes.iter().enumerate() {
            let from = (i as u64 + 1) << 32;
            state.shown.insert(i as u64 * 3 + 2, Shown { copy: Weak::new(), from, to: from + n });
        }
        state
    }

    /// Every byte in turn, none left out, whatever the slice.
    #[test]
    fn all_buffers_are_looked_at_in_turn() {
        let sizes = [100u64, 7, 1000, 1, 350];
        let total: u64 = sizes.iter().sum();
        for most in [1u64, 10, 99, 350, 1000, 5000] {
            let mut s = state(&sizes);
            let mut seen = std::collections::BTreeMap::<u64, u64>::new();
            // (A turn ends with the list: it may take one more.)
            let turns = total.div_ceil(most) + sizes.len() as u64 + 1;
            for _ in 0..turns {
                let turn = s.turn(most);
                assert!(turn.iter().map(|r| r.2).sum::<u64>() <= most);
                for (id, at, n) in turn {
                    let from = s.shown[&id].from;
                    for b in at - from..at - from + n.min(2000) {
                        *seen.entry(id << 32 | b).or_default() += 1;
                    }
                }
            }
            assert_eq!(seen.len() as u64, total, "slices of {most}");
        }
    }

    #[test]
    fn what_was_found_has_no_turn() {
        let mut s = state(&[64, 64, 64]);
        s.found.insert(5);
        let turn = s.turn(1000);
        assert_eq!(turn.iter().map(|r| r.0).collect::<Vec<_>>(), [2, 8]);
        assert!(state(&[]).turn(10).is_empty());
        let mut all = state(&[64]);
        all.found.insert(2);
        assert!(all.turn(10).is_empty());
    }
}

#[test]
fn rotating_evidence_survives_a_skipped_frame_and_is_consumed_once() {
    let mut s = State::default();
    s.shown.insert(2, Shown { copy: Weak::new(), from: 0, to: 16 });
    s.shown.insert(5, Shown { copy: Weak::new(), from: 0, to: 16 });
    s.pending.insert(5, (4, 5));
    assert!(s.take_frame(0, 16).is_empty());
    assert_eq!(s.pending.get(&5), Some(&(4, 5)));
    assert!(s.take_frame(0, 16).is_empty(), "only one turn per frame");
    assert_eq!(s.take_frame(1, 16), [(5, 4)]);
    assert!(s.pending.is_empty());
    assert!(s.found.contains(&5));
    assert!(s.take_frame(2, 32).is_empty());
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
