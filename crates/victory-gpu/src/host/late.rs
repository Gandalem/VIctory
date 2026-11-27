//! Reference late-submission dependency and settlement boundary.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, host.rs
//! touches, join_ranges, blocked_by, late_blocked, send_late and settle.
//! A pump may own ONE signalled title flight, never two. Shared marks,
//! card flags and physical aliases still require checked settlement.

use super::{Host, HostList, SentTitle, Step};
use crate::list::Detached;
use std::sync::{MutexGuard, OnceLock};

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

#[test]
fn ranges_touch_where_they_overlap() {
    let mut ranges = Vec::new();
    join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
    assert_eq!(ranges, [(10, 25), (40, 60)]);
    join_ranges(&mut ranges, [(25, 40)]);
    assert_eq!(ranges, [(10, 60)]);
    let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
    for (from, to, hit) in [
        (0, 10, false),
        (0, 11, true),
        (24, 25, true),
        (25, 40, false),
        (25, 41, true),
        (59, 200, true),
        (60, 100, false),
        (100, 101, true),
        (101, 500, false),
        (30, 30, false),
        (50, 50, false),
    ] {
        assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
    }
    assert!(!touches(&[], 0, usize::MAX));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SettleWhy {
    CopySource,
    CopyDestination,
    Fill,
    Texture,
    Refresh,
    Write,
    Unknown,
    MemoryWait,
    Fence,
    SharedState,
    PumpEnd,
    Dropped,
}

#[derive(Default)]
struct FlightWrites {
    written: Vec<(usize, usize)>,
    unknown: bool,
}

impl FlightWrites {
    fn add(&mut self, at: usize, len: usize) {
        if len != 0 {
            if let Some(end) = at.checked_add(len) {
                self.written.push((at, end));
            } else {
                self.unknown = true;
            }
        }
    }

    fn join(&mut self) {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, std::mem::take(&mut self.written));
        self.written = ranges;
    }

    fn capture(runs: &[&HostList], sent: &SentTitle) -> Self {
        let mut out = Self::default();
        for list in sent.uploads.as_ref().into_iter().flat_map(|u| u.lists())
            .chain(runs.iter().copied())
        {
            let d = list.deferred();
            out.unknown |= d.card_unbounded || d.card_picked
                || !d.after.is_empty() || !d.missed.is_empty()
                || !d.tile_writes.is_empty() || !d.tile_sources.is_empty();
            for step in &d.copies {
                match step {
                    Step::GpuWrite(_, at, len, _, _) => out.add(*at, *len),
                    Step::Memory(..) | Step::Heap(..) => out.unknown = true,
                    _ => {}
                }
            }
            for &(at, _, len) in &d.writes {
                out.add(at, len);
            }
        }
        for (_, at, len) in &sent.picked {
            out.add(*at, *len);
        }
        out.join();
        out
    }

    fn touches(&self, at: usize, len: usize) -> bool {
        len != 0 && at.checked_add(len)
            .is_none_or(|end| touches(&self.written, at, end))
    }

    /// Source copy_blocked: source takes precedence when both overlap.
    /// No fill_is_titles exemption until physical aliases are handled.
    fn copy_blocked(&self, dst: usize, src: usize, len: usize) -> Option<SettleWhy> {
        if self.touches(src, len) {
            Some(SettleWhy::CopySource)
        } else if self.touches(dst, len) {
            Some(SettleWhy::CopyDestination)
        } else {
            None
        }
    }

    /// Source blocked_by, with conservative refresh/write checks.
    /// Read-only inspection; no dirty state or title pointers consumed.
    fn blocked_by(&self, runs: &[&HostList]) -> Option<SettleWhy> {
        if self.unknown {
            return Some(SettleWhy::Unknown);
        }
        if self.written.is_empty() {
            return None;
        }
        for list in runs {
            let d = list.deferred();
            for step in &d.copies {
                let reason = match step {
                    Step::Copy(dst, src, len) => self.copy_blocked(*dst, *src, *len),
                    Step::Fill(_, src, len) => self.touches(*src, *len).then_some(SettleWhy::Fill),
                    Step::Texture(staging, ..) => {
                        let (at, len) = staging.memory();
                        self.touches(at, len).then_some(SettleWhy::Texture)
                    }
                    Step::Refresh(_, at, len) => self.touches(*at, *len).then_some(SettleWhy::Refresh),
                    Step::GpuWrite(_, at, len, ..) => self.touches(*at, *len).then_some(SettleWhy::Write),
                    Step::Run(..) | Step::Host(..) | Step::Heap(..) | Step::Memory(..) => Some(SettleWhy::Unknown),
                };
                if reason.is_some() {
                    return reason;
                }
            }
            if d.writes.iter().any(|&(at, _, len)| self.touches(at, len)) {
                return Some(SettleWhy::Write);
            }
        }
        None
    }

    fn read_reason(&self, at: usize, len: usize) -> SettleWhy {
        if self.unknown {
            SettleWhy::Unknown
        } else if self.touches(at, len) {
            SettleWhy::Write
        } else {
            // A disjoint virtual address does not prove physical independence.
            SettleWhy::MemoryWait
        }
    }
}

/// First, pump-local stage of the source late wait switch.
pub(crate) fn late_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_LATE_WAIT").is_some_and(|v| v == "1")
            && crate::list::detach_enabled()
            && std::env::var_os("VICTORY_GPU_TIMING").is_none()
    })
}

struct Flight<'a> {
    lists: Vec<Detached>,
    sent: SentTitle,
    submitted: MutexGuard<'a, u64>,
}

/// Own the actual recording and submission guard until settlement.
pub(crate) struct PendingTitle<'a> {
    host: &'a Host,
    flight: Option<Flight<'a>>,
    writes: FlightWrites,
}

impl Host {
    /// Reference execute_late/send_late: return after the signal.
    /// This first stage retains the guard, so nothing else is submitted.
    pub(crate) fn execute_late(&self, lists: Vec<Detached>) -> PendingTitle<'_> {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let runs: Vec<&HostList> = lists.iter().map(Detached::host).collect();
        let sent = self.submit_title(&runs, &mut submitted);
        drop(runs);
        // Establish the completion owner before allocating range metadata.
        let mut pending = PendingTitle {
            host: self,
            flight: Some(Flight { lists, sent, submitted }),
            writes: FlightWrites::default(),
        };
        let flight = pending.flight.as_ref().expect("new flight");
        let runs: Vec<&HostList> = flight.lists.iter().map(Detached::host).collect();
        pending.writes = FlightWrites::capture(&runs, &flight.sent);
        drop(runs);
        pending
    }
}

impl PendingTitle<'_> {
    pub(crate) fn late_blocked(&self, next: &[Detached]) -> SettleWhy {
        let runs: Vec<&HostList> = next.iter().map(Detached::host).collect();
        self.writes.blocked_by(&runs).unwrap_or(SettleWhy::SharedState)
    }

    pub(crate) fn read_reason(&self, at: usize, len: usize) -> SettleWhy {
        self.writes.read_reason(at, len)
    }

    pub(crate) fn settle(mut self, why: SettleWhy) {
        self.complete(why);
    }

    fn complete(&mut self, why: SettleWhy) {
        if let Some(Flight { lists, sent, submitted }) = self.flight.take() {
            tracing::trace!(?why, fence = sent.fence, "settling one late title flight");
            let runs: Vec<&HostList> = lists.iter().map(Detached::host).collect();
            self.host.settle_title(&runs, sent, submitted);
            drop(runs);
            lists.into_iter().for_each(Detached::recycle);
        }
    }
}

impl Drop for PendingTitle<'_> {
    fn drop(&mut self) {
        self.complete(SettleWhy::Dropped);
    }
}

#[test]
fn flight_copy_dependencies_preserve_half_open_ranges_and_unknown_writes() {
    let mut w = FlightWrites::default();
    w.add(100, 16);
    w.add(116, 4);
    w.join();
    assert_eq!(w.written, [(100, 120)]);
    assert_eq!(w.copy_blocked(200, 119, 1), Some(SettleWhy::CopySource));
    assert_eq!(w.copy_blocked(110, 200, 4), Some(SettleWhy::CopyDestination));
    assert_eq!(w.copy_blocked(120, 96, 4), None);
    assert_eq!(w.copy_blocked(100, 100, 0), None);
    assert_eq!(w.read_reason(119, 1), SettleWhy::Write);
    assert_eq!(w.read_reason(120, 1), SettleWhy::MemoryWait);
    assert!(w.touches(usize::MAX - 2, 8));
    w.add(usize::MAX - 2, 8);
    assert_eq!(w.read_reason(500, 4), SettleWhy::Unknown);
    assert_eq!(w.blocked_by(&[]), Some(SettleWhy::Unknown));
}
