//! Parallel verification of completed card readbacks.
//! Source: DoubleGhosty/cool_fox@f4b48ad766ddbec1a1867022cd42f7645b0dd55e, land.rs
//! Pool/each, block ordering and jobs; torn.rs Check::look.
//! This branch validates the whole batch before any landing. Keep title
//! merges serial, since physical aliases and guard ownership differ.

use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};

const BLOCK: usize = 64 << 10;
const JOB: usize = 256 << 10;

/// Explicit opt-in: the reference's default depends on safecopy/exact_back,
/// which has a different contract here. No change with the option off.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND").is_some_and(|v| v == "1"))
}

/// How many threads land, the relay's own counted (`VICTORY_LAND_THREADS`).
fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let asked = std::env::var("VICTORY_LAND_THREADS").ok().and_then(|v| v.trim().parse().ok());
        asked.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| n.get() / 4).clamp(1, 4)).clamp(1, 16)
    })
}


/// Threads kept for landing, and the work they are at.
struct Pool {
    state: Mutex<State>,
    more: Condvar,
    idle: Condvar,
    /// One batch at a time.
    turn: Mutex<()>,
    threads: usize,
}

struct State {
    /// What to do with each number below `count`.
    work: Option<Work>,
    next: usize,
    count: usize,
    /// Helpers inside a call.
    running: usize,
    panicked: bool,
}

#[derive(Clone, Copy)]
struct Work(*const (dyn Fn(usize) + Sync));

// SAFETY: the closure may be called from any thread (`Sync`), and
// `Pool::each` keeps it alive until no helper is inside a call.
unsafe impl Send for Work {}

impl Pool {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn new(name: &'static str, threads: usize) -> &'static Pool {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State { work: None, next: 0, count: 0, running: 0, panicked: false }),
            more: Condvar::new(),
            idle: Condvar::new(),
            turn: Mutex::new(()),
            threads,
        }));
        for i in 1..threads {
            // (A helper that is not there is only missed.)
            let _ = std::thread::Builder::new().name(format!("{name}-{i}")).spawn(move || {
                pool.help()
            });
        }
        pool
    }

    fn help(&self) {
        let mut state = self.state();
        loop {
            let Some((work, i)) = take(&mut state) else {
                state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            state.running += 1;
            drop(state);
            // SAFETY: counted in `running`, so `each` has not returned.
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*work.0)(i) }));
            state = self.state();
            state.running -= 1;
            state.panicked |= ran.is_err();
            if state.running == 0 {
                self.idle.notify_all();
            }
        }
    }

    /// Calls `f` with every number below `count`, on the pool's threads and
    /// this one, and returns when every call has.
    fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        if count < 2 || self.threads < 2 {
            (0..count).for_each(f);
            return;
        }
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lifetime is left out only until this call returns, and
        // it does not before no helper is inside `f` and none can enter.
        let work = Work(unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(f) });
        let mut state = self.state();
        (state.work, state.next, state.count, state.panicked) = (Some(work), 0, count, false);
        drop(state);
        if count > self.threads {
            self.more.notify_all();
        } else {
            (1..count).for_each(|_| self.more.notify_one());
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let Some((_, i)) = take(&mut self.state()) else { break };
                f(i);
            }
        }));
        let mut state = self.state();
        // (After a panic here: nothing more is begun.)
        state.work = None;
        while state.running > 0 {
            state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        let panicked = state.panicked;
        drop(state);
        if let Err(panic) = ran {
            std::panic::resume_unwind(panic);
        }
        assert!(!panicked, "a job of landing panicked");
    }
}

/// The next number of the batch, if there is one.
fn take(state: &mut State) -> Option<(Work, usize)> {
    let work = state.work?;
    if state.next >= state.count {
        return None;
    }
    state.next += 1;
    Some((work, state.next - 1))
}


/// Pieces (title block address, bytes), sorted by block. Source jobs
/// keeps work in a title block together.
fn jobs(pieces: &[(usize, usize)]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    let (mut from, mut bytes) = (0, 0);
    for (i, &(at, len)) in pieces.iter().enumerate() {
        let new_block = i == 0 || pieces[i - 1].0 / BLOCK != at / BLOCK;
        if bytes >= JOB && new_block {
            out.push(from..i);
            (from, bytes) = (i, 0);
        }
        bytes += len;
    }
    if from < pieces.len() {
        out.push(from..pieces.len());
    }
    out
}


pub(super) struct HashInput<'a> {
    pub(super) block: usize,
    pub(super) bytes: &'a [u8],
    pub(super) expected: u32,
}

/// Source Check::look hashes whole pieces on landing helpers. Here the
/// checked fence and stable readback lease cover every read. Separate
/// verification from landing; no helper writes GPU or title memory.
pub(super) fn hash_many(inputs: &[HashInput<'_>]) -> Vec<bool> {
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.sort_by_key(|&i| inputs[i].block);
    let keys: Vec<_> = order.iter().map(|&i| (inputs[i].block, inputs[i].bytes.len())).collect();
    let jobs = jobs(&keys);
    if jobs.len() < 2 || threads() < 2 {
        return inputs.iter().map(|p| super::torn::hash(p.bytes) == p.expected).collect();
    }
    let results: Vec<_> = inputs.iter().map(|_| AtomicBool::new(false)).collect();
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new("victory-land", threads())).each(jobs.len(), &|j| {
        for &i in &order[jobs[j].clone()] {
            let input = &inputs[i];
            results[i].store(super::torn::hash(input.bytes) == input.expected, Ordering::Relaxed);
        }
    });
    results.into_iter().map(AtomicBool::into_inner).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn jobs_keep_a_title_block_together() {
        let pieces = [(0, JOB), (BLOCK / 2, 16), (BLOCK, JOB), (2 * BLOCK, BLOCK)];
        assert_eq!(jobs(&pieces), [0..2, 2..3, 3..4]);
        assert!(jobs(&[]).is_empty());
    }

    #[test]
    fn landing_pool_runs_every_job_before_return_and_can_be_reused() {
        let pool = Pool::new("victory-land-test", 3);
        let counts: Vec<_> = (0..97).map(|_| AtomicUsize::new(0)).collect();
        for expected in [1, 2] {
            pool.each(counts.len(), &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(counts.iter().all(|n| n.load(Ordering::Relaxed) == expected));
        }
    }

    #[test]
    fn hash_many_keeps_input_order_and_detects_every_mismatch() {
        let original = vec![0x5a; BLOCK];
        let mut altered = original.clone();
        altered[31] ^= 1;
        let expected = super::super::torn::hash(&original);
        let inputs: Vec<_> = (0..24).rev().map(|i| HashInput {
            block: i * BLOCK,
            bytes: if i % 3 == 0 { &altered } else { &original },
            expected,
        }).collect();
        let got = hash_many(&inputs);
        let wanted: Vec<_> = (0..24).rev().map(|i| i % 3 != 0).collect();
        assert_eq!(got, wanted);
    }
}
