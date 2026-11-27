//! Completed-fetch landing worker adapted from the reference's
//! host.rs Line, relay_worker, landable and relay_land.
//! Only already fenced work is admitted: the current renderer has one
//! shared readback pool, one submit lock and no detached title lists.
//! The caller can record the next fetch while the worker lands a batch,
//! but must receive its completion before submitting or reusing memory.

use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::thread::{self, JoinHandle};

type Land = Box<dyn FnOnce() -> u64 + Send>;
type Result = thread::Result<u64>;

struct Flight {
    fence: u64,
    land: Land,
    done: mpsc::SyncSender<Result>,
}

#[derive(Default)]
struct Line {
    line: Vec<Flight>,
    completed: u64,
    stopped: bool,
}

#[derive(Default)]
struct Queue {
    line: Mutex<Line>,
    moved: Condvar,
}

/// Source's rule: land only the completed prefix, in queue order.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// Two flights and what follows each: nothing of the second lands
/// before the card has run it, and nothing behind a flight lands
/// before the flight.
#[test]
fn flights_land_in_the_order_they_were_sent() {
    // Flight (fence 5), its write, a signal, flight (fence 6), its write.
    let line = [Some(5), None, None, Some(6), None];
    assert_eq!(landable(&line, 4), 0);
    assert_eq!(landable(&line, 5), 3);
    assert_eq!(landable(&line, 6), 5);
    assert_eq!(landable(&line, u64::MAX), 5);
    // Signals with no flight ahead land at once; behind one they wait.
    assert_eq!(landable(&[None, None], 0), 2);
    assert_eq!(landable(&[None, Some(9), None], 8), 1);
    assert_eq!(landable(&[], 3), 0);
    // After the first three landed, the rest when its turn comes.
    assert_eq!(landable(&line[3..], 5), 0);
    assert_eq!(landable(&line[3..], 6), 2);
}

pub(super) struct Worker {
    queue: Arc<Queue>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Option<Self> {
        let queue = Arc::new(Queue::default());
        let incoming = queue.clone();
        let thread = thread::Builder::new().name("victory-relay".into())
            .spawn(move || relay_worker(incoming)).ok()?;
        Some(Self { queue, thread: Some(thread) })
    }

    /// Admit a checked fetch. The exclusive submission owner must keep
    /// its lock and not touch the landing mappings until the receipt
    /// has completed. No uncompleted GPU fence is admitted in this port.
    pub(super) fn submit(&self, fence: u64, completed: u64,
        land: impl FnOnce() -> u64 + Send + 'static) -> Receipt
    {
        assert!(fence <= completed && completed != u64::MAX, "relay fetch fence not complete");
        let (done, ack) = mpsc::sync_channel(1);
        let mut line = self.queue.line.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!line.stopped && line.line.is_empty(), "one queued relay fetch");
        line.completed = line.completed.max(completed);
        line.line.push(Flight { fence, land: Box::new(land), done });
        drop(line);
        self.queue.moved.notify_one();
        Receipt { ack: Some(ack) }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        {
            let mut line = self.queue.line.lock().unwrap_or_else(|e| e.into_inner());
            line.stopped = true;
        }
        self.queue.moved.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Source relay_worker: take only the completed prefix under the line
/// lock, land it outside that lock, and tell its waiters in the same
/// order. This branch has no flight that needs its own event yet.
fn relay_worker(queue: Arc<Queue>) {
    loop {
        let items = {
            let mut line = queue.line.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if line.line.is_empty() {
                    if line.stopped {
                        return;
                    }
                    line = queue.moved.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<_> = line.line.iter().map(|flight| Some(flight.fence)).collect();
                let n = landable(&fences, line.completed);
                // Admission above guarantees a nonempty completed prefix.
                assert!(n != 0);
                break line.line.drain(..n).collect::<Vec<_>>();
            }
        };
        for flight in items {
            // A partially landed job must never be retried inline.
            // Report any panic to the waiting owner rather than leave
            // its shared readback lease permanently outstanding.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(flight.land));
            let _ = flight.done.send(result);
        }
    }
}

pub(super) struct Receipt {
    ack: Option<mpsc::Receiver<Result>>,
}

impl Receipt {
    pub(super) fn finish(mut self) -> u64 {
        match self.ack.take().expect("relay receipt").recv() {
            Ok(Ok(written)) => written,
            _ => super::fail(format_args!("relay fetch landing failed")),
        }
    }
}

impl Drop for Receipt {
    fn drop(&mut self) {
        // Even if recording the next fetch unwinds, keep the submitting
        // owner's readback lease until the in-progress worker is done.
        if let Some(ack) = self.ack.take() {
            let _ = ack.recv();
        }
    }
}

/// Start once on explicit opt-in. A worker-start failure is safe to
/// fall back from, since no landing was ever submitted to it.
pub(super) fn worker() -> Option<&'static Worker> {
    static WORKER: OnceLock<Option<Worker>> = OnceLock::new();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_RELAY_WORKER").is_some_and(|v| v == "1")) {
        return None;
    }
    WORKER.get_or_init(|| {
        let worker = Worker::start();
        if worker.is_none() {
            tracing::warn!("could not start relay landing worker; retaining inline landing");
        }
        worker
    }).as_ref()
}

#[test]
fn completed_fetch_worker_acknowledges_after_landing() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let worker = Worker::start().expect("relay test worker");
    let caller = thread::current().id();
    let landed = Arc::new(AtomicUsize::new(0));
    let state = landed.clone();
    let first = worker.submit(5, 5, move || {
        assert_ne!(thread::current().id(), caller);
        state.store(1, Ordering::Release);
        17
    });
    assert_eq!(first.finish(), 17);
    assert_eq!(landed.load(Ordering::Acquire), 1);
    let state = landed.clone();
    let second = worker.submit(6, 6, move || {
        assert_eq!(state.fetch_add(1, Ordering::AcqRel), 1);
        23
    });
    // Drop also waits, so even unwinding cannot release the mapping
    // before the worker has stopped accessing it.
    drop(second);
    assert_eq!(landed.load(Ordering::Acquire), 2);
}
