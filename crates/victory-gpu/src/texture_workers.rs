//! Bounded workers for independent texture preparation. Adapted from the
//! wide_each worker model in cool_fox and VIctory-7900XTX. GPU commands
//! and resource-state transitions remain on the submitting thread.

use std::sync::{Arc, Mutex, OnceLock, mpsc};

type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    jobs: mpsc::Sender<Job>,
    helpers: usize,
}

fn choose_threads(logical: usize, requested: Option<usize>) -> usize {
    let logical = logical.max(1);
    let auto = if logical < 6 { 1 } else { (logical / 4).clamp(2, 4) };
    requested.unwrap_or(auto).clamp(1, 8).min(logical)
}

pub(crate) fn configured_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
        let requested = std::env::var("VICTORY_TEXTURE_WORKERS").ok()
            .and_then(|v| v.trim().parse().ok());
        choose_threads(logical, requested)
    })
}

impl Pool {
    fn new(threads: usize) -> Self {
        let (jobs, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut helpers = 0;
        for i in 0..threads.saturating_sub(1) {
            let receiver = receiver.clone();
            match std::thread::Builder::new().name(format!("victory-texture-{i}"))
                .spawn(move || loop {
                    let job = {
                        let receiver = receiver.lock().unwrap_or_else(|e| e.into_inner());
                        receiver.recv()
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                })
            {
                Ok(_) => helpers += 1,
                Err(e) => {
                    tracing::warn!(error = %e, "texture worker unavailable; using fewer workers");
                    break;
                }
            }
        }
        Self { jobs, helpers }
    }

    /// Preserve input order. Finish every scheduled job even when one
    /// panics, before unwinding through the caller's resource lifetimes.
    fn map<T: Send + 'static, R: Send + 'static>(
        &self, input: Vec<T>, threads: usize, operation: fn(T) -> R,
    ) -> Vec<R> {
        let count = input.len();
        let width = count.min(threads).min(self.helpers + 1);
        if width < 2 {
            return input.into_iter().map(operation).collect();
        }
        let mut parts: Vec<Vec<(usize, T)>> = (0..width).map(|_| Vec::new()).collect();
        for (i, item) in input.into_iter().enumerate() {
            parts[i % width].push((i, item));
        }
        let (tx, rx) = mpsc::channel();
        for (i, part) in parts.into_iter().enumerate() {
            let done = tx.clone();
            let job: Job = Box::new(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    part.into_iter().map(|(at, item)| (at, operation(item))).collect::<Vec<_>>()
                }));
                let _ = done.send(result);
            });
            if i + 1 == width {
                job();
            } else if let Err(e) = self.jobs.send(job) {
                // No worker receiver remains. Complete the job inline.
                (e.0)();
            }
        }
        drop(tx);
        let mut results: Vec<Option<R>> = (0..count).map(|_| None).collect();
        let mut failure = None;
        for result in rx {
            match result {
                Ok(part) => {
                    for (at, value) in part {
                        results[at] = Some(value);
                    }
                }
                Err(e) => {
                    if failure.is_none() {
                        failure = Some(e);
                    }
                }
            }
        }
        if let Some(e) = failure {
            std::panic::resume_unwind(e);
        }
        results.into_iter().map(|r| r.expect("each texture job returned")).collect()
    }
}

pub(crate) fn ordered_map<T: Send + 'static, R: Send + 'static>(
    input: Vec<T>, threads: usize, operation: fn(T) -> R,
) -> Vec<R> {
    if input.len() < 2 || threads < 2 {
        return input.into_iter().map(operation).collect();
    }
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(configured_threads())).map(input, threads, operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_reserves_cpu_capacity_and_honors_override() {
        assert_eq!(choose_threads(2, None), 1);
        assert_eq!(choose_threads(8, None), 2);
        assert_eq!(choose_threads(16, None), 4);
        assert_eq!(choose_threads(32, Some(64)), 8);
        assert_eq!(choose_threads(2, Some(8)), 2);
        assert_eq!(choose_threads(8, Some(0)), 1);
    }

    #[test]
    fn worker_results_preserve_order_and_panics_do_not_strand_jobs() {
        let pool = Pool::new(3);
        let expected: Vec<_> = (0..97u64).map(|i| i * i).collect();
        assert_eq!(pool.map((0..97u64).collect(), 3, |i| i * i), expected);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.map(vec![0, 1, 2, 3], 3, |i| {
                assert_ne!(i, 1, "synthetic worker panic");
                i
            })
        }));
        assert!(result.is_err());
        assert_eq!(pool.map(vec![5, 2, 9], 3, |i| i + 1), vec![6, 3, 10]);
    }
}
