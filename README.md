# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can (shaders and pipelines too, see Getting started) and on first use
otherwise. See [docs/architecture.md](docs/architecture.md).

## Status (2026-11-25)

**Not playable yet, because of the framerate: about 1 frame per second in the
world.** Nothing else keeps the game from working: the title boots, runs its
menus at its own 30 fps, loads the world, plays the intro cutscene and
reaches the **prologue**, the mission after it, with the graphics fully
displayed. 

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about a minute) | works |
| The intro cutscene | plays at about 1 fps (0.9-1.0 s a frame, warm cache, quiet machine): textured and lit characters with whole hair, buildings, sky, depth of field, the title card, credits and interface. Missing: ray-traced shadows and indirect light, the local-light shadow mask. Seen in some shots: a pink band over nearer bodies, a stand-in head on a secondary character for a few frames |
| The prologue | reached; the graphics are fully displayed, at about 1 fps |

In the intro cutscene at 1 fps the opening credits land over other shots than
on a console: they follow the frame count, the shots follow the clock.

Tested on one machine: Windows 11, AMD RX 6700 XT (12 GB).

> **Warning.** Entering the world on a real GPU hung it in earlier builds,
> and Windows then reset the display driver (every monitor goes black for a
> moment). The latest runs, several minutes each in the world, did not, but
> it is not ruled out; a run can also end on a GPU fault (`crashes.log`). A
> diagnostics build with `VICTORY_D3D12_WARP=1` renders on the software
> adapter instead (too slow to reach the world in minutes).

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | The scene is drawn and presented: mesh pipelines, indirect execution, views the shaders pick from the heap themselves, title memory read by address, depth written from compute, lists that wait for values other queues write, textures made when a shader first reads them, pipelines pre-built or made by worker threads and kept on disk, depth bounds on supporting adapters (synthetic tests) | The title's ray-tracing structures are never built (no ray-traced shadows or indirect light in the intro cutscene); about 1 s a frame; draws of untranslated shaders are dropped; real queries, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL, ahead of time (`translate --cache`) or at first use, cached on disk. Menus: all. World: all but under 300 of 24,387 programs | Merged geometry stages (195 programs) and under 100 others; fused multiply-adds follow the host GPU; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Menus, loading and the world run with the PC upscaler (`VICTORY_FIDELITYFX_DIR`, about 1 s a frame) | The title's own upscaler DLLs: six of their passes are untranslated, so their picture is wrong |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

To install Rust, use rustup (`winget install --id Rustlang.Rustup -e`, or
the installer from https://rustup.rs). The first `cargo` command in this
repository then fetches the pinned toolchain. Building needs the MSVC C++
build tools: Visual Studio or its Build Tools with the "Desktop development
with C++" workload.

Keep the build paths short. From a checkout about 170 characters deep, the
`blake3` dependency's build script fails in `ml64.exe` with
`A1009: line too long`, even with long paths enabled in Windows, because
its object files land under `target`. Set `CARGO_TARGET_DIR` to a short
absolute path (`$env:CARGO_TARGET_DIR = 'C:\vt'`; `victory.exe` is then
in `C:\vt\release` instead of `target\release`), or check out the
repository nearer the drive's root.

A run in the world takes about 33 GB of Windows commit (RAM plus paging
file) and about 10 GB of video memory. The tested machine has 32 GB of RAM
and a commit limit of about 80 GB. Check **Committed** under **Task Manager →
Performance → Memory**. If `crashes.log` reports Windows error 1455, close
other applications or increase the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe translate --cache --pipelines <game dir>\shaders.rpf <game dir>\<the title's exe> <game dir>\XboxBvhBuilder.dll <game dir>\amd_fidelityfx_upscaler_xbox.dll
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The `translate --cache` step (minutes; again after a change of the shader
translator) translates every shader the title can ask for into the cache
(about 1.3 GB), so a run does not translate them while it plays; without it
they are translated at first use, and their draws wait. With `--pipelines`
it then makes every pipeline in the title's shader pack (inside
`shaders.rpf`) into the host pipeline library (about 27,000 pipelines, 4 MB;
estimated at 20 minutes on 16 threads, but about 2 pipelines a second after a
translator change that touches most programs, which is hours), so a run
compiles none of those either; the library belongs to the GPU and driver it
was made on. Run again, it skips what is there in seconds; stopped
early, it keeps what it made. Without it, a run makes the pipelines it needs
on worker threads and keeps them, so the first runs after a translator
change are slower and show missing draws for a while.

To get into the game, press A (Enter) on each first-boot screen and again on
the start screen. Loading the world takes about a minute; the intro cutscene
plays, then the prologue starts.

Upscaling (Series S mode): the title upscales with AMD FidelityFX. To run
it with the PC build on your GPU instead of the console build, get the
[FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
and point `VICTORY_FIDELITYFX_DIR` (or `[paths].fidelityfx_dir`) at its
`Kits/FidelityFX/signedbin` folder, which holds
`amd_fidelityfx_upscaler_dx12.dll`. Without it, or if that DLL offers no
upscaler for your GPU (the log says so), the title's own DLLs run.

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Set `VICTORY_WINDOW_MAX=1920x1080` before launching to limit the title's
top-level window to that bounding box. Smaller windows keep their size,
larger windows keep their proportions, and the monitor's work area still
limits them. This scales the displayed image; it does not change the
console render resolution or the swap-chain buffers. With the variable
unset, window sizing is unchanged.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm] [--effects]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory translate [paths...] --cache [--pipelines] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

`translate --jobs N` requires a positive worker count. Omit it to use the
default worker count.

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

### Windows shader-test validator

The shader tests use `dxil.dll` to validate Shader Model 6.6 containers.
An older Windows SDK validator can load successfully but reject them with
`Unknown shader model 'ps_6_6'`. If this happens, use a current x64 validator
from [Microsoft's DirectX Shader Compiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases)
or a current Windows SDK. Set its path for the test session:

```powershell
$env:VICTORY_DXIL = 'C:\tools\dxc\bin\x64\dxil.dll'
cargo test -p victory-shader --lib --locked -- --test-threads=1
```

Without this override, the validator loader selects the newest installed
Windows SDK's `bin/<version>/x64/dxil.dll`, then falls back to the DLL search
path when no SDK copy is found. This is a test/offline-validation dependency;
the translated shaders do not require it at runtime.

## GPU descriptor compatibility (experimental)

The runtime probes for a 2,097,152-entry shader-visible view heap; otherwise
it selects the portable 1,000,000-entry capacity. The capacity is fixed
before shader translation and included in shader cache keys and prebuilt
pipeline tags. Mirrored slots, dynamic tables, persistent pairs and null
descriptors occupy separate regions. Oversized title heaps are clipped;
complete remapping of all oversized heaps remains unimplemented.

This follows the heap capacity and cache-key design in
[DoubleGhosty/cool_fox](https://github.com/DoubleGhosty/cool_fox/tree/f4b48ad766ddbec1a1867022cd42f7645b0dd55e),
adapted to the earlier runtime.

## Memory retention (experimental)

The idle CPU buffer cache (snapshots and temporary texture detiling reads)
is bounded by installed RAM, available physical RAM, and available commit.
It is checked at most once per second during submissions or cache use; the default ceiling is 2048 MiB and the normal budget on a 32 GiB
system is at most 512 MiB. Below 1 GiB available physical RAM or commit, idle
snapshots are released. `VICTORY_SNAPSHOT_POOL_MB` overrides the ceiling
(0–2048 MiB, 0 disables idle retention). It does not limit live snapshots
or active temporary reads.

Under pressure (physical RAM available <= 3 GiB or available commit
<= 4 GiB), idle CPU storage is capped at 64 MiB; normal retention resumes
when physical RAM reaches 5 GiB and commit headroom reaches 6 GiB.
Below 1 GiB of either reserve, the idle budget is still zero. Failed OS
memory samples retain the last successful observation. This adapts
[cool_fox residency](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/residency.rs).

After a completed command list is reset, retained scratch chunks are capped
at 4 MiB per list (`VICTORY_SCRATCH_KEEP_MB`, 0–64 MiB), further capped at
256 KiB during pressure. Temporary tiled texture reads reuse the bounded
CPU pool rather than retaining a full allocation per worker.
Active uploads and resources referenced by GPU work are not reclaimed.
No required texture uploads are intentionally skipped.

Diagnostics report a `memory cache census` with working set, private commit,
physical and commit headroom, pressure state, idle CPU buffer bytes/budget,
and the count/logical size of loose shadows. The
logical shadow sizes are not measurements of physical RAM or VRAM residency.
This change does not yet reclaim loose shadows: they may be referenced by
persistent descriptors and page tables. Validate on identical synthetic or
legally supplied workloads before drawing performance conclusions.

The pressure-aware retention policy is informed by the RAM cache pressure
handling in [soldsawn/victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb)
(commit `c09a44e97df4170ef0caa8cc236f7cb46fd1ce90`), with independent changes to
this repository's older snapshot and scratch implementations.

## Exact buffer refresh and completed-submission merge

Within a submission, refreshes now maintain disjoint byte intervals per
host buffer. Overlapping or adjacent views join; unrelated views do not
cause the bytes between them to be refreshed. Subsequent flushes subtract
the already refreshed intervals until a title-memory write invalidates
them. This uses the joined/outside range approach of
[cool_fox](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/shadow.rs)
and [victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb/blob/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90/crates/victory-gpu/src/shadow.rs),
adapted to synchronous submissions.

Copy-back still waits for GPU completion and preserves concurrent title
writes using the original bytewise merge rule. Equal blocks are skipped,
unmixed blocks copied in bulk, and mixed blocks merged byte by byte.
Temporary CPU read buffers reuse the bounded pool and are allocated only
when needed. The mapped-memory checks and write tracking are unchanged.

Experimental `VICTORY_EXACT_BACK=1` ports the exact changed-run search
from cool_fox's selective copy-back. Declared-write pieces are merged
using the existing GPU-wins-on-collision rule, then only changed byte
runs are written to title memory (at most 64 writes per piece; a more
fragmented piece falls back to the original full-piece write). The
temporary before-merge buffer uses the existing bounded scratch pool.
Changed pieces stay dirty for the next refresh, rather than claiming
that a concurrent title write outside the landed ranges was mirrored.
The option is off by default, and does not change marked-page copy-back
or introduce a GPU comparison shader or asynchronous readback.

The reference `CARD_COMPARE` shader, its 64 KiB piece range arithmetic
and CPU hash oracle are ported in the Windows module
`host/card_compare.rs`. The compiler and range tests run in normal CI;
the ignored `card_compare_detects_and_fetches_changed_pieces` test
executes hash initialization, comparison, compare-and-update and
selective readback on a D3D12 adapter.

Experimental `VICTORY_CARD_BUFFERS=1` connects the reference's two-sided
registered Shadow buffers to synchronous title submission: dirty CPU
ranges are uploaded and hashed before execution, and changed GPU pieces
are fetched after the checked fence before the existing CPU copy-back.
The original shared-buffer path remains the default and allocation or
pipeline failures fall back before publication. Limits follow the source
(16 MiB minimum and 512 MiB total for local budgets up to 12 GiB), with
`VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_CARD_BUFFERS_MIN_KB` and
`VICTORY_CARD_BUFFERS_MB` overrides. Admission leaves 20% of the reported
local budget free. The source's shared fetch pool now packs changed
ranges into one persistent readback, growing in 16 MiB increments and
reusing it after each checked fence. Admission reserves space for the
largest copy; if later growth fails, bounded batches reuse that space.
The source's sync/check/fetch command allocator and list reuse is also
connected: one pair per phase, returned only after a checked fence.
The internal lists bypass the unrelated tile-upload path. Experimental
`VICTORY_CARD_COMPARE_RANGES=1` adapts the source's declared-range
planner to the current write marks. It compares only joined whole
64 KiB pieces of bounded declarations plus both GPU write bitmaps;
unbounded root ranges compare their whole copy. Heap-indexed and
title-memory shaders, unknown marks, and undeclared-only submissions
fall back to full comparison. The default remains full coverage.
Experimental `VICTORY_PICKED_WRITES=1` ports the reference's descriptor
store classifier and found-copy discovery in observation-only mode.
A bounded background queue analyzes shaders using this branch's native
IR; the current UAV hulls and existing completed GPU comparison flags
identify possible undeclared writes. Pieces intersecting a declaration
or write mark are excluded, as are ambiguous title-memory submissions.
No extra GPU compare or reduced coverage is introduced.
Experimental `VICTORY_PICKED_WRITES=2` additionally declares the
currently visible UAV hulls of previously discovered, tracked card
copies after the final refresh. Only changes made in that subsequent
submission are merged. Existing declarations, aliased title pages,
ambiguous memory callbacks, held-back writes or unknown/overlapping
GPU marks take precedence: cancel automatic landing and release its
snapshot. This does not retroactively land the original discovery.
Experimental `VICTORY_PICKED_ROTATE=1` (with `VICTORY_PICKED_WRITES=1`
or `2`) ports the reference's once-per-frame rotation scheduler and
coverage tests. `VICTORY_PICKED_MB` sets the logical probe slice (32 MiB
by default). This branch retains one unexplained flagged piece per
candidate until its turn, so a one-time observation is not dropped
between frames. It rotates discovery over completed comparison flags,
not GPU hashing: the normal comparison, readback and merge still run
at their existing coverage. A removed UAV cannot be used for a later
automatic declaration.
Experimental `VICTORY_CARD_LOOK_ONLY=1` uses the source's synchronous
mode 1 compare without overwriting its baseline. The changed pieces
still enter the existing full-coverage or declared-range readback path;
each packed fetch batch records mode 0 rehashes of only the fetched
pieces after its copies. The existing checked fence covers both
operations. Unlike the source's probe-only range, a discovered change
is still fetched into this branch's CPU mirror, so later snapshots
cannot mistake it for a new write. This adds hashing for changed
pieces; it does not yet reduce GPU comparison coverage or implement
probe-only no-readback or async relay. The default is mode 2.
The source's `card_hash_size` optimization is available via
`VICTORY_CARD_HASH_LEN=1`: all comparison, dirty rehash and deferred
fetch-rehash passes use the title buffer's logical length rounded to 16
bytes rather than its 64 KiB-rounded allocation. The allocation and CPU
upload remain full sized. This reduces padding hashes/readback, not
live UAV comparison coverage; the default remains full allocation
because padding-only GPU writes are not mirrored in this optional
mode. The sync list also records all uploads, a single transition batch,
then all independent hash passes, as in the source's card_before.
Experimental `VICTORY_CARD_CHECK_TAIL=1` ports the reference's
CardPlan::check tail recording: prepare the full compare/flag-readback
list before title work, queue it immediately after the title group
with a separate ExecuteCommandLists call (preserving buffer decay),
and use the same checked title fence. Keep its allocator and owners
until completion; the existing fetch and CPU copy-back still follow.
This removes one independent check fence wait on eligible submissions.
With `VICTORY_CARD_COMPARE_RANGES=1` or per-list GPU timing enabled,
use the previous post-fence check path. This is not asynchronous relay:
the next title submission still waits for the current CPU merge.
Experimental `VICTORY_SEND_OWN=1` ports the source's submission-owned
send snapshot and short aligned word scatter shader. Long and unaligned
runs use byte copies; aligned runs up to 256 bytes use packed offset/word
pairs. Every snapshot survives its checked sync fence, then returns to
a bounded 128 MiB idle pool. A 512 MiB active allocation ceiling, memory
pressure or allocation/pipeline failure falls back to existing synchronous
staging copies. This separates ownership for future relay work; no
concurrent title submissions or BAR allocation are enabled.
Experimental `VICTORY_CARD_SEND_ORDER=1` ports the source's SentRuns
and outside-range logic: taking dirty intervals records them against
the upcoming sync fence; readback landing excludes unsent intervals
and sends newer than the fetch fence. Previous entries retire when
the next serial submission has consumed all older fetches. This
protects the CPU staging mirror.
Experimental `VICTORY_CARD_RELAY_MERGE=1` together with
`VICTORY_CARD_SEND_ORDER=1` and `VICTORY_PICKED_WRITES=2` ports
the reference's `merge_piece` and `card_merge_relay` into the
completed fetch path for previously discovered, declared picked
ranges only. GPU-changed bytes enter title memory and the held GPU
baseline; title-only bytes stay in title memory for the next refresh.
Newer sent or unsent CPU ranges are excluded. Never re-upload the
merged GPU bytes or repeat the old picked copy-back. Unknown marks,
overlapping marks, withheld memory and other writes use the existing
serial path. Undeclared parts of a fetched piece still only update
the staging mirror. The same relay-merge option also enables the
reference's `Flights` range ownership: one logical submission number
is passed to regular and picked declarations; duplicate declarations
count once, and a completion releases only one claim. Refresh observes
the counted ranges and snapshot teardown waits for the last owner.
The default keeps the existing serial range set. The source's flight
lifetime tests are adapted to this pending owner.
Experimental `VICTORY_CARD_RELAY_WORKER=1` (with the relay-merge,
send-order and picked-write options above) uses a dedicated
`victory-relay` landing thread. It takes the completed prefix of the
source-style flight queue and lands each checked fetch batch in order,
holding its readback and Shadow owners. The submitting thread may
record the next fetch while the worker merges, but waits for its
receipt before executing into the shared readback pool or publishing
title completion. Worker startup failure falls back to inline landing;
a failure during landing is fatal rather than repeating a partial merge.
Experimental `VICTORY_CARD_RELAY_FETCH=1` with the worker option ports
the source's direct-queue RelayFetch list, allocator, private fence and
event. Each checked fetch increments only its own fence; the send
generation remains the last main-queue fence, since every send is
ordered ahead of this fetch under the submission lock. The worker
receives the actual fetch fence and completed value. Reuse the private
list only after completion; keep the existing packed shared readback
pool and wait for its landing receipt before reuse. Creation failure
falls back to the main-fence path. The event closes with its owner.
Run the ignored `private_relay_fetch_has_own_fence_and_reuses_list`
test on a real adapter. This is a bounded, fenced worker port, not the
source's full asynchronous submission pipeline: title submissions
still do not overlap, and generation retirement and mapped-buffer
access rely on the submission lock.
Experimental `VICTORY_FETCH_CHECK=1` ports the reference's CPU
hash (scalar and AVX2) and folded GPU-flag validation into the fenced
packed fetch path. Widen each readback to the exact 16-byte-aligned
shader hash extent, compare every flagged piece before landing any
bytes of the batch, then use only the original logical intervals
for merge and deferred rehash. A mismatch retries the batch on the
direct queue once and fails closed if it remains inconsistent.
Experimental `VICTORY_RELAY_COPY=1` additionally ports the source's
dedicated high-priority copy queue and matching copy allocator/list.
It requires the private relay-fetch worker path and
`VICTORY_FETCH_CHECK=1`. Deferred compute rehash
(`VICTORY_CARD_LOOK_ONLY=1`) retains the direct queue. The copy queue
waits for the completed main fence; its own fence is checked before
hash validation and landing. A hash mismatch is retried on the direct
queue; a second mismatch fails closed. Queue creation failure falls
back to the private direct fetch. Run the ignored
`relay_copy_queue_reads_completed_direct_writes` test on an adapter
with a high-priority copy queue.
This is still a serialized queue handoff, not the source's concurrent
copy-queue relay. Persistent suspect/refetch tracking and overlapping
title submissions remain unported.
Experimental `VICTORY_LIST_DETACH=1` adapts the reference's Detached
list/recording ownership to this branch's segmented Arc-based queues.
All submitted segments share one owner for uploads and draw state,
including dynamic descriptor slots; repeat submissions of the same
closed recording retain it too. Reset and Release can leave these
owners with the queue. Recycle after checked execution, and keep
resources until the last submitted or title reference is gone. Unlike
the source, retain the closed recording for repeat submissions until
Reset instead of replacing it at submission time. On Reset, prepare
every required replacement before changing the recording; allocation
failure returns E_OUTOFMEMORY without consuming its resources.
The queue remains synchronous. Run the ignored
`detached_segments_survive_title_reset_and_release` test on an adapter.
The default keeps the existing Arc/leave path. The ignored
`carded_shadow_refresh_compare_and_copy_back` and
`picked_writes_land_only_after_discovery` adapter tests can also
be run with `VICTORY_FETCH_CHECK=1`. Run the ignored
`owned_card_send_keeps_original_bytes_after_staging_changes` test with
the card-buffer options and `VICTORY_SEND_OWN=1` on a real adapter.
Run the ignored `card_check_tail_shares_title_fence` test on an adapter
with the card-buffer and tail options, ranges/timing off.
Run the ignored `logical_card_hash_length_excludes_padding_writes`
test with the card-buffer options and `VICTORY_CARD_HASH_LEN=1`.
Run the ignored `deferred_card_hash_does_not_repeat_a_fetched_change`
test with the card-buffer options and `VICTORY_CARD_LOOK_ONLY=1`.
Run the ignored `picked_writes_land_only_after_discovery` test with
the card-buffer options and `VICTORY_PICKED_WRITES=2` on an adapter,
including a separate run with `VICTORY_CARD_SEND_ORDER=1` and
`VICTORY_CARD_RELAY_MERGE=1`, and another with
`VICTORY_CARD_RELAY_WORKER=1` as well, including a separate run with
`VICTORY_CARD_RELAY_FETCH=1` and a separate run with
`VICTORY_RELAY_COPY=1`, `VICTORY_FETCH_CHECK=1`, look-only off.
These optional paths need real-adapter verification.
Loose windows and exact-view arenas stay on the existing path. Run the
ignored `carded_shadow_refresh_compare_and_copy_back` test with
`VICTORY_CARD_BUFFERS=1` and `VICTORY_CARD_BUFFERS_MIN_KB=64` on a real
adapter before using the experimental path in a game.

Diagnostics report the logical refresh gap and already-covered bytes
avoided since the last report. These counters are not physical PCIe
transfer measurements. No new GPU-local mirror, shared-heap mapping,
or asynchronous submission lifetime is introduced here.

## GPU memory budget (experimental)

DXGI local/nonlocal process usage and budget are observed separately.
Pressure queries reuse the last valid segment samples for up to 100 ms;
the eviction pass still requires a fresh local observation.
`VICTORY_TEXTURE_ROOM_PERCENT` (50–97, default 90) selects the local
pressure threshold. The normal upstream texture eviction policy keeps
its five-percentage-point recovery margin. Scratch and idle texture upload
allocations are trimmed under CPU or DXGI pressure, never during active GPU
work.

A positive `VICTORY_TEXTURE_BUDGET_MB` opts into a soft logical sampled
texture budget. `VICTORY_CARD_ROOM_FOR_TARGETS_MB` selects outstanding
render-target headroom (default 3072 MiB when the sampled budget is set,
limited to one third of the reported local budget). Only upstream
evictable textures are reclaimed; required resources are not rejected.
Logical allocation counters are not measurements of resident VRAM.
A positive sampled texture budget also opts into the allocation-time MIP
profile below. Without that budget or a MIP profile, sizes are unchanged.

## Low-VRAM texture profile (experimental)

`VICTORY_TEXTURE_DROP_MIPS=0` (default) keeps the original texture sizes.
`1` omits the largest mip of eligible sampled 2D textures, and `2`–`4`
omit additional levels. `auto` selects one omitted level on a detected
adapter with up to 8 GiB dedicated video memory, or zero when the adapter
is unknown or has more memory. In `auto` mode, or with a positive
`VICTORY_TEXTURE_BUDGET_MB`, a new eligible texture may omit more levels
(up to four) if its estimated allocation exceeds current sampled headroom.
The baseline is process-wide; no live texture changes shape. Eligible textures have at least five mips and
512 texels on the shorter side. Render targets, depth, UAV, MSAA and
incompatible formats are excluded. Eligible texture chains typically
use about a quarter of their original host bytes when one mip is omitted,
not a quarter of total VRAM. This trades texture detail for memory.

SRV mip ranges, tiled upload source mips, array slice indices, texture
copies, alias transfers and readbacks map to retained levels. Copies or
readbacks of omitted levels are skipped; resolves involving shortened
textures are unsupported. Full title-memory layout and tracking remain
intact. Existing full-resolution copies and known writable aliases stay
full-sized. The CPU may still read the full tiled allocation.

The mip mapping follows
[fxceless/victory-fork PR 610](https://github.com/fxceless/victory-fork/pull/610)
and [soldsawn/victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb/tree/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90).
Validate both modes, including retained array uploads, before deployment.

## Bounded texture preparation

The existing submission-owned upload path uses the fork's persistent CPU
workers for changed textures. `VICTORY_TEXTURE_WORKERS` overrides the
automatic worker count (1–8, limited to logical CPUs). Temporary reads
are batched in groups of up to 32 textures and 128 MiB of source spans;
a larger individual texture is processed alone. Under RAM pressure work
is serial. Tracked reads use the existing checked copy; untracked reads
are serial unless `VICTORY_PARALLEL_UNTRACKED=1` opts into owned kernel
snapshots. GPU recording, upload lifetimes and fence ordering are unchanged.

Independent buffer copies use the same bounded workers when at least
four jobs are ready and CPU memory is not pressured. Exact uncovered
intervals for each Shadow remain serial. An unmarked GPU write refreshes
only the copy whose snapshot it needs; marked writes keep the upstream
page-arm shortcut. All jobs are joined before title writes or GPU execution.

Tracked texture preparation reads only dirty retained source spans into
temporary CPU scratch, sharing packed-tail reads and falling back to the
whole layout when needed. Failed detiles are retried even if the dirty
tracker is clean; partial uploads are never published.

`VICTORY_PARTIAL_UNTRACKED_UPLOADS=1` also selects changed retained spans
for untracked textures (up to 256 subresources), using content hashes.
It still reads and hashes the whole source, but avoids unchanged
detiling and GPU copies. The default untracked upload path is unchanged.

## Experimental bindless lookahead and ready lists

`VICTORY_HEAP_WORKER=1` reverse-ports the earlier bounded bindless
worker (ac64cdb1) to the current submission-owned upload allocator.
Weak heap requests are deduplicated in a queue of eight, and work is
rotated through at most 32 entries. A pass is bounded by a combined
16 MiB source/private upload allocation budget (4 MiB source per
texture), with an 8 ms soft time limit between entries. A busy
submission lock or memory pressure skips speculation. Untracked source
preparation uses an owned checked snapshot and tentative digest state;
the submitter must re-read and verify its full hash before publishing
that state or using the prepared bytes.

`VICTORY_HEAP_READY_LISTS=1` additionally ports e6be9a0c's closed-list
handoff. Up to eight prepared batches, each owning its private upload
buffers, are kept; age over 250 ms or pressure discards them at the next
worker or submission pass. Closed-list batches contain only fully
tracked sampled textures. The submitter validates every destination,
staging owner, refresh revision and clean tracker or checked digest.
Discarded tracked uploads force a mandatory full retry; discarded
untracked preparations leave their prior digests unchanged. No private
buffers are recycled with the submission pool. Accepted lists precede
the normal upload and title lists and retain owners through the fence;
a failed fence wait stops the run instead of recycling live buffers.

As in ff33360, CPU lookahead remains available with
`VICTORY_SRV_MIP_UPLOADS=1`, but closed speculative GPU lists are
bypassed. The submitter selects the actual union from the private
prepared dirty set and acknowledges only copied levels, retaining
unseen MIPs for later requests. Partially tracked sources use the
ordinary fallback. No asynchronous GPU execution or deferred descriptor
publication. Both options are off by default pending Windows/GPU validation.

## Experimental SRV MIP upload selection

`VICTORY_SRV_MIP_UPLOADS=1` reverse-ports the conservative SRV range
selection in ff33360. For sampled single-slice RGBA8 2D textures, heap
SRV masks use the same retained-MIP shift as the real host descriptors.
Duplicates, rewrites, multiple heaps and aliases contribute to the union
before the submission prepares a texture. Initial uploads cover all MIPs;
default or unsupported SRVs and explicit texture copies keep the
full-range request. The placed-table SRV path also adapts cool_fox's
`upload_from`: per-list requests of the same staging owner union their
ranges, then merge across lists and bindless heaps. A whole-resource
copy request dominates every narrower request. Hidden dirty levels stay pending even after dirty
pages are consumed or a clean content hash is observed. Since current
upload buffers are submission-owned, their source bytes are read again
when a pending MIP becomes visible rather than retaining stale staging.

The existing bounded workers, packed source-span handling, full-size
staging allocations, resource lifetime and submission fence remain.
This is not sparse residency, nor asynchronous lookahead. It is disabled
by default pending Windows GPU readback and mixed copy/write validation.

## Experimental tiled sampled textures

Experimental `VICTORY_TILED_TEXTURE_POOL=1` ports the physical tile
allocator and MIP-group mapping mechanism of
[cool_fox tiles.rs](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/tiles.rs).
On tier-2 adapters, eligible single-slice RGBA8 sampled textures with
multiple retained MIPs and 1–64 MiB of tile backing use reserved
resources, 64 KiB tiles, and a pool of up to sixteen 64 MiB heaps.
Standard MIPs are independent groups; a packed tail is indivisible.
Free runs coalesce on return and completely free heaps are trimmed
under memory pressure. A private COM owner retains the physical heap
leases for the full lifetime of the resource, including descriptor
slots, recorded lists, and retired descriptors. All allocated texels
are initialized to zero after an aliasing barrier on a separate fenced
queue, before the resource is published. Castable formats are
preserved or the existing placed/committed allocation path is used. Reserved
resources do not participate in committed-resource Evict/MakeResident.

The production path maps **all retained MIPs** up front. The incremental
mapping core now ports cool_fox's per-group admission: requesting any
packed level selects the entire tail; already mapped groups are not
remapped; newly leased groups are initialized before their mask is
published. The same resource-lifetime COM owner retains later leases.
An isolated ignored Windows GPU test
(`sparse_group_expansion_preserves_existing_texels_and_leases`) creates
a partial map, writes a coarse texel, expands to another MIP, then checks
readback, accounting and lifetime. This test must be run on a tier-2
adapter before considering live activation.

The reference's backed GPU-write mask is connected to whole and per-MIP
copies, resolves and alias copies, retaining the whole-resource
host-written guard. Each recorded request retains the actual destination
host copy. At submission, merge the MIP masks, claim and back missing
groups, zero only new backing and complete the mapper fence before any
title list runs. Restore the previous resource state. Failed admission
releases tentative claims and stops submission rather than silently
dropping GPU writes. Resetting an unsubmitted list allocates nothing.
The GPU-authored mask becomes visible only after the checked title
fence, before on-read descriptor creation. The isolated ignored GPU test
uses this submission path and exercises reversed list order and
expansion without clearing the previous write.

The reference's `show_from` clamp is shared by initial creation and
fenced re-publication of existing placed 2D SRVs. A completed card write
that changes the first shown level triggers a scan of all live view
heaps, checking the record generation and exact retained host resource
before rebuilding a matching slot. Stale or failed publication uses the
ordinary null/read retry path. No new host texture is created by this
pass. Fresh dynamic tables retain their recorded placed 2D SRV words and
weak identity of the actual host copy. A deferred submission callback
refreshes the descriptor if its first shown level changed since the
table was made or last submitted. It rechecks the heap identity, live
dynamic allocation and retained resource; it must not resolve a
different title texture or use a recycled slot. A list reset discards
its callback. Other SRV forms and changes inside a running submission
still require coordination before live sparse admission.

`VICTORY_TILED_MIP_JUDGEMENT=1` (with the tile pool enabled) also ports
the reference's placed-view first-level and next-start judgement and
title-held-level calculation. A sparse per-heap index collects placed
view hints, merged across heaps; retained source spans, the packed
source block and page commitment are checked for sampled tiled textures
during submission preparation. Changed masks are logged at DEBUG as
`observed title-held MIPs (advisory)`. These are inferred from title
descriptors and memory commitment, not measured shader MIP access.
Unsupported layouts conservatively report all retained levels.

The observed mask is deliberately separate from the active held mask.
Production still fully backs and initializes every retained MIP, so
its first shown level is zero. The observation does not change uploads,
allocation or sampling. The reference `uploaded_on` policy is ported
as `admit_held`: under pressure, held source levels can wait while
card-authored levels cannot, and existing tile leases are retained.
Production creation uses this policy with the full mask. The ignored
`held_mip_admission_waits_for_room_and_keeps_existing_backing` GPU test
exercises sparse admission and expansion with simulated room changes.
The common upload recorder now retains the exact destination owner
and MIP mask for ordinary, ahead and baked texture uploads. Only lists
actually submitted request missing tile groups; newly mapped backing
is initialized before state preludes and GPU execution. After the
checked copy fence, the source-completed mask is updated separately
from GPU-authored levels and existing placed SRVs are republished if
their first shown level changed. The baked-upload path uses the same
admission/completion helpers. Mandatory recorded copies fail closed if
backing cannot be obtained rather than acknowledge missing writes.
The isolated ignored GPU test
`sparse_source_upload_is_published_only_after_the_copy_fence` submits
test texel copies tagged as source uploads, checking abandoned recording,
partial expansion, completion masks and readback.

This connects recorded source uploads to the sparse mapper; it does
not activate live partial allocation. The reference's one-second
waiting-source retry is adapted for nonmandatory bindless refreshes:
before detiling, missing backing is admitted only when there is room.
A failed admission leaves the dirty tracker and hashes untouched and
waits before retrying. Successful new backing forces a fresh source
read, invalidates untracked digests and older speculative revisions
while preserving pending levels; no upload buffer is retained during
the wait. Allocation-only held-level admission never publishes a MIP:
a packed sibling's zero initialization is not a completed source
upload. The full-chain production fallback explicitly keeps its
zero-initialized visibility. Explicit table/copy
requests and recorded GPU writes retain mandatory admission. Ahead and
baked uploads decline incomplete backing so they cannot consume the
waiting source's dirty state.

The SRV clamp now limits a partial descriptor to a contiguous visible
run inside its requested MIP range, using the reference LOD adjustment.
A request with no overlap, or an implicit/unsupported partial view,
takes the existing null-view fallback. The complete visibility mask,
not only its first bit, invalidates existing placed and recorded table
2D SRVs when a hole fills. The full-chain path keeps the original SRV.

Production creation still maps the full chain. The observed held
judgement does not narrow initial allocations or uploads, and the
existing first-upload full-chain rule remains. In-submission descriptor
changes, live partial admission and GPU readback remain unverified;
Windows build and hardware validation are required before activation. Like the reference, mapped tiles are not returned while
the resource lives. This option is disabled by
default: the dedicated initialization fence and zero fill may outweigh
pooling savings; validate both correctness and performance on Windows.
The upstream first-use texture creation and whole-texture descriptor
invalidation/eviction remain in use. Placed and reserved resources are
excluded from the committed-resource residency state machine below.

## Experimental deferred descriptor upload

`VICTORY_DEFERRED_DESCRIPTOR_UPLOADS=1` adapts the synchronous
`prepare_bake_slot` / `publish_uploaded_slot` path of
[DoubleGhosty/cool_fox](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/descriptor/host.rs).
After a shader first reads an unmade placed sampled SRV, eligible fully
tracked textures are prepared and their full retained chain uploaded
before publishing the descriptor. Batches are limited to 32 slots and
16 MiB of combined source and staging bytes, deduplicated by host copy.
The existing direct queue and an explicitly checked fence settle the
upload without recursively submitting or recycling active staging.
The slot's heap identity, generation and actual host copy are checked
again before publication. Failed preparation or stale generation
does not publish the old descriptor; a later read retries it.
Writable, alias-written, untracked, oversized, pressure-limited and
other ineligible views retain the existing on-read path.

The default bake is synchronous. With both
`VICTORY_DEFERRED_DESCRIPTOR_UPLOADS=1` and
`VICTORY_ASYNC_DESCRIPTOR_UPLOADS=1`, the reference's fence-delayed
`texture_slots_flip` is adapted to one outstanding direct-queue batch.
The prior title submission's copy-backs and completion callbacks run
before baking its source bytes. Eligible slots stay null until the
bake fence completes. The next title submission settles that fence,
then republishes only matching live heap identities, record generations
and exact host copies, before eviction, heap sync or staging recycling.
The pending list retains its command allocator, upload buffers and
texture owners. Ahead workers skip preparation while a bake is pending;
the idle batch remains bounded even if no further title submission runs.
Ineligible views continue through the original on-read path. This
allows the title CPU to proceed while the bounded bake executes; it
does not allow general title command lists to execute asynchronously.
Both options are disabled by default. Actual game behavior and
performance still need hardware testing.

Dynamic sparse MIP mapping is still not enabled: all retained tile
groups are backed and initialized before a resource is published.

## Experimental committed texture residency

`VICTORY_TEXTURE_RESIDENCY=1` restores the conservative state machine
from 0d427fb4 and ff33360. Independent committed sampled textures
of at least 1 MiB may be explicitly evicted only while never published
to GPU callers, after two seconds under local or sampled residency
pressure. A per-copy gate restores residency before the first GPU-facing
access and permanently excludes exposed copies from explicit eviction.
Exposed cold textures receive LOW priority after five seconds and
return to NORMAL when active or pressure clears. Oldest unpublished
copies are considered first, with at most 128 transitions per second.
Placed, reserved, writable and twin resources are excluded.

`VICTORY_LAZY_RESIDENCY=1` (requires the first switch) also restores
CREATE_NOT_RESIDENT admission for eligible committed fallbacks, retaining
castable formats and the resident fallback on unsupported drivers.
Existing tile and placed allocation paths take precedence. Failure to
restore residency stops the run rather than publishing a nonresident
texture. Logical nonresident bytes are balanced at creation, eviction,
restoration and destruction. Local headroom uses the greater of DXGI
usage and the estimated resident allocation size; the explicit sampled
allocation cap still counts all logical allocations.

The latest renderer already makes host textures on demand and rebuilds
evicted descriptors; this port does not reintroduce eager creation.
Most newly created textures are exposed immediately, so these options
may provide little additional saving or cost more than they recover.
No physical VRAM savings have been measured. Both switches are off by
default pending Windows and GPU validation.

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

The on/off switches below (`...=1`) are off when unset or set to `0`.

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_POKE=<rva>:<byte>,...` | writes a title byte twice a second (a hang watchdog's tick, say) |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_BREAKPOINTS=1` | appends first-chance breakpoints to `crashes.log`, including ones a handler may recover from |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_CAPTURE_WORLD=<list>` | with `VICTORY_CAPTURE`, also saves (and with `VICTORY_DUMP_TARGETS` dumps) the frames at these offsets from the first frame that ran 500 draws (the world's first; its number depends on how fast the title loaded) |
| `VICTORY_DUMP_WRITERS=<hash>,...` | with the two above, also the textures a pipeline containing a listed shader hash writes, the first frame it runs, and for its buffers a log line (dwords changed, dwords not zero) |
| `VICTORY_PEEK=<hex address>[:<bytes>],...` | in listed frames, those bytes of title memory logged after each submission. Listed frames also log targets of 16 texels or fewer as values, each table view's view words and whether its host slot is null, the host copy of every buffer of 64 bytes or fewer a draw reads (at submission and after its list), and each 1D or 3D texture a dispatch writes, read back right after it |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | skip host pipeline creation and omit draws and dispatches, including indirect ones, when the pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands; a device removal names the list it happened in and lists all its commands |
| `VICTORY_TRACE_SHADERS=<hash>,...` | with `VICTORY_D3D12_DRED`, every draw and dispatch of a listed shader logged with what it binds (user data, heap slots, root views and their first dwords; the first 500, and all of them in listed capture frames); the records of its indirect executions are read back and logged in every frame (records with work, the first four, the frame) |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_PIPELINE_LOG=<file>` | a line per title pipeline created (key, shaders, state), first used, and made or loaded (library name); `translate --cache --pipelines` writes the last kind too |
| `VICTORY_PREBUILD_KEYS=<file>` | `translate --cache --pipelines` makes nothing and lists each pack record's key, shaders and state (or why it is skipped) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

Every build appends unhandled exceptions, panics, and host GPU failures to
`crashes.log` in the title's data folder.

For a startup breakpoint (`0x80000003`), run the diagnostics build with
`VICTORY_CRASH_BREAKPOINTS=1`. Keep the console output along with the new
`crashes.log` entries. A host DLL refusal includes the caller stack in the
console output. The refusal is expected for DLLs outside the host policy,
including `rpcrt4.dll`.

On Series 1.0.2.1, a missing `appdata0.rpf` reproduced the breakpoint at
`+0x2237178` before the first window. Restore the archive from the same
package beside the title executable. `run --execute` checks that it can
read the RPF9 header before starting this build.

To compare hardware and software rendering, repeat with
`VICTORY_D3D12_WARP=1` and check the console for
`host device on the software adapter (WARP)` to confirm the WARP device was
created. Compare the exception's module and offset, stack frames, and
preceding messages across the two runs.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
