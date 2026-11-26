# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can (shaders and pipelines too, see Getting started) and on first use
otherwise. See [docs/architecture.md](docs/architecture.md).

## Status (2026-11-24)

**Not playable yet.** The title boots, runs its menus at its own 30 fps,
loads the world and shows it: slowly, and with rendering defects.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | drawn and shown: streets, buildings, vehicles, characters, the title card and credits; 1-2 s per frame. Surfaces are unlit, a band of distant geometry is missing, characters come out broken in some shots |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU has hung it in earlier
> builds, and Windows then resets the display driver (every monitor goes
> black for a moment). The latest runs did not, but it is not ruled out; a
> run can also end on a GPU fault (`crashes.log`). A diagnostics build with
> `VICTORY_D3D12_WARP=1` renders on the software adapter instead (too slow
> to reach the world in minutes).

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
| GPU: world | The scene is drawn and presented: mesh pipelines, indirect execution, views the shaders pick from the heap themselves, title memory read by address (ray-tracing structures), depth written from compute, pipelines pre-built or made by worker threads and kept on disk | Lighting (surfaces are unlit) and some missing geometry; frames take 1-2 s; draws of untranslated shaders are dropped; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL, ahead of time (`translate --cache`) or at first use, cached on disk. Menus: all. World: mesh 8,765 of 8,765, vertex/pixel 12,550 of 12,592, compute 2,579 of 2,637 | Merged geometry stages (195 programs) and about 100 others; fused multiply-adds follow the host GPU; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

To install Rust, use rustup (`winget install --id Rustlang.Rustup -e`, or
the installer from https://rustup.rs). The first `cargo` command in this
repository then fetches the pinned toolchain. Building needs the MSVC C++
build tools: Visual Studio or its Build Tools with the "Desktop development
with C++" workload.

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

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
some 20 minutes on 16 threads, most of it the 2,700 compute ones), so a run
compiles none of those either; the library belongs to the GPU and driver it
was made on. Run again, it skips what is there in seconds; stopped
early, it keeps what it made.

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

## Structured-buffer compatibility (experimental)

Untyped structured SRV/UAV descriptors are normalized to raw views with
their original byte extent and stride metadata preserved. The translator
computes byte offsets once and converts raw byte dimensions back to
record counts for structured accesses. The descriptor and refresh paths
use the same backing copy; views off 16-byte alignment use exact copies.

This adapts the no-descriptor-twin approach described by
[avatarmanz/VIctory-7900XTX b68e65e](https://github.com/avatarmanz/VIctory-7900XTX/commit/b68e65e08c4276e2159eb3c23b203a35985a0b7c)
and its registered-buffer regression
[ce42cc7](https://github.com/avatarmanz/VIctory-7900XTX/commit/ce42cc74b27e211f214673e135d87ac071b820d7).
The test harness uses the same raw descriptor representation as runtime.
The existing expected-value stride and gather tests are unchanged. This
is a candidate fix for both failures, not a claim of a verified hardware
result. Full GPU and shader regression testing is required before merge.

## Low-VRAM texture profile (experimental)

`VICTORY_TEXTURE_DROP_MIPS=0` (default) keeps the original texture sizes.
`1` omits the largest mip of eligible sampled 2D textures, and `2`–`4`
omit additional levels. `auto` selects one omitted level on a detected
adapter with up to 8 GiB dedicated video memory, or zero when the adapter
is unknown or has more memory. This is a process-wide choice; no live
texture changes shape. Eligible textures have at least five mips and
512 texels on the shorter side. Render targets, depth, UAV, MSAA and
incompatible formats are excluded. Eligible texture chains typically
use about a quarter of their original host bytes when one mip is omitted,
not a quarter of total VRAM. This trades texture detail for memory.

SRV mip ranges, tiled upload source mips, array slice indices, texture
copies and alias transfers map to retained levels. Copies involving
omitted levels do not copy those levels; resolves involving shortened
textures are unsupported. Texture-to-buffer readback is not implemented
in this older runtime. Full title-memory layout and tracking remain
intact; the CPU may still read the full tiled allocation.

DXGI local and nonlocal process usage and budget are sampled once per
second. `VICTORY_TEXTURE_ROOM_PERCENT` (50–97, default 90) is a pressure
threshold for reducing completed-list idle scratch retention, with a
10-percentage-point recovery margin. This is **not** a hard VRAM cap:
the older runtime creates textures eagerly and cannot safely reject
required allocations. Deferred residency, eviction and queued texture
admission are not included in this change.

The mip mapping follows
[fxceless/victory-fork PR 610](https://github.com/fxceless/victory-fork/pull/610)
and [soldsawn/victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb/tree/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90).
Validate both modes, including retained array uploads, before deployment.

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
CPU pool rather than retaining a full allocation in every texture's Last.
Active uploads and resources referenced by GPU work are not reclaimed.
No required texture uploads are intentionally skipped.

Diagnostics report a `memory cache census` with working set, private commit,
physical and commit headroom, pressure state, idle CPU buffer bytes/budget,
logical staging bytes and the count/logical size of loose shadows. The
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

Diagnostics report the logical refresh gap and already-covered bytes
avoided since the last report. These counters are not physical PCIe
transfer measurements. No new GPU-local mirror, shared-heap mapping,
or asynchronous submission lifetime is introduced here.

## Bounded texture preparation workers (stage 1)

Explicitly referenced texture staging buffers are collected and refreshed
before a memory-writing step or GPU submission. The independent tracked
reads use a persistent CPU worker pool, following the wide_each approach
in [cool_fox](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/shadow.rs)
and [VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX).
Only tracked source memory uses workers; untracked hash reads remain
serial. The submitting thread participates and waits for every job before
recorded GPU work can execute. Existing resource ownership, command
order and GPU fences remain unchanged.

`VICTORY_TEXTURE_WORKERS` selects total preparation threads including the
submitting thread (1–8, bounded by available CPUs). The default is one on
fewer than six logical CPUs and two to four otherwise. RAM pressure
forces serial processing. Batches contain at most 32 textures and up to
128 MiB of logical tiled read sizes; an oversized texture runs alone.
The bound is on these temporary reads, not on all process memory.
Texture preparation timing measures elapsed wall time.

Stage 2 uses the same workers for bindless texture preparation under the
existing view lock, checking shared staging aliases in order. Only
changed textures record copies. Bindless buffer coverage is the exact
union of live descriptor ranges, cached per copy and access kind rather
than a single enclosing hull. GPU submission remains synchronous.

## Parallel buffer refresh

Independent host buffer copies share the bounded preparation worker
pool when four or more copies are pending; ranges of one copy stay on
one worker. This follows the separate-copy refresh and flush_one approach
in [cool_fox host.rs](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/host.rs).
Before a shader-write snapshot, only the corresponding buffer must be
refreshed. Other pending copies stay batched until a memory write or
submission barrier. All worker jobs join before the GPU executes.
`VICTORY_TEXTURE_WORKERS=1` disables parallel preparation of both buffers
and textures; RAM pressure also selects serial processing. Timing counts
elapsed wall time. GPU copy-back is still synchronous.

## Idle GPU upload reuse and allocation recovery

Texture upload buffers use a bounded spare cache, adapted from
[victory-nvidia-8gb staging](https://github.com/soldsawn/victory-nvidia-8gb/blob/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90/crates/victory-gpu/src/texture.rs).
Recorded lists retain staging owners until successful reset, rather than
just their COM buffer. Only fully released staging allocations can be
reused; reuse clears the logical bytes. Entries are rounded to 64 KiB,
at most 32 MiB each; idle retention defaults to 128 MiB / 128 entries.
`VICTORY_STAGING_POOL=0` disables it, and `VICTORY_STAGING_POOL_MB` sets
the idle byte limit (0–512). RAM or DXGI pressure clears idle uploads.

For mapped upload and shared-buffer allocation failures with
E_OUTOFMEMORY, reclaim idle upload/CPU caches and retry once if anything
was freed. Idle GPU retention stays disabled for at least 500 ms after
the failure, following the backoff idea in
[cool_fox residency](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/residency.rs).
Required allocations are not refused merely because of a budget or
backoff. This is not live texture eviction or deferred texture admission.
Diagnostics count idle upload bytes, allocations/reuses and OOM hold-down.

`VICTORY_STAGING_BLOCKS=1` experimentally enables staging suballocation,
adapted from [cool_fox pool.rs](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/pool.rs).
Requests up to 32 MiB use 64 KiB-aligned slices of mapped 64 MiB D3D12
upload blocks, with at most 16 blocks including pending allocations.
Each recorded list retains the staging lease until safe reset; GPU
copy footprints add the slice's base offset, while CPU detiling keeps
relative offsets. Slices are zeroed before reuse. Only entirely idle
blocks count toward the existing idle-byte cap and can be released on
pressure or OOM. New blocks charge their full commit cost and are not
grown under pressure. Larger requests and unavailable blocks fall back
to the original dedicated upload path. This is off by default until
Windows GPU tests validate the shared-buffer path; it does not pool
host texture allocations or title-address upload arenas.

## Dirty texture source spans

Tracked texture updates read and detile only retained mip and array
subresources overlapping dirty pages. Packed tails share a conservative
source span; volumes use the full layout. First use fills every retained
subresource. A failed detile retries even if dirty flags were consumed.
This adapts retained-source handling from
[victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb/blob/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90/crates/victory-gpu/src/texture.rs).
Untracked memory still uses the original whole-layout hash.
Bindless GPU uploads record only pending retained subresources.
Pending changes accumulate across refreshes and are acknowledged on
recording under the synchronous submission lock. Explicit pre-recorded
lists still copy the complete retained chain because the later dirty
ranges are not known at record time. Untracked hash changes also keep
the conservative full-chain path. This reduces logical copy traffic
for tracked bindless resources; actual PCIe savings require profiling.

## Memory-pressure admission and idle staging reclamation

A weak registry of staging owners is swept once per second. A staging
allocation with no recorded-list owner becomes reclaimable after ten
seconds of disuse (one second under RAM/DXGI pressure). Host textures
and descriptor identities are unchanged; the next use rebuilds staging.

New speculative bindless staging allocations reserve at least 512 MiB
of system commit after allocation and require at least 1 GiB before
growing committed upload storage. Reusing an already committed spare
requires only 512 MiB and does not charge the commit estimate again.
The soft submission budget is 16 MiB below 1536 MiB of available commit
or 64 MiB under memory pressure or on hosts with at most 32 GiB of RAM
when live logical staging reaches 1 GiB. Long sweeps resample system
commit at most every 32 ms and tighten their remaining budget as needed.
A single larger first allocation may cross the soft budget only if
commit permits. The 500 ms OOM hold-down pauses speculative work.
Existing uploads and explicitly recorded texture uploads are not throttled. Unserved descriptors are
checked again on later submissions. This is a conservative admission
policy inspired by cool_fox residency.rs, not its full asynchronous
prefetch queue. A fence wait failure now stops submission processing
rather than releasing storage without confirmed GPU completion.

## Soft sampled texture budget and target reserve

Adapted from the room policy in
[victory-nvidia-8gb](https://github.com/soldsawn/victory-nvidia-8gb/blob/c09a44e97df4170ef0caa8cc236f7cb46fd1ce90/crates/victory-gpu/src/resource.rs).
`VICTORY_TEXTURE_DROP_MIPS=auto` or a positive
`VICTORY_TEXTURE_BUDGET_MB` enables allocation-time adaptation.
`VICTORY_CARD_ROOM_FOR_TARGETS_MB` defaults to 3072 MiB, capped at
one third of the local DXGI budget. The outstanding target reserve
reduces room for sampled textures. Eligible new textures use the most
detailed retained chain that fits, or the smallest supported chain
if none fits. Live compatible textures retain their shape. Targets,
depth, UAV and ineligible textures retain their original shape.
Numeric mip profiles without a budget keep their previous behavior.
Required allocations are not refused to enforce a soft budget.

Diagnostics distinguish logical sampled/target allocation bytes from
actual DXGI usage. DXGI observations are cached for 100 ms and retain
the last good local/non-local segment independently on query failure.
When adaptive texture budgeting and experimental residency are enabled,
the outstanding target reserve also triggers cold-resource residency
maintenance before the raw DXGI usage threshold is crossed. Recovery
requires a headroom margin (5% of budget, at most 256 MiB). This is soft
headroom management, not physical VRAM reservation or live tiled residency.

## Experimental sampled-texture residency

`VICTORY_TEXTURE_RESIDENCY=1` enables an opt-in residency sweep. Under
DXGI or configured sampled-texture budget pressure, committed sampled
textures of at least 1 MiB that have **never exposed** their host resource
can be evicted after two seconds and made resident synchronously before
their first host access. Every host pointer/Arc publication and transition
passes a per-texture residency gate. Published resources cannot be
explicitly evicted in this renderer: existing recorded commands and
descriptor slots may still name them. Their paging priority can instead
be reduced after five seconds of inactivity and restored when active or
pressure clears. Writable textures and twins are excluded. At most 128
transitions per second are attempted; unsupported eviction falls back to
a resident resource. Restore failure stops rather than submitting an
unresident resource.

This implements conservative eviction and OS residency priority hints,
not the full lazy resource recreation, tile-level eviction and
descriptor rebuilding architecture in
[cool_fox](https://github.com/DoubleGhosty/cool_fox/blob/f4b48ad766ddbec1a1867022cd42f7645b0dd55e/crates/victory-gpu/src/resource.rs).
Logical evicted allocation bytes are not a measured VRAM reduction.
The default is off until validated on Windows hardware.

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
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands; a device removal names the list it happened in and lists all its commands |
| `VICTORY_TRACE_SHADERS=<hash>,...` | with `VICTORY_D3D12_DRED`, every draw and dispatch of a listed shader logged with what it binds (user data, heap slots, root views and their first dwords; the first 500) |
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
