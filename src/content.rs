//! Content selection: which update, which DLC, out of everything installed.
//!
//! This replaces the selection half of `RegisteredCache::ProcessFiles` in
//! `src/core/file_sys/registered_cache.cpp`.
//!
//! # The defect it replaces
//!
//! The C++ code builds `std::map<u64, CNMT>` and fills it with
//!
//! ```cpp
//! out_meta.insert_or_assign(title_id, std::move(cnmt));
//! ```
//!
//! keyed on the *title id*, which for a patch is the same id at every version. So when two
//! updates for one game are installed — the ordinary case after the user installs v1.1 over
//! v1.0 and the old NCAs were never removed — **whichever CNMT the directory walk reached last
//! wins**. Directory order is filesystem order, so the game boots on an arbitrary version. Users
//! see this as "the update didn't apply", or as a title showing the new version number while
//! running the old code.
//!
//! # What this module does instead
//!
//! [`select`] takes every CNMT found and returns a [`Selection`] with one winner per title,
//! chosen by explicit rules, plus an account of everything it rejected and why:
//!
//! 1. For each title id, the **highest `version`** wins. Ties are impossible among distinct
//!    installs of the same content (a repeated id/version pair is the same content) but are still
//!    resolved deterministically by keeping the first in a stable sort, and reported as
//!    [`Rejected::Duplicate`] for the rest.
//! 2. **Application vs Patch vs AddOnContent are different slots.** A patch does not compete with
//!    the base game; it overrides it. Both are retained.
//! 3. **A patch whose base title is absent is retained but flagged** ([`Warning::OrphanPatch`]),
//!    because it cannot be applied and the user needs to know the base is missing.
//! 4. **A DLC whose base title is absent is flagged** the same way, since it will never load.
//! 5. Nothing is silently dropped: every input entry appears in exactly one of
//!    [`Selection::chosen`] or [`Selection::rejected`].
//!
//! # Title id arithmetic
//!
//! Nintendo assigns related content ids inside one `0x2000`-aligned block. The constants are
//! Citron's own, from `src/core/file_sys/common_funcs.h`:
//!
//! ```text
//! BASE_TITLE_ID_MASK = 0xFFFFFFFFFFFFE000   (the low 13 bits are content type/index)
//! AOC_TITLE_ID_OFFSET = 0x1000
//! AOC_TITLE_ID_MASK   = 0x7FF
//!
//! base     = 0x0100000000010000    (application)
//! patch    = base + 0x800          (exactly one update per game)
//! dlc[n]   = base + 0x1000 + n     (n = 0..0x7FF, all inside the same block)
//! ```
//!
//! So a single mask — `& 0xFFFFFFFFFFFFE000` — recovers the application for an application, a
//! patch and a DLC alike. An earlier version of this module used `& !0xFFF` and subtracted
//! `0x1000` for DLC; both were wrong, and the test that compares against the real layout caught
//! it. Do not "simplify" this back into per-type branches: there is nothing to branch on.

use std::collections::BTreeMap;
use std::fmt;

/// Mask that keeps the base application id: `0xFFFFFFFFFFFFE000`. From Citron's
/// `BASE_TITLE_ID_MASK`.
pub const BASE_TITLE_ID_MASK: u64 = 0xFFFF_FFFF_FFFF_E000;

/// Where a game's add-on content block starts, relative to the base id. From Citron's
/// `AOC_TITLE_ID_OFFSET`.
pub const AOC_TITLE_ID_OFFSET: u64 = 0x1000;

/// Mask that recovers the DLC index from an add-on title id. From Citron's `AOC_TITLE_ID_MASK`.
pub const AOC_TITLE_ID_MASK: u64 = 0x7FF;

/// What a CNMT describes itself as.
///
/// Mirrors `TitleType` in `src/core/file_sys/cnmt.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TitleType {
    /// The game itself.
    Application,
    /// A title update (patch).
    Patch,
    /// Add-on content.
    AddOnContent,
    /// Delta fragment of a patch.
    Delta,
}

impl TitleType {
    /// Read the `u8` discriminant from a CNMT header.
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0x80 => Self::Application,
            0x81 => Self::Patch,
            0x82 => Self::AddOnContent,
            0x83 => Self::Delta,
            _ => return None,
        })
    }

    /// The `u8` discriminant as it appears in a CNMT header and in the C ABI.
    ///
    /// **Do not use `self as u32`.** The enum has no explicit discriminants, so a cast yields the
    /// variant index (0..=3) and the C side, which compares against `0x80..=0x83`, silently sees
    /// every entry as unknown. That bug was caught by the C++ integration check, not by the Rust
    /// tests, which is exactly why that check exists.
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Application => 0x80,
            Self::Patch => 0x81,
            Self::AddOnContent => 0x82,
            Self::Delta => 0x83,
        }
    }
}

/// One installed CNMT, reduced to what selection needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The title id the CNMT names.
    pub title_id: u64,
    /// The version the CNMT declares.
    pub version: u32,
    /// What kind of content this is.
    pub title_type: TitleType,
    /// A caller-chosen stable identifier for the underlying file (an NCA id, a path hash, an
    /// index). Carried through so the C++ side can map a winner back to its file without a
    /// second lookup. Never used for ordering.
    pub source: u64,
}

impl Entry {
    /// The application this content belongs to.
    pub fn base_title_id(&self) -> u64 {
        base_title_id(self.title_id, self.title_type)
    }
}

/// The base application id for a title id of a known kind.
///
/// One mask covers every kind: the application, its patch (`+0x800`) and all of its add-on content
/// (`+0x1000 + n`, `n <= 0x7FF`) share the same high bits once the low 13 are cleared. See the
/// module docs for the constants and where they come from.
///
/// `title_type` is accepted so callers cannot accidentally use this on an id whose kind is unknown;
/// it does not change the result.
pub fn base_title_id(title_id: u64, title_type: TitleType) -> u64 {
    let _ = title_type;
    title_id & BASE_TITLE_ID_MASK
}

/// The DLC index of an add-on title id, or `None` for anything that is not in the AOC block.
pub fn aoc_index(title_id: u64, title_type: TitleType) -> Option<u16> {
    if title_type != TitleType::AddOnContent {
        return None;
    }
    let low = title_id & !BASE_TITLE_ID_MASK;
    if low < AOC_TITLE_ID_OFFSET {
        return None;
    }
    Some((low - AOC_TITLE_ID_OFFSET) as u16 & AOC_TITLE_ID_MASK as u16)
}

/// Why an entry lost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejected {
    /// A higher version of the same title id was chosen.
    Superseded {
        /// Version of the entry that won.
        chosen_version: u32,
    },
    /// Same title id and version as an entry already accepted — the identical content installed
    /// twice.
    Duplicate,
    /// The title type discriminant was not one the format defines; filtered before selection.
    UnknownType {
        /// The raw byte from the CNMT.
        raw: u8,
    },
}

/// Something the user should be told even though selection succeeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Warning {
    /// A patch is installed but its base application is not.
    OrphanPatch { title_id: u64, version: u32 },
    /// Add-on content is installed but its base application is not.
    OrphanDlc { title_id: u64, version: u32 },
}

/// The result of a selection pass.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Chosen entry per title id, at most one application, one patch and one DLC each.
    pub chosen: BTreeMap<u64, Entry>,
    /// Everything not chosen, with the reason. Every input entry is here or in `chosen`.
    pub rejected: Vec<(Entry, Rejected)>,
    /// Non-fatal problems worth surfacing.
    pub warnings: Vec<Warning>,
}

impl Selection {
    /// The chosen application for `base`, if any.
    pub fn application(&self, base: u64) -> Option<&Entry> {
        self.chosen
            .get(&base)
            .filter(|e| e.title_type == TitleType::Application)
    }

    /// The chosen patch for `base`, if any. This is the one a launcher must apply.
    pub fn patch(&self, base: u64) -> Option<&Entry> {
        let patch_id = base | 0x800;
        self.chosen
            .get(&patch_id)
            .filter(|e| e.title_type == TitleType::Patch)
    }

    /// All chosen DLC for `base`, ordered by title id.
    pub fn dlc(&self, base: u64) -> Vec<&Entry> {
        self.chosen
            .values()
            .filter(|e| e.title_type == TitleType::AddOnContent && e.base_title_id() == base)
            .collect()
    }

    /// Total entries considered.
    pub fn total_inputs(&self) -> usize {
        self.chosen.len() + self.rejected.len()
    }
}

/// Choose one entry per title id: highest version wins, ties are reported, nothing is lost.
///
/// The input order does not affect the result: two permutations of the same entries produce the
/// same [`Selection`]. That property is the actual fix, and it is tested.
pub fn select(entries: impl IntoIterator<Item = Entry>) -> Selection {
    let mut by_title: BTreeMap<u64, Vec<Entry>> = BTreeMap::new();
    for entry in entries {
        by_title.entry(entry.title_id).or_default().push(entry);
    }

    let mut chosen: BTreeMap<u64, Entry> = BTreeMap::new();
    let mut rejected: Vec<(Entry, Rejected)> = Vec::new();

    for (title_id, mut candidates) in by_title {
        // Sort by version descending; ties broken by `source` ascending so the outcome never
        // depends on iteration order. `source` is stable per file, so this is deterministic
        // across runs even though it is arbitrary in meaning — and ties are reported anyway.
        candidates.sort_by(|a, b| b.version.cmp(&a.version).then(a.source.cmp(&b.source)));

        let mut iter = candidates.into_iter();
        // There is always at least one: the map was built from the entries themselves.
        let winner = match iter.next() {
            Some(w) => w,
            None => continue,
        };
        let winner_version = winner.version;
        let winner_source = winner.source;
        for loser in iter {
            let reason = if loser.version == winner_version && loser.source == winner_source {
                Rejected::Duplicate
            } else {
                Rejected::Superseded {
                    chosen_version: winner_version,
                }
            };
            rejected.push((loser, reason));
        }
        chosen.insert(title_id, winner);
    }

    // Orphans: a patch or DLC with no application to attach to.
    let mut warnings = Vec::new();
    for entry in chosen.values() {
        if entry.title_type == TitleType::Application {
            continue;
        }
        let base = entry.base_title_id();
        let has_base = chosen
            .get(&base)
            .is_some_and(|e| e.title_type == TitleType::Application);
        if has_base {
            continue;
        }
        warnings.push(match entry.title_type {
            TitleType::AddOnContent => Warning::OrphanDlc {
                title_id: entry.title_id,
                version: entry.version,
            },
            TitleType::Patch | TitleType::Delta => Warning::OrphanPatch {
                title_id: entry.title_id,
                version: entry.version,
            },
            TitleType::Application => unreachable!("filtered above"),
        });
    }

    Selection {
        chosen,
        rejected,
        warnings,
    }
}

/// Human-readable summary of a selection, for the one INFO line the emulator logs on refresh.
impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let apps = self
            .chosen
            .values()
            .filter(|e| e.title_type == TitleType::Application)
            .count();
        let patches = self
            .chosen
            .values()
            .filter(|e| e.title_type == TitleType::Patch)
            .count();
        let dlc = self
            .chosen
            .values()
            .filter(|e| e.title_type == TitleType::AddOnContent)
            .count();
        write!(
            f,
            "{} application(s), {} patch(es), {} add-on(s); {} superseded, {} warning(s)",
            apps,
            patches,
            dlc,
            self.rejected.len(),
            self.warnings.len()
        )
    }
}
