//! Cost-optimal pipeline: enum grid, anytime frontier, certificates.

use super::*;

/// One `(k, m)` enum-arm task result: `(lde, m, truncated, min-cost find)`.
type EnumArmOutcome = (u32, u32, bool, Option<(usize, SynthResultQ)>);

impl SynthesizerQ {
    /// Dispatch the (k, m ≥ 1) arms under the deadline. Below 1e-7 they
    /// run as sequential lowest-m-first phases (interleaving lets m=2's
    /// fan-out starve the deep m=1 units that hold the decisive finds; the
    /// incumbent carries forward as each phase's prune floor); interleaved
    /// above. A phase whose frontier finishes early donates its leftover
    /// deadline to later phases. Env: CYCLOSYNTH_SEQ_M, CYCLOSYNTH_SEQ_M_SPLIT,
    /// CYCLOSYNTH_SEQ_ROLLFWD.
    pub(crate) fn run_frontier_grouped_by_m(
        &self,
        target: &Mat2,
        tasks: &[(u32, u32)],
        deadline_ms: u64,
        shared_best: &std::sync::atomic::AtomicUsize,
    ) -> (Option<(usize, SynthResultQ)>, Vec<bool>) {
        let seq_m = match std::env::var("CYCLOSYNTH_SEQ_M").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => self.epsilon < 1e-7,
        };
        let mut m_groups: Vec<u32> = tasks.iter().map(|&(_, m)| m).collect();
        m_groups.sort_unstable();
        m_groups.dedup();
        if !seq_m || m_groups.len() <= 1 {
            return self.min_cost_frontier_search(
                target,
                tasks,
                std::time::Duration::from_millis(deadline_ms),
                shared_best,
            );
        }

        let split: Vec<u64> = std::env::var("CYCLOSYNTH_SEQ_M_SPLIT")
            .ok()
            .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
            .unwrap_or_default();
        let equal_share = (deadline_ms / m_groups.len() as u64).max(1);
        let rollfwd = split.is_empty()
            && std::env::var("CYCLOSYNTH_SEQ_ROLLFWD").as_deref() != Ok("0");
        let t_phases = std::time::Instant::now();
        let mut best_fr: Option<(usize, SynthResultQ)> = None;
        let mut trunc_by_task: Vec<((u32, u32), bool)> = Vec::new();
        for (gi, &mg) in m_groups.iter().enumerate() {
            let share = if rollfwd {
                let left = deadline_ms
                    .saturating_sub(t_phases.elapsed().as_millis() as u64);
                (left / (m_groups.len() - gi) as u64).max(1)
            } else {
                split
                    .get(gi)
                    .or(split.last())
                    .copied()
                    .unwrap_or(equal_share)
                    .max(1)
            };
            let group: Vec<(u32, u32)> =
                tasks.iter().copied().filter(|&(_, m)| m == mg).collect();
            let (g_fr, g_tr) = self.min_cost_frontier_search(
                target,
                &group,
                std::time::Duration::from_millis(share),
                shared_best,
            );
            trunc_by_task.extend(group.iter().copied().zip(g_tr));
            if let Some((c, r)) = g_fr {
                if best_fr.as_ref().is_none_or(|(bc, _)| c < *bc) {
                    best_fr = Some((c, r));
                }
            }
        }
        let truncated = tasks
            .iter()
            .map(|t| {
                trunc_by_task
                    .iter()
                    .find(|(tt, _)| tt == t)
                    .map(|&(_, tr)| tr)
                    .unwrap_or(true)
            })
            .collect();
        (best_fr, truncated)
    }

    /// Anytime merged-frontier enum stage (fast path, certify off): the
    /// prefix work-units of every (k, m) arm, tagged with the sound
    /// floor `cost(U_L) + class_cost_lb(d_R)` (one currency across
    /// arms), sorted floor-ascending (k-ascending tie-break: smaller SE
    /// regions drop the incumbent faster), transpose-interleaved across
    /// chunks, and stopped by deadline or floor-exhaustion — both cut
    /// only candidates costing ≥ the incumbent. A large per-prefix node
    /// cap backstops pathological prefixes. `cost_lb(lde_inner)` is NOT in
    /// the floor (unsound — see `prefix_split_search_q`).
    ///
    /// Returns the min-cost find plus a per-level truncation flag
    /// (parallel to `levels`): a level is marked truncated when any of
    /// its units was deadline-skipped, deadline-aborted, or hit the
    /// backstop cap. Conservative over-marking (a walk that finished
    /// cleanly right at the deadline may be marked) keeps the ledger
    /// honest; sound floor-kills are NOT truncation, as today.
    pub(crate) fn min_cost_frontier_search(
        &self,
        target: &Mat2,
        levels: &[(u32, u32)],
        deadline: std::time::Duration,
        shared_best_cost: &std::sync::atomic::AtomicUsize,
    ) -> (Option<(usize, SynthResultQ)>, Vec<bool>) {
        use rayon::prelude::*;

        let q_cost_x2 = self.q_cost_x2;
        let d_target = det_phase_of(target);
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
        let best_cost = shared_best_cost;
        let start = std::time::Instant::now();

        // Backstop node cap per unit — generous (the deadline is the
        // primary stop), but bounded so one pathological prefix can't
        // monopolise the frontier.
        let per_prefix_cap = pass2_prefix_leaf_cap_for(epsilon)
            .saturating_mul(self.optimal_budget_multiplier.max(1));

        // Keep the per-m prefix caches alive for the unit borrows below.
        let level_prefixes: Vec<Arc<Vec<U2Q>>> =
            levels.iter().map(|&(_, m)| build_fgkm_prefix_set(m)).collect();
        let level_costs: Vec<Arc<Vec<(usize, usize)>>> =
            levels.iter().map(|&(_, m)| build_fgkm_prefix_gate_counts(m)).collect();

        #[derive(Clone, Copy)]
        struct PrefixWorkUnit<'a> {
            u_l: &'a U2Q,
            lde_total: u32,
            d_r: u32,
            /// `cost(U_L) + class_cost_lb_half_units(d_R)` — the sound
            /// per-prefix bound from `prefix_split_search_q`, in the half-unit
            /// currency shared by every (k, m) arm.
            floor: usize,
            level_idx: usize,
        }

        let truncated: Vec<AtomicBool> =
            levels.iter().map(|_| AtomicBool::new(false)).collect();

        let mut units: Vec<PrefixWorkUnit> = Vec::new();
        for (li, &(lde_total, m)) in levels.iter().enumerate() {
            // Mirror `run_enum_arm`: m ≥ k arms don't run (the
            // D&C split needs lde_inner ≥ 1 for every prefix).
            if m == 0 || m >= lde_total {
                continue;
            }
            // Same filter the task grid uses: open at ε ≤ 1e-5, else
            // the per-m first-hit defaults.
            let filter = if self.optimal_open_dr_filter {
                Vec::new()
            } else {
                default_inner_det_phase_filter(m)
            };
            let mut cands: Vec<(usize, u32, usize)> = Vec::new();
            for (pi, (u_l, &(t, q))) in level_prefixes[li]
                .iter()
                .zip(level_costs[li].iter())
                .enumerate()
            {
                if u_l.k >= lde_total {
                    continue;
                }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                if !filter.is_empty() && !filter.contains(&d_r) {
                    continue;
                }
                let u_l_cost = 2 * t + q_cost_x2 * q;
                let floor = u_l_cost.saturating_add(
                    crate::synthesis::cost_bound::class_cost_lb_half_units(d_r),
                );
                cands.push((pi, d_r, floor));
            }
            // Right-coset dedup of this arm's post-filter set: one rep
            // per (orbit, k) class ∩ usable, the min-floor member (the
            // floor is the frontier's sort/prune currency).
            // CYCLOSYNTH_ZETA_COSET=0 disables. See `coset_keep_mask`.
            if *ZETA_COSET_DEDUP && cands.len() > 1 {
                let keys = build_fgkm_prefix_coset_keys(m);
                let iw: Vec<(usize, usize)> =
                    cands.iter().map(|&(pi, _, f)| (pi, f)).collect();
                let mask = coset_keep_mask(&iw, &keys);
                let mut it = mask.iter();
                cands.retain(|_| *it.next().unwrap());
            }
            for (pi, d_r, floor) in cands {
                units.push(PrefixWorkUnit {
                    u_l: &level_prefixes[li][pi],
                    lde_total,
                    d_r,
                    floor,
                    level_idx: li,
                });
            }
        }

        if units.is_empty() {
            return (None, truncated.into_iter().map(|t| t.into_inner()).collect());
        }

        // Ascending floor sort, k-ascending tie-break (smaller SE regions
        // complete sooner → incumbent drops faster). Queue dispatch
        // consumes this order directly; the chunked path approximates it
        // with a cost-rank transpose-interleave.
        units.sort_by(|a, b| a.floor.cmp(&b.floor).then(a.lde_total.cmp(&b.lde_total)));
        let n_threads = rayon::current_num_threads().max(1);
        let queue_dispatch = *FRONTIER_QUEUE_DISPATCH;
        if OPTIMAL_PREFIX_INTERLEAVE && !queue_dispatch {
            units = crate::synthesis::stride_interleave(&units, n_threads);
        }
        let chunk = (units.len() / n_threads).max(1);
        let opt_chunk = if OPTIMAL_PAR_MIN_LEN == 0 { chunk } else { OPTIMAL_PAR_MIN_LEN };

        // Per-unit watch: the watcher enforces both the sound
        // incumbent-floor kill (as in `prefix_split_search_q`) and the deadline
        // abort (which additionally marks the unit's level truncated —
        // the watcher is the only place that knows WHY it killed a walk).
        let watches: Vec<PrefixWatch> = units
            .iter()
            .map(|u| PrefixWatch {
                abort: AtomicBool::new(false),
                active: AtomicBool::new(false),
                floor: u.floor,
            })
            .collect();

        let per_unit = |scratch: &mut IntScratch16,
                        idx: usize,
                        u: &PrefixWorkUnit|
         -> Option<(usize, SynthResultQ)> {
            // (a) deadline pre-dispatch: never-started units leave their
            // level truncated (work provably remained at the cutoff).
            if start.elapsed() >= deadline {
                truncated[u.level_idx].store(true, Ordering::Relaxed);
                return None;
            }
            // (b) floor-exhaustion: sound prune, NOT truncation.
            if best_cost.load(std::sync::atomic::Ordering::Relaxed) <= u.floor {
                return None;
            }

            let lde_inner = u.lde_total - u.u_l.k;
            let m_inner = prefix_dag_times_target_q(u.u_l, target);
            let v_inner = unitary_to_uv_zeta(&m_inner);
            let budget_hit = AtomicBool::new(false);
            let u_l_local = *u.u_l;
            let floor = u.floor;
            let should_stop = |_x: &[i64; 16]| -> bool {
                // Incumbent-abort (sound) OR deadline (anytime cutoff).
                // Leaf hits only — a handful per walk, so the Instant
                // read is noise.
                best_cost.load(std::sync::atomic::Ordering::Relaxed) <= floor
                    || start.elapsed() >= deadline
            };
            let w = &watches[idx];
            w.active.store(true, Ordering::Relaxed);
            let sols = find_aligned_lattice_points_auto_prec(
                scratch, v_inner, Some((u.u_l, target)), self.deep_rot_src.as_ref(), lde_inner, epsilon,
                per_prefix_cap, &budget_hit, should_stop,
                Some(&w.abort), None,
            );
            w.active.store(false, Ordering::Relaxed);

            // Backstop cap, or the walk ran into the deadline (whether
            // aborted mid-tree or merely unfinished business remains
            // indistinguishable here — mark conservatively).
            if budget_hit.load(std::sync::atomic::Ordering::Relaxed)
                || start.elapsed() >= deadline
            {
                truncated[u.level_idx].store(true, Ordering::Relaxed);
            }

            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in &sols {
                let u_r = solution_to_u2q_with_det_phase(sol, lde_inner, u.d_r);
                let u_full = u_l_local * u_r;
                let dist = diamond_distance_u2q_float(&u_full, target);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&u_full);
                    let cost = gates_cost(&gates, q_cost_x2);
                    match &best {
                        Some((bcost, _)) if *bcost <= cost => {}
                        _ => best = Some((cost, SynthResultQ {
                            gates: Some(gates),
                            lde: u.lde_total,
                            distance: dist,
                        })),
                    }
                }
            }
            if let Some((c, _)) = &best {
                best_cost.fetch_min(*c, std::sync::atomic::Ordering::Relaxed);
            }
            best
        };

        let make_scratch = || {
            let mut s = Box::new(IntScratch16::new(epsilon));
            s.use_f64_gs = use_f64_gs;
            s.bkz_block_size = bkz_block_size;
            s
        };

        let result_pair: Option<(usize, SynthResultQ)> = with_incumbent_watcher(
            &watches,
            best_cost,
            || start.elapsed() >= deadline,
            |i| {
                truncated[units[i].level_idx].store(true, Ordering::Relaxed);
            },
            || {
                if queue_dispatch {
                    let cursor = std::sync::atomic::AtomicUsize::new(0);
                    let merged: std::sync::Mutex<Option<(usize, SynthResultQ)>> =
                        std::sync::Mutex::new(None);
                    let per_unit = &per_unit;
                    let make_scratch = &make_scratch;
                    let units = &units;
                    let cursor = &cursor;
                    let merged_ref = &merged;
                    rayon::scope(|sc| {
                        for _ in 0..n_threads.min(units.len()) {
                            sc.spawn(move |_| {
                                let mut scratch = make_scratch();
                                let mut local: Option<(usize, SynthResultQ)> = None;
                                loop {
                                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                                    if i >= units.len() {
                                        break;
                                    }
                                    if let Some(r) = per_unit(&mut scratch, i, &units[i]) {
                                        if local.as_ref().is_none_or(|b| r.0 < b.0) {
                                            local = Some(r);
                                        }
                                    }
                                }
                                if let Some(r) = local {
                                    let mut g = merged_ref.lock().unwrap();
                                    if g.as_ref().is_none_or(|b| r.0 < b.0) {
                                        *g = Some(r);
                                    }
                                }
                            });
                        }
                    });
                    merged.into_inner().unwrap()
                } else {
                    units
                        .par_iter()
                        .enumerate()
                        .with_min_len(opt_chunk)
                        .map_init(make_scratch, |s, (i, u)| per_unit(s, i, u))
                        .reduce(
                            || None,
                            |a, b| match (a, b) {
                                (None, x) | (x, None) => x,
                                (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                            },
                        )
                }
            },
        );

        (
            result_pair,
            truncated.into_iter().map(|t| t.into_inner()).collect(),
        )
    }

    /// Certified synthesis: exhaustively enumerate every Clifford+√T circuit
    /// with reduced lde ≤ `k_max` (single unbudgeted shell enumeration per
    /// parity branch — see [`CostCertificate`] for the covering argument),
    /// floor with the Clifford+T baseline, and report a proven optimality
    /// interval.
    ///
    /// Wall time grows exponentially with `k_max`; `certified_optimal`
    /// requires `upper ≤ cost_lb_half_units(k_max + 1)` ≈ k_max, so
    /// closing the certificate for a cost-C circuit needs k_max ≳ C
    /// half-units under the current slope-1/2 staircase; a tighter
    /// staircase shrinks the required horizon proportionally without
    /// touching this code.
    pub fn synthesize_exhaustive_certified(
        &self,
        target: Mat2,
        k_max: u32,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let target = project_det_to_zeta_coset(&target);
        let g = Complex64::from_polar(1.0, PI / 16.0);
        let target_odd: Mat2 = [
            [target[0][0] * g, target[0][1] * g],
            [target[1][0] * g, target[1][1] * g],
        ];

        // T-baseline floor only when the target's det class is even:
        // Clifford+T determinants are even ζ₁₆ powers, so an odd-class
        // target would make the baseline sweep its whole lde range
        // rejecting every prefix.
        let d_even = det_phase_of(&target).is_multiple_of(2);
        let baseline: Option<(usize, SynthResultQ)> = if !d_even { None } else {
            crate::synthesis::clifford_t::SynthesizerT::new(self.epsilon)
                .synthesize(target)
                .and_then(|r| {
                    // NaN-safe reject: `!(d < eps)` also rejects a NaN distance,
                    // unlike `d >= eps`.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(r.distance < self.epsilon) {
                        return None;
                    }
                    r.gates.map(|gs| {
                        let c = gates_cost(&gs, self.q_cost_x2);
                        (c, SynthResultQ { gates: Some(gs), lde: r.lde, distance: r.distance })
                    })
                })
        };

        // One full enumeration per parity branch at shell k_max. The
        // lattice pipeline (LLL + SE) is only well-behaved for
        // k > BRUTE_LIMIT — at tiny shells it degenerates (that's why
        // the production path brute-forces k ≤ 3) — so small horizons
        // route to the exact brute enumerator instead.
        let trace = crate::synthesis::diag::trace_enabled();
        let mut best: Option<(usize, SynthResultQ)> = baseline;
        for (label, t) in [("even", &target), ("odd", &target_odd)] {
            let t_branch = std::time::Instant::now();
            let d = det_phase_of(t);
            let found: Option<(usize, SynthResultQ)> = if k_max <= BRUTE_LIMIT {
                let mut branch_best: Option<(usize, SynthResultQ)> = None;
                let shell = brute_shell_cached(k_max);
                let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
                let thr = brute_prefilter_threshold(self.epsilon);
                for (sol, m) in shell.sols.iter().zip(&shell.mats) {
                    if brute_dist_est(m, zd, t) >= thr {
                        continue;
                    }
                    // Shells above the minimum contain √2-scaled images
                    // of lower-lde circuits (that's the covering
                    // mechanism); reduce before decomposing — the
                    // decomposer expects primitive denominators.
                    let cand: U2Q = solution_to_u2q_with_det_phase(sol, k_max, d).reduced();
                    let dist = diamond_distance_u2q_float(&cand, t);
                    if dist < self.epsilon {
                        let gates = BlochDecomposer.decompose(&cand);
                        let c = gates_cost(&gates, self.q_cost_x2);
                        match &branch_best {
                            Some((bc, _)) if *bc <= c => {}
                            _ => branch_best = Some((c, SynthResultQ {
                                gates: Some(gates), lde: k_max, distance: dist,
                            })),
                        }
                    }
                }
                branch_best
            } else {
                let v = unitary_to_uv_zeta(t);
                let mut scratch: Option<Box<IntScratch16>> = None;
                self.direct_lattice_search_at(
                    t, d, v, k_max, u64::MAX, &mut scratch, /*cost_min=*/true,
                )
                .0
            };
            if trace {
                eprintln!(
                    "[zeta] certified branch={label} k={k_max} d={d} {} t={:.0}ms",
                    found.as_ref().map(|(c, _)| format!("cost={c}"))
                        .unwrap_or_else(|| "none".into()),
                    t_branch.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if let Some((c, r)) = found {
                match &best {
                    Some((bc, _)) if *bc <= c => {}
                    _ => best = Some((c, r)),
                }
            }
        }

        let (upper, result) = best?;
        let beyond = crate::synthesis::cost_bound::cost_lb_half_units(k_max + 1);
        let cert = CostCertificate {
            upper_half_units: upper,
            lower_half_units: upper.min(beyond),
            k_searched: k_max,
            certified_optimal: upper <= beyond,
        };
        Some((result, cert))
    }

    /// Single-search lattice probe at lde `k` for one `(d, m)` arm, returning
    /// the best `(cost, SynthResultQ)` under the current `optimize_cost` mode.
    /// Mirrors the `try_lattice_k`/`check_sols` closures in
    /// `first_hit::synthesize_with_unverified_levels`. Called both sequentially
    /// (the certified m-sweep) and concurrently (per-parity `thread::scope`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn direct_lattice_search_at(
        &self,
        target: &Mat2,
        d: u32,
        v: [f64; 4],
        k: u32,
        budget: u64,
        scratch: &mut Option<Box<IntScratch16>>,
        cost_min: bool,
    ) -> (Option<(usize, SynthResultQ)>, bool) {
        let epsilon = self.epsilon;
        let s = scratch.get_or_insert_with(|| {
            let mut sb = Box::new(IntScratch16::new(epsilon));
            sb.use_f64_gs = self.use_f64_gs;
            sb.bkz_block_size = self.bkz_block_size;
            sb
        });
        let budget_hit = AtomicBool::new(false);
        let should_stop = |x: &[i64; 16]| -> bool {
            if cost_min { return false; }
            let cand = solution_to_u2q_with_det_phase(x, k, d);
            diamond_distance_u2q_float(&cand, target) < epsilon
        };
        let sols = find_aligned_lattice_points_auto_prec(
            s.as_mut(), v, None, self.deep_rot_src.as_ref(), k, epsilon, budget, &budget_hit, should_stop, None, None,
        );
        let hit = budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        let cands = sols.iter().map(|sol| (solution_to_u2q_with_det_phase(sol, k, d), k));
        (self.pick_min_cost_result(cands, target, !cost_min), hit)
    }

    /// Cost-optimal synthesis. Three stages:
    ///
    /// 1. **Brute regime** (k ≤ BRUTE_LIMIT): `enumerate_unitary_norm_shell` enumerates
    ///    the full norm shell exactly, so the min-cost candidate at the
    ///    smallest feasible k is already optimal there.
    /// 2. **Screen**: run the *production first-hit path* (a clone with
    ///    `optimize_cost` off) to locate the smallest feasible lde.
    ///    This inherits the first-hit path's concurrent-lde dispatch and
    ///    2-pass budget completeness, and is far cheaper per no-solution
    ///    lde than an enumerating sweep.
    /// 3. **Enum**: flatten `[find_lde .. find_lde+window] × m_sweep`
    ///    into independent parallel tasks, all pruning against one
    ///    shared best-cost tracker seeded with the screen candidate's
    ///    cost. The screen candidate is the floor for the final min, so
    ///    this stage can only improve it.
    pub(crate) fn synthesize_optimal(&self, target: Mat2) -> Option<SynthResultQ> {
        self.run_optimal_search_certified(target).map(|(r, _)| r)
    }

    /// Production search + certificate: same hybrid search, with the
    /// truncation ledger folded into a [`CostCertificate`]. The lower
    /// bound comes from the coverage horizon: per parity branch, the
    /// largest level whose m = 0 task completed WITHOUT budget
    /// truncation (one full level covers all lower lde via √2-scaled
    /// points); anything above the smaller branch horizon costs at
    /// least `cost_lb_half_units(horizon + 1)`. With `certify` off no
    /// m = 0 tasks run and the certificate is vacuous (lower = 0).
    pub fn synthesize_with_certificate(
        &self,
        target: Mat2,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let mut certified = self.clone();
        certified.certify = true;
        certified.run_optimal_search_certified(target)
    }

    pub(crate) fn run_optimal_search_certified(
        &self,
        target: Mat2,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let branch_horizon = |ledger: &[(u32, u32, bool)]| -> u32 {
            ledger
                .iter()
                .filter(|(_, m, truncated)| *m == 0 && !truncated)
                .map(|(k, _, _)| *k)
                .max()
                .unwrap_or(0)
        };
        let finish = |r: SynthResultQ, horizon: u32, q_cost_x2: usize| {
            let upper = gates_cost(r.gates.as_deref().unwrap_or(""), q_cost_x2);
            let beyond = crate::synthesis::cost_bound::cost_lb_half_units(horizon + 1);
            let cert = CostCertificate {
                upper_half_units: upper,
                lower_half_units: upper.min(beyond),
                k_searched: horizon,
                certified_optimal: upper <= beyond,
            };
            (r, cert)
        };

        if !self.odd_parity_branch {
            let mut ledger = Vec::new();
            let r = self.synthesize_optimal_inner(target, /*with_baseline=*/true, &mut ledger)?;
            // Single-branch search covers only one parity class: the
            // other class is unsearched, so the horizon is vacuous.
            return Some(finish(r, 0, self.q_cost_x2));
        }
        // Parity branches: the pipeline pins det to ζ₁₆^{d(target)} and
        // Q-count ≡ d (mod 2), so one target reaches only half the pool.
        // Rotating by e^{iπ/16} shifts d by 1 and opens the odd-Q half;
        // diamond distance is phase-invariant, so odd finds are valid.
        // The Clifford+T baseline skips the odd branch (T-circuit dets
        // are even ζ₁₆ powers — it would burn max_lde finding nothing).
        let g = Complex64::from_polar(1.0, PI / 16.0);
        let target_odd: Mat2 = [
            [target[0][0] * g, target[0][1] * g],
            [target[1][0] * g, target[1][1] * g],
        ];
        // One shared incumbent serves both branches: costs compare
        // directly across parities, and the staircase bound
        // (cost < c̃ ⇒ lde ≤ c̃ + 1) lets each branch use it as a dynamic
        // lde clamp, which is what allows the branches to run concurrently
        // rather than serially capped. 16 MiB stacks for the deep SE
        // recursion.
        let global_best =
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let mut even_self = self.clone();
        even_self.global_best_cost = Some(global_best.clone());
        let mut odd_self = self.clone();
        odd_self.global_best_cost = Some(global_best.clone());
        odd_self.deep_rot_src = Some((target, 1));
        // Stage-2 handshake flags (see field docs): each branch's
        // frontier dispatch waits until the peer's screen is done.
        let even_screen_done = std::sync::Arc::new(AtomicBool::new(false));
        let odd_screen_done = std::sync::Arc::new(AtomicBool::new(false));
        even_self.my_screen_done = Some(even_screen_done.clone());
        even_self.peer_screen_done = Some(odd_screen_done.clone());
        odd_self.my_screen_done = Some(odd_screen_done.clone());
        odd_self.peer_screen_done = Some(even_screen_done.clone());
        let mut ledger_even = Vec::new();
        let mut ledger_odd = Vec::new();
        let trace = crate::synthesis::diag::trace_enabled();
        let t_branches = std::time::Instant::now();
        // At deep ε each branch saturates the pool alone, so concurrent
        // parities dilute both (~2× wall for slightly lower cost); run
        // them sequentially. CYCLOSYNTH_SEQ_PARITY=0 forces concurrency.
        // The shared incumbent flows identically either way.
        let force_sequential = self.seq_parity.unwrap_or_else(|| {
            self.epsilon < 2.5e-8
                && std::env::var("CYCLOSYNTH_SEQ_PARITY").as_deref() != Ok("0")
        });
        if force_sequential {
            // No peer exists in sequential mode — pre-set BOTH handshake
            // flags or the frontier dead-sleeps its full 4×deadline
            // bound waiting on a screen that never starts.
            even_screen_done.store(true, Ordering::Release);
            odd_screen_done.store(true, Ordering::Release);
            let r_e = even_self.synthesize_optimal_inner(
                target, /*with_baseline=*/ true, &mut ledger_even,
            );
            let r_o = odd_self.synthesize_optimal_inner(
                target_odd, /*with_baseline=*/ false, &mut ledger_odd,
            );
            let horizon =
                branch_horizon(&ledger_even).min(branch_horizon(&ledger_odd));
            return match (r_e, r_o) {
                (Some(a), Some(b)) => {
                    let ca =
                        gates_cost(a.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                    let cb =
                        gates_cost(b.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                    Some(finish(if cb < ca { b } else { a }, horizon, self.q_cost_x2))
                }
                (a, b) => a.or(b).map(|r| finish(r, horizon, self.q_cost_x2)),
            };
        }
        let (r_even, r_odd) = std::thread::scope(|s| {
            let even_ledger = &mut ledger_even;
            let odd_ledger = &mut ledger_odd;
            let even_ref = &even_self;
            let odd_ref = &odd_self;
            let even_done = &even_screen_done;
            let odd_done = &odd_screen_done;
            let h_even = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn_scoped(s, move || {
                    let t0 = std::time::Instant::now();
                    let r = even_ref.synthesize_optimal_inner(
                        target, /*with_baseline=*/ true, even_ledger,
                    );
                    // Branch done ⇒ screen trivially "done" (covers
                    // returns before stage 2, e.g. stage-1 brute finds)
                    // so the peer's handshake wait can't outlive us.
                    even_done.store(true, Ordering::Release);
                    (r, t0.elapsed())
                })
                .expect("spawn even parity branch");
            let h_odd = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn_scoped(s, move || {
                    let t0 = std::time::Instant::now();
                    let r = odd_ref.synthesize_optimal_inner(
                        target_odd, /*with_baseline=*/ false, odd_ledger,
                    );
                    odd_done.store(true, Ordering::Release);
                    (r, t0.elapsed())
                })
                .expect("spawn odd parity branch");
            let (r_even, dt_even) = h_even.join().unwrap();
            let (r_odd, dt_odd) = h_odd.join().unwrap();
            if trace {
                eprintln!(
                    "[zeta] optimal branches even={:.0}ms odd={:.0}ms scope={:.0}ms",
                    dt_even.as_secs_f64() * 1000.0,
                    dt_odd.as_secs_f64() * 1000.0,
                    t_branches.elapsed().as_secs_f64() * 1000.0,
                );
            }
            (r_even, r_odd)
        });
        // Coverage holds only up to the SMALLER branch horizon: a level
        // is closed only when both parity worlds enumerated it fully.
        let horizon = branch_horizon(&ledger_even).min(branch_horizon(&ledger_odd));
        match (r_even, r_odd) {
            (Some(a), Some(b)) => {
                let ca = gates_cost(a.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                let cb = gates_cost(b.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                Some(finish(if cb < ca { b } else { a }, horizon, self.q_cost_x2))
            }
            (a, b) => a.or(b).map(|r| finish(r, horizon, self.q_cost_x2)),
        }
    }

    /// Scan ε-close candidates, decompose each, and keep the min-cost
    /// one — or return the FIRST ε-close one when `first_hit` (the
    /// legacy non-optimal semantics, which must stay order-sensitive).
    pub(crate) fn pick_min_cost_result<I>(
        &self,
        cands: I,
        target: &Mat2,
        first_hit: bool,
    ) -> Option<(usize, SynthResultQ)>
    where
        I: IntoIterator<Item = (U2Q, u32)>,
    {
        let mut best: Option<(usize, SynthResultQ)> = None;
        for (cand, lde) in cands {
            let dist = diamond_distance_u2q_float(&cand, target);
            if dist < self.epsilon {
                let gates = BlochDecomposer.decompose(&cand);
                let cost = gates_cost(&gates, self.q_cost_x2);
                let result = SynthResultQ { gates: Some(gates), lde, distance: dist };
                if first_hit {
                    return Some((cost, result));
                }
                match &best {
                    Some((bc, _)) if *bc <= cost => {}
                    _ => best = Some((cost, result)),
                }
            }
        }
        best
    }

    /// Stage 1 of the optimal pipeline: exact min-cost scan of the brute
    /// shells (k ≤ BRUTE_LIMIT). A find here is already optimal at the
    /// smallest feasible k.
    pub(crate) fn brute_min_cost(&self, target: &Mat2, d: u32) -> Option<(usize, SynthResultQ)> {
        let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
        let thr = brute_prefilter_threshold(self.epsilon);
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let shell = brute_shell_cached(k);
            let close = shell
                .sols
                .iter()
                .zip(&shell.mats)
                .filter(|(_, m)| brute_dist_est(m, zd, target) < thr)
                .map(|(sol, _)| (solution_to_u2q_with_det_phase(sol, k, d), k));
            let best = self.pick_min_cost_result(close, target, false);
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// Stage 2 of the optimal pipeline: the first-hit screen and the
    /// Clifford+T baseline, in parallel. T-only solutions live at lde ≈
    /// T-count — far above the enum window — so covering them requires
    /// synthesizing them directly, which also makes the result
    /// never-worse-than-Clifford+T by construction and seeds the stage-3
    /// prune. Returns `(screen result, unclear levels, baseline as a
    /// √T-shaped (cost, result) candidate)`.
    pub(crate) fn screen_and_baseline(
        &self,
        target: Mat2,
        with_baseline: bool,
    ) -> (Option<SynthResultQ>, Vec<u32>, Option<(usize, SynthResultQ)>) {
        // Clifford+T dets are even ζ₁₆ powers — odd-class targets make
        // the baseline burn its whole lde sweep finding nothing.
        let with_baseline = with_baseline && det_phase_of(&target).is_multiple_of(2);
        let (first, unclear, t_baseline) = std::thread::scope(|s| {
            let baseline_handle = if with_baseline {
                Some(
                    std::thread::Builder::new()
                        // 16 MiB: deep SE recursion.
                        .stack_size(16 * 1024 * 1024)
                        .spawn_scoped(s, || {
                            let t0 = std::time::Instant::now();
                            let r = crate::synthesis::clifford_t::SynthesizerT::new(self.epsilon)
                                .synthesize(target);
                            crate::synthesis::diag::T_STAGE_BASELINE_NS
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            r
                        })
                        .expect("spawn clifford_t baseline thread"),
                )
            } else {
                None
            };
            let mut first_hit = self.clone();
            first_hit.optimize_cost = false;
            first_hit.odd_parity_branch = false;
            // Truncated-and-never-cleared levels below find-lde must
            // reach the enum grid or the window silently misses them.
            let mut unclear = Vec::new();
            let first = first_hit.synthesize_with_unverified_levels(target, Some(&mut unclear));
            (first, unclear, baseline_handle.and_then(|h| h.join().unwrap()))
        });
        // The baseline's gate string contains no Q, so its cost is
        // exactly 2·T_count half-units.
        let baseline: Option<(usize, SynthResultQ)> = t_baseline.and_then(|r| {
            let dist = r.distance;
            // NaN-safe reject (see the screen_and_baseline note).
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(dist < self.epsilon) {
                return None;
            }
            r.gates.map(|g| {
                let c = gates_cost(&g, self.q_cost_x2);
                (c, SynthResultQ { gates: Some(g), lde: r.lde, distance: dist })
            })
        });
        (first, unclear, baseline)
    }

    pub(crate) fn synthesize_optimal_inner(
        &self,
        target: Mat2,
        with_baseline: bool,
        ledger_out: &mut Vec<(u32, u32, bool)>,
    ) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        let trace = diag::trace_enabled();

        // The enum stage runs SE walks of its own, so the guard must
        // span both stages.
        let _verify_guard = VerifyGuard::enable_for(self.epsilon);

        let d = det_phase_of(&target);
        let v = unitary_to_uv_zeta(&target);

        // Stage 1: brute regime, exact min-cost at the smallest k.
        if let Some((c, r)) = self.brute_min_cost(&target, d) {
            // Publish the brute win before returning — otherwise gate-like
            // targets leave the peer branch's dynamic lde clamp unseeded
            // and its screen sweeps to max_lde for nothing.
            if let Some(g) = &self.global_best_cost {
                g.fetch_min(c, std::sync::atomic::Ordering::Relaxed);
            }
            return Some(r);
        }

        let t_s = std::time::Instant::now();
        let (first, mut screen_unclear, baseline) =
            self.screen_and_baseline(target, with_baseline);
        diag::T_STAGE_SCREEN_NS
            .fetch_add(t_s.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // Signal screen completion to the peer parity branch; the
        // matching wait sits just before the frontier dispatch below.
        if let Some(flag) = &self.my_screen_done {
            flag.store(true, Ordering::Release);
        }
        let baseline_cost = baseline.as_ref().map(|(c, _)| *c).unwrap_or(usize::MAX);

        // If the √T screen found nothing within the configured bounds
        // (max_lde, budgets), return None: the baseline is a cost floor
        // for comparison, not a fallback — returning it would silently
        // bypass the caller's search bounds.
        let first = first?;
        let fl = first.lde;
        let first_cost = first
            .gates
            .as_deref()
            .map(|g| gates_cost(g, self.q_cost_x2))
            .unwrap_or(usize::MAX);
        if trace {
            eprintln!(
                "[zeta] optimal screen lde={fl} cost={first_cost} baseline(T)={baseline_cost}  t={:.0}ms",
                t_s.elapsed().as_secs_f64() * 1000.0);
        }

        // Stage 3: enum over the (lde, m) grid against one pre-seeded
        // incumbent (fetch_min — a peer's earlier cheaper find must
        // survive). Certify adds m = 0 tasks per level: the only variant
        // whose untruncated completion proves a level exhausted, which
        // is what moves the certificate horizon.
        let local_best = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let shared_best: &std::sync::atomic::AtomicUsize =
            self.global_best_cost.as_deref().unwrap_or(&local_best);
        shared_best.fetch_min(
            first_cost.min(baseline_cost),
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut tasks: Vec<(u32, u32)> = (0..=self.optimal_lde_window)
            .map(|i| fl + i)
            .filter(|&k| k <= self.max_lde)
            .flat_map(|k| self.optimal_m_sweep.iter().map(move |&m| (k, m)))
            .collect();
        if self.certify {
            for i in 0..=self.optimal_lde_window {
                let k = fl + i;
                if k <= self.max_lde && !tasks.contains(&(k, 0)) {
                    tasks.push((k, 0));
                }
            }
        }
        // Unverified below-fl levels get the same arm set as window
        // levels — the find at fl short-circuited their pass-2 retry, so
        // they may still hold a cheaper candidate.
        screen_unclear.sort_unstable();
        screen_unclear.dedup();
        screen_unclear.retain(|&k| k < fl && k <= self.max_lde);
        if !screen_unclear.is_empty() {
            if trace {
                eprintln!("[zeta] optimal screen left levels {screen_unclear:?} unverified below fl={fl} — adding to enum grid");
            }
            for &k in &screen_unclear {
                for &m in &self.optimal_m_sweep {
                    if !tasks.contains(&(k, m)) {
                        tasks.push((k, m));
                    }
                }
                if self.certify && !tasks.contains(&(k, 0)) {
                    tasks.push((k, 0));
                }
            }
        }
        // ── Anytime merged frontier (fast path) ─────────────────────
        // With a deadline configured and certify off, all (k, m ≥ 1)
        // arms run as ONE floor-ordered prefix frontier under a wall
        // deadline instead of per-arm node budgets (see
        // `min_cost_frontier_search`). The legacy task grid below remains the
        // certify path (honest budget-truncation semantics) and the
        // deep-ε path (deadline default None), and still handles
        // m = 0 arms (single-shot probes are not prefix work-units).
        if !self.certify
            && !tasks.is_empty()
            && tasks.iter().all(|&(_, m)| m >= 1)
        {
            if let Some(deadline_ms) = self.optimal_deadline_ms {
                // Wait for the peer's screen before flooding the pool (a
                // frontier starves a running screen badly); bounded, and
                // the peer's branch-return store guarantees progress on
                // early exits.
                if let Some(peer) = &self.peer_screen_done {
                    let t_wait = std::time::Instant::now();
                    let cap = std::time::Duration::from_millis(4 * deadline_ms.max(100));
                    while !peer.load(Ordering::Acquire) && t_wait.elapsed() < cap {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    if trace {
                        eprintln!(
                            "[zeta] optimal frontier handshake wait={:.0}ms",
                            t_wait.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                let t_w = std::time::Instant::now();
                let (fr, level_truncated) =
                    self.run_frontier_grouped_by_m(&target, &tasks, deadline_ms, shared_best);
                diag::T_STAGE_FRONTIER_NS
                    .fetch_add(t_w.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if trace {
                    eprintln!(
                        "[zeta] optimal frontier {:?} deadline={}ms t={:.0}ms truncated={:?}",
                        tasks, deadline_ms,
                        t_w.elapsed().as_secs_f64() * 1000.0,
                        tasks.iter().zip(level_truncated.iter())
                            .filter(|(_, &tr)| tr).map(|(t, _)| *t)
                            .collect::<Vec<_>>(),
                    );
                }
                let mut best: (usize, SynthResultQ) = (first_cost, first);
                if let Some((bc, br)) = baseline {
                    if bc < best.0 {
                        best = (bc, br);
                    }
                }
                if let Some((c, res)) = fr {
                    if trace {
                        eprintln!("[zeta]   frontier best lde={:>2} cost={c} dist={:.3e}",
                            res.lde, res.distance);
                    }
                    if c < best.0 {
                        best = (c, res);
                    }
                }
                *ledger_out = tasks
                    .iter()
                    .zip(level_truncated)
                    .map(|(&(k, m), tr)| (k, m, tr))
                    .collect();
                return Some(best.1);
            }
        }

        let t_w = std::time::Instant::now();
        let task_results: Vec<EnumArmOutcome> =
            std::thread::scope(|s| {
                let handles: Vec<_> = tasks
                    .iter()
                    .map(|&(k, m)| {
                        // 16 MiB stack: these threads run rayon's in-place
                        // execution of prefix_split_search_q, whose
                        // per-prefix scratch + SE recursion overflow the
                        // 2 MiB scoped-thread default.
                        std::thread::Builder::new()
                            .stack_size(16 * 1024 * 1024)
                            .spawn_scoped(s, move || {
                                let (r, truncated) = self.run_enum_arm(
                                    target, d, v, k, m, /*cost_min=*/true,
                                    Some(shared_best),
                                );
                                (k, m, truncated, r)
                            })
                            .expect("spawn lde-window thread")
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
        // The grid is the frontier stage's deep-ε/certify form — same
        // scoreboard column.
        diag::T_STAGE_FRONTIER_NS
            .fetch_add(t_w.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if trace {
            eprintln!("[zeta] optimal enum {:?} parallel t={:.0}ms",
                tasks, t_w.elapsed().as_secs_f64() * 1000.0);
        }
        let mut best: (usize, SynthResultQ) = (first_cost, first);
        if let Some((bc, br)) = baseline {
            if bc < best.0 {
                best = (bc, br);
            }
        }
        // Truncation ledger: (level, m, truncated) for every enum task.
        let mut ledger: Vec<(u32, u32, bool)> = Vec::new();
        for (k, m, truncated, r) in task_results {
            ledger.push((k, m, truncated));
            if let Some((c, res)) = r {
                if trace {
                    eprintln!("[zeta]   enum  lde={:>2}  cost={c} m={m} dist={:.3e}",
                        res.lde, res.distance);
                }
                if c < best.0 {
                    best = (c, res);
                }
            }
        }

        // Floor-driven extension (certify mode): keep running full m=0
        // levels above the window while the proven beyond-horizon floor
        // is still below the incumbent and the extension time budget
        // lasts. Every completed (untruncated) level raises the
        // certificate's lower bound by 4 half-units.
        if self.certify && self.certify_extra_ms > 0 {
            let t_ext = std::time::Instant::now();
            let mut k = fl + self.optimal_lde_window + 1;
            while k <= self.max_lde
                && crate::synthesis::cost_bound::cost_lb_half_units(k) < best.0
                && (t_ext.elapsed().as_millis() as u64) < self.certify_extra_ms
            {
                let (r, truncated) =
                    self.run_enum_arm(target, d, v, k, 0, true, Some(shared_best));
                ledger.push((k, 0, truncated));
                if trace {
                    eprintln!("[zeta] certify-extend k={k} truncated={truncated} t={:.0}ms",
                        t_ext.elapsed().as_secs_f64() * 1000.0);
                }
                if let Some((c, res)) = r {
                    if c < best.0 {
                        best = (c, res);
                    }
                }
                if truncated {
                    break; // deeper levels will only be bigger
                }
                k += 1;
            }
        }

        *ledger_out = ledger;
        Some(best.1)
    }

    /// One (lde, m) variant of the optimal search: m=0 → single-shot
    /// lattice probe, m≥1 → FGKM-prefix split with the default d_R
    /// filter. Extracted from the m-sweep loop so the enum phase can run
    /// all (k, m) pairs as independent parallel tasks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_enum_arm(
        &self,
        target: Mat2,
        d: u32,
        v: [f64; 4],
        k: u32,
        m: u32,
        cost_min: bool,
        shared_best_cost: Option<&std::sync::atomic::AtomicUsize>,
    ) -> (Option<(usize, SynthResultQ)>, bool) {
        let budget_mult = self.optimal_budget_multiplier.max(1);
        if m == 0 {
            // In certify mode the m = 0 tasks are the coverage proof —
            // a truncated one contributes nothing to the horizon, so
            // give them room (32×) to actually finish the level.
            let cert_boost: u64 = if self.certify { 32 } else { 1 };
            let cap = PASS1_CAP
                .saturating_mul(budget_mult)
                .saturating_mul(cert_boost);
            let mut local_scratch: Option<Box<IntScratch16>> = None;
            let (r, hit) =
                self.direct_lattice_search_at(&target, d, v, k, cap, &mut local_scratch, cost_min);
            if hit && crate::synthesis::diag::trace_enabled() {
                eprintln!("[zeta]   enum (k={k}, m=0) BUDGET-HIT — coverage lost");
            }
            (r, hit)
        } else if m < k {
            // The d_R filters were tuned for first-hit *speed*; in enum
            // mode they may exclude det-phase classes containing the
            // cost optimum. `optimal_open_dr_filter` lifts them.
            let filter = if self.optimal_open_dr_filter {
                Vec::new()
            } else {
                default_inner_det_phase_filter(m)
            };
            let cap = pass1_prefix_leaf_cap_for(self.epsilon).saturating_mul(budget_mult);
            let (r, budget_hit) = self.prefix_split_search_q(
                &target, k, m,
                PrefixSplitOpts {
                    dr_filter_override: Some(&filter),
                    per_prefix_cap: cap,
                    external_abort: None,
                    consumed: None,
                    cost_min_override: Some(cost_min),
                    shared_best_cost,
                },
            );
            if budget_hit && crate::synthesis::diag::trace_enabled() {
                eprintln!("[zeta]   enum (k={k}, m={m}) BUDGET-HIT — level truncated");
            }
            (r.map(|res| {
                let c = gates_cost(res.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                (c, res)
            }), budget_hit)
        } else {
            (None, false)
        }
    }
}
