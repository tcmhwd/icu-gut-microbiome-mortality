#!/usr/bin/env Rscript
# 21_lean_figS4_external_transportability_v2.R -- v2-package nomenclature
# pass (2026-10-02, FINAL_LEAN_SUBMISSION_2026-10-02_v2): copy of
# 21_lean_figS4_external_transportability.R. Only change: "M1" shorthand
# replaced with reader-facing "literature-derived microbiome" wording in
# panel A's y-axis title, panel B's title, and the composition caption (all
# in-figure visible text). "External benchmarking"/"partial biological
# reproducibility" framing (not "replication") in the caption is confirmed
# unchanged. No statistic, CI, p-value, data point, or axis range changed.
#
# 21_lean_figS4_external_transportability.R -- Lean-submission consolidation
# (FIGURE_CONSOLIDATION_REPORT.md Task C, 2026-10-02): merges three existing,
# already-computed panels into ONE new Supplementary Fig. S4. NO new
# statistics -- all three panels are re-composed from already-computed values
# read live from their original source CSVs, same geoms/mappings as their
# standalone originals (reports/paper1/critical_care_external/make_cc_figures.R):
#   Panel A: external decision-curve analysis, PRJNA842425 (originally
#            Fig_external_dca / p4).
#   Panel B: cross-cohort effect-size pattern of the frozen 16-marker M1 panel
#            (originally Fig_cross_cohort_M1_replication / p2).
#   Panel C: external application of the Enterococcus >=30% domination flag
#            (originally Fig_enterococcus_replication / p3, built via
#            hs_forest()).
#
# Panel C is built with hs_forest(), which bakes in its OWN complete
# theme_house(base_size=11) internally -- the project's global
# theme_set(theme_manuscript()) (00_setup_theme.R) never reaches it, and
# appending a second COMPLETE theme after it would discard its own
# theme()-tweaks (panel.grid.major.y blank, legend.position etc). Per the
# landmine already found and fixed this session (R_fig/scripts/
# 15b_paper1_figures_final.R's hs_text_bump()), every panel gets a PARTIAL,
# absolute-pt theme() override (figS4_text_bump(), below) rather than a
# second complete theme -- this gives every panel (hs_forest-built or plain
# ggplot) IDENTICAL pt sizes per element type and leaves each panel's own
# non-text theme()-tweaks (legend.position, panel grid, axis angle)
# untouched.
#
# That override is applied to EACH panel directly with `+`, not broadcast
# across the composed patchwork with `&` -- a second, independent bug (not
# the hs_forest one above) was found and confirmed this session while
# building Lean_FigS3 (script 20): `patchwork_object + plot_annotation(...)
# & theme(...)` corrupts a discrete-axis panel's tick-LABEL positions
# (collapsed/interleaved text), even though the identical panels compose
# cleanly with EITHER plot_annotation() alone OR `&`-broadcast alone, just
# not combined. This figure's panel B (discrete cohort x-axis, 16-level
# marker y-axis) is exactly the shape that triggers it, and this figure uses
# plot_annotation() for its title/caption, so `&` is avoided entirely here.
source("/home/tcmhwd/microbiome/ICU_public/semantic_microbiome/R_fig/scripts/00_setup_theme.R")
suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(stringr); library(grid); library(ragg) })

CC <- file.path(SEM, "reports", "paper1", "critical_care_external")

wrap <- function(x, w) paste(strwrap(x, width = w), collapse = "\n")

# device-accurate text-width wrapper (reused verbatim from make_cc_figures.R /
# make_addendum2_figures.R -- character-count wrapping does not account for
# point size or canvas width at all, confirmed insufficient by this project's
# own earlier font-size-fix passes).
wrap_to_width_in <- function(text, pt, budget_in, family = "Arial", fontface = "plain") {
  text_width_in <- function(s) {
    tmp <- tempfile(fileext = ".png")
    ragg::agg_png(tmp, width = 20, height = 4, units = "in", res = 100)
    grid::pushViewport(grid::viewport(gp = grid::gpar(fontfamily = family, fontsize = pt, fontface = fontface)))
    w <- grid::convertWidth(grid::stringWidth(s), "inches", valueOnly = TRUE)
    grid::popViewport(); grDevices::dev.off(); unlink(tmp)
    w
  }
  words <- strsplit(text, "\\s+")[[1]]
  lines <- character(0); cur <- ""
  for (w in words) {
    cand <- if (cur == "") w else paste(cur, w)
    if (cur == "" || text_width_in(cand) <= budget_in) cur <- cand
    else { lines <- c(lines, cur); cur <- w }
  }
  if (cur != "") lines <- c(lines, cur)
  paste(lines, collapse = "\n")
}

# ---------------------------------------------------------------------------
# combined-figure sizing, computed from scratch (not reused unchanged from
# each standalone panel's own prior fix). Governing element across all three
# panels: legend.text (rel 0.80 equivalent, panel B's "Spearman rho" fill
# legend) -- the tightest of the elements these three panels actually use.
# width = 9.0in (set by panel C/B's own content -- long row labels + p-value
# column in C, 16 long marker names in B; both already needed close to this
# width standalone). scale = 180/(9.0*25.4) = 0.787402
#   base_size = 20: legend.text final_pt = 20*0.80*0.787402 = 12.60 (clears
#   11.5pt floor and the 12pt target); plot.title = 15.75, axis.title =
#   14.96, axis.text = 13.39, legend.title = 13.39 -- all comfortably clear.
# plot.caption deliberately uses the 0.80 ratio (not theme_manuscript()'s
# native 0.65), same deviation and same rationale as 15b_paper1_figures_
# final.R's hs_text_bump(): this figure's caption carries substantive
# methodological wording (not a de-emphasised footnote) that must itself
# clear the readability floor.
FIGW <- 9.0
BASE <- 20
figS4_text_bump <- function(base_size) {
  theme(
    plot.title    = element_text(size = base_size * 1.00, face = "bold"),
    plot.subtitle = element_text(size = base_size * 0.85),
    plot.caption  = element_text(size = base_size * 0.80),
    axis.title    = element_text(size = base_size * 0.95),
    axis.text     = element_text(size = base_size * 0.85),
    legend.text   = element_text(size = base_size * 0.80),
    legend.title  = element_text(size = base_size * 0.85),
    strip.text    = element_text(size = base_size * 0.85, face = "bold")
  )
}

# ===========================================================================
# Panel A -- external decision-curve analysis (same data/geoms as
# Fig_external_dca / p4 in make_cc_figures.R)
# ===========================================================================
dca <- read_csv(file.path(CC, "prjna842425_dca_results.csv"), show_col_types = FALSE)
n_excl_dca <- sum(dca$CI_excludes_zero)
panelA <- ggplot(dca, aes(x = threshold, y = delta_NB)) +
  geom_ribbon(aes(ymin = delta_NB_lo, ymax = delta_NB_hi), fill = PAL$combined, alpha = 0.15) +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.4) +
  geom_line(colour = PAL$combined, linewidth = 0.6) +
  geom_point(data = filter(dca, CI_excludes_zero), colour = PAL$combined, shape = 17, size = 2.6) +
  geom_point(data = filter(dca, !CI_excludes_zero), colour = PAL$combined, shape = 21, fill = "white", size = 1.8) +
  scale_x_continuous(labels = scales::percent) +
  labs(title = wrap_to_width_in("A. External decision-curve analysis (PRJNA842425, SOFA-referenced)",
                                 pt = BASE * 1.0, budget_in = FIGW - 0.3, fontface = "bold"),
       # v2 render fix (2026-10-02): the longer reader-facing y-axis title
       # (vs the original short "SOFA+M1") overflowed panel A's own row
       # height when rendered as a rotated label -- a rotated axis title's
       # footprint is set by its LONGEST individual line, and the first line
       # here was now 51 chars (vs the original 27-char "Delta NB (SOFA+M1 -
       # SOFA),"), confirmed on the first v2 render to bleed upward past
       # panel A's row and overlap both the composition title above it and
       # panel B's title below it. Adding a second line break keeps every
       # line <=31 chars, matching the original validated bottleneck line
       # ("net true positives per patient", 31 chars) instead of exceeding it.
       x = "Threshold probability", y = "Delta NB (SOFA + lit.-derived\nmicrobiome - SOFA),\nnet true positives per patient") +
  # plot.title.position = "plot" deliberately OMITTED here (default "panel"
  # kept): panel A has a tall rotated y-axis title, and anchoring the plot
  # title to the full plot width (spanning that left-side axis-label gutter)
  # made the wrapped 2-line title overlap the rotated axis text at the
  # top-left corner on the previous render -- confirmed by a cropped close-up
  # inspection. Default "panel" anchors the title to the panel's own left
  # edge, clear of the axis-label gutter.
  theme(legend.position = "none") +
  figS4_text_bump(BASE)

# ===========================================================================
# Panel B -- cross-cohort effect-size pattern, 16 frozen M1 markers x 3
# cohorts (same data/geoms as Fig_cross_cohort_M1_replication / p2)
# ===========================================================================
mk <- read_csv(file.path(CC, "M1_CROSS_COHORT_MARKER_RESULTS.csv"), show_col_types = FALSE)
mk$marker_short <- hs_readable_marker(str_remove(mk$marker, "^Literature_|^Genome_resolved_"))
mk$cohort <- factor(mk$cohort, levels = c("Gangnam", "AGI_BGISEQ", "PRJNA842425"),
                     labels = c("Internal (N=302)", "AGI_BGISEQ (N=210)", "PRJNA842425 (N=71)"))
ord <- mk %>% filter(cohort == "Internal (N=302)") %>% arrange(desc(spearman_rho)) %>% pull(marker_short)
mk$marker_short <- factor(mk$marker_short, levels = rev(ord))
lim <- max(abs(mk$spearman_rho), na.rm = TRUE)

corr <- read_csv(file.path(CC, "cross_cohort_effect_size_correlation.csv"), show_col_types = FALSE)
piv <- mk %>% select(marker, cohort, spearman_rho) %>%
  mutate(cohort = recode(as.character(cohort), "Internal (N=302)" = "Gangnam",
                          "AGI_BGISEQ (N=210)" = "AGI_BGISEQ", "PRJNA842425 (N=71)" = "PRJNA842425")) %>%
  pivot_wider(names_from = cohort, values_from = spearman_rho)
n_agi <- sum(sign(piv$Gangnam) == sign(piv$AGI_BGISEQ), na.rm = TRUE)
n_prj <- sum(sign(piv$Gangnam) == sign(piv$PRJNA842425), na.rm = TRUE)
bp_agi <- binom.test(n_agi, 16, 0.5, alternative = "greater")$p.value
bp_prj <- binom.test(n_prj, 16, 0.5, alternative = "greater")$p.value
c_ga <- corr %>% filter(cohort_a == "Gangnam", cohort_b == "AGI_BGISEQ")
c_gp <- corr %>% filter(cohort_a == "Gangnam", cohort_b == "PRJNA842425")
c_ap <- corr %>% filter(cohort_a == "AGI_BGISEQ", cohort_b == "PRJNA842425")

# in-tile rho-label mm size: baseline 2.6mm at theme_manuscript(base_size=11)
# (house-style convention used throughout this script set), scaled by the
# same BASE/11 ratio as the rest of this panel's text.
rho_label_mm <- 2.6 * (BASE / 11)
panelB <- ggplot(mk, aes(x = cohort, y = marker_short, fill = spearman_rho)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f%s", spearman_rho, ifelse(spearman_p < 0.05, "*", ""))),
            size = rho_label_mm, colour = "black") +
  scale_fill_gradient2(low = HS_PAL$blue, mid = "white", high = HS_PAL$red, midpoint = 0,
                        limits = c(-lim, lim), name = "Spearman\nrho\n(vs mortality)") +
  labs(title = wrap_to_width_in("B. Cross-cohort effect-size pattern: 16 frozen literature-derived microbiome markers",
                                 pt = BASE * 1.0, budget_in = FIGW - 0.3, fontface = "bold"), x = NULL, y = NULL) +
  theme(plot.title.position = "plot",
        axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1)) +
  figS4_text_bump(BASE)

# ===========================================================================
# Panel C -- external application of the Enterococcus >=30% domination flag
# (same data/geoms as Fig_enterococcus_replication / p3, hs_forest()-built)
# ===========================================================================
ee <- read_csv(file.path(CC, "ENTEROCOCCUS_REPLICATION_RESULTS.csv"), show_col_types = FALSE)
fisher_ci <- function(a, b, c, d) {
  m <- matrix(c(a, c, b, d), nrow = 2)
  ft <- fisher.test(m)
  c(ft$estimate[[1]], ft$conf.int[1], ft$conf.int[2])
}
ee_rows <- list()
for (i in seq_len(nrow(ee))) {
  r <- ee[i, ]
  ci <- fisher_ci(r$deaths_exposed, r$N_exposed - r$deaths_exposed,
                   r$deaths_unexposed, r$N_unexposed - r$deaths_unexposed)
  ee_rows[[length(ee_rows) + 1]] <- data.frame(
    label = sprintf("%s (unadjusted, N=%d exposed)", r$cohort, r$N_exposed),
    estimate = ci[1], ci_low = ci[2], ci_high = min(ci[3], 50), p = r$fisher_p, cohort = r$cohort)
  if (!is.na(r$adjusted_OR)) {
    ee_rows[[length(ee_rows) + 1]] <- data.frame(
      label = sprintf("%s (SOFA-adjusted)", r$cohort),
      estimate = r$adjusted_OR, ci_low = r$adjusted_OR_lo, ci_high = r$adjusted_OR_hi,
      p = r$LRT_p, cohort = r$cohort)
  }
}
ee_dat <- bind_rows(ee_rows)
ee_dat$label <- factor(ee_dat$label, levels = rev(ee_dat$label))
p_agi <- ee$fisher_p[ee$cohort == "AGI_BGISEQ"]
p_prj <- ee$fisher_p[ee$cohort == "PRJNA842425"]

# x-axis right-margin fix for the p-value column: hs_forest()'s own internal
# margin is a fixed 0.32-of-log10-span fraction sized for its native 2.7mm
# label -- NOT proportional to base_size, so a naive ratio-based rescale of
# that fraction under-margined on a real render for this same panel
# standalone (make_cc_figures.R, FONT_SIZE_FIX_S5_S6_S8_S9_S10.md: the first,
# ratio-scaled attempt at this panel's own larger base_size, 18.0, was
# re-tried at a corrected base_size of 22.5 and the ratio-scaled margin still
# clipped; 0.55 is the value that render-confirmed clears it). Reused
# directly here (same panel content, same hs_forest() call, only a different
# combined-figure base_size) rather than re-derived from scratch, since it is
# an empirically-validated floor, not a formula.
ee_rng_hi <- max(c(ee_dat$ci_high, ee_dat$estimate, 1), na.rm = TRUE)
ee_rng_lo <- min(c(ee_dat$ci_low, ee_dat$estimate, 1), na.rm = TRUE)
ee_l_hi <- log10(max(ee_rng_hi, 1e-6)); ee_l_lo <- log10(max(ee_rng_lo, 1e-6))
ee_span <- max(ee_l_hi - ee_l_lo, 1e-6)
ee_xlim_hi <- 10 ^ (ee_l_hi + ee_span * 0.55)

panelC <- hs_forest(ee_dat,
                     xlab = wrap_to_width_in("Odds ratio for mortality, Enterococcus domination >=30% (95% CI, log scale)",
                                              pt = BASE * 0.85, budget_in = 3.4),
                     ref = 1, log_x = TRUE, order = "none") +
  labs(title = "C. External application: Enterococcus >=30% domination flag") +
  theme(panel.grid.major.y = element_blank(), plot.title.position = "plot") +
  scale_x_log10(limits = c(NA, ee_xlim_hi)) +
  figS4_text_bump(BASE)

# hs_forest()'s internal p-value label is a fixed 2.7mm GeomText (sized for
# its own theme_house(base_size=11)) -- scaled here by the same BASE/11 ratio
# as every other element, same pattern as 15b_paper1_figures_final.R's
# bump_text()/make_cc_figures.R's enterococcus_ratio.
entero_ratio <- BASE / 11
for (i in seq_along(panelC$layers)) {
  if (inherits(panelC$layers[[i]]$geom, "GeomText") && !is.null(panelC$layers[[i]]$aes_params$size)) {
    panelC$layers[[i]]$aes_params$size <- panelC$layers[[i]]$aes_params$size * entero_ratio
  }
}

# ===========================================================================
# Compose: single-column vertical stack (A on top, then B, then C -- B needs
# much more vertical room than A/C for its 16 marker rows, which a side-by-
# side layout would force A/C's columns to match or leave wasted white
# space). Each panel already carries its own `figS4_text_bump(BASE)` directly
# (above), NOT a `&`-broadcast theme here -- a real bug was found and
# confirmed by an isolated render test while building Lean_FigS3 (script 20):
# `patchwork_object + plot_annotation(...) & theme(...)` corrupts a discrete
# x/y-axis panel's tick-LABEL positions (collapsed/interleaved text) even
# though the identical panels compose cleanly with EITHER plot_annotation()
# alone OR `&`-broadcast alone, just not combined. Panel B's discrete cohort
# x-axis and 16-level marker y-axis make this figure exactly the shape that
# triggers it, so the same per-panel-theme + explicit plot_annotation(theme=)
# fix is used here too.
#
# A third defect found and fixed on the first real render: by default
# patchwork ALIGNS the plot PANEL (not just the cell) across stacked plots --
# since panel B/C have much wider left-side row-axis-label text than panel
# A's short y-axis title, patchwork shrank panel A's actual plot area down to
# a tiny column on the right to keep all three panels' left edges aligned
# (visible on that render as panel A's DCA plot floating as a small square
# in mostly-empty space, with its own title/axis-label overlapping the
# figure's overall title above it). These three panels have unrelated x-axes
# (threshold / cohort-name / log-OR) and should NOT be alignment-locked
# together, so each is wrapped in `patchwork::free()` (patchwork >= 1.2) to
# exempt it from that cross-panel alignment and let it use its own full cell.
# ===========================================================================
combined <- (free(panelA) / free(panelB) / free(panelC)) +
  plot_layout(heights = c(0.85, 1.85, 0.95)) +
  plot_annotation(
    title = wrap_to_width_in("External clinical and biological transportability of microbiome signals",
                              pt = BASE * 1.0, budget_in = FIGW - 0.3, fontface = "bold"),
    caption = wrap_to_width_in(sprintf(paste(
      "All three panels report EXTERNAL APPLICATION of frozen, never-re-optimized internal-cohort",
      "results to independent cohorts (AGI_BGISEQ and/or PRJNA842425) -- none refit any model or",
      "re-select any threshold/marker. This is partial biological reproducibility, not replication",
      "confirmed and not validation: concordant direction without matching significance/magnitude.",
      "A: paired bootstrap decision-curve delta-NB (SOFA + literature-derived microbiome vs SOFA alone), 25 thresholds 2-50%%,",
      "in-hospital-death primary endpoint (N=71, 27 events); %d/25 threshold evaluations exclude",
      "zero (no multiplicity correction across this exploratory grid). B: Spearman rho between",
      "each marker's abundance/index and mortality, computed independently per cohort; * nominal",
      "p<0.05 uncorrected. 16-marker directional concordance with the internal cohort: AGI_BGISEQ",
      "%d/16 (%.0f%%, one-sided binomial p=%.3f vs chance), PRJNA842425 %d/16 (%.0f%%, p=%.3f) --",
      "neither is distinguishable from a coin flip at this N. C: the internal cohort's frozen",
      ">=30%% Enterococcus-domination cut, applied unchanged; neither external cohort shows a",
      "significant association (Fisher exact p=%.3f AGI_BGISEQ, p=%.3f PRJNA842425); PRJNA842425",
      "has only 6/71 exposed patients, so its SOFA-adjusted estimate is imprecise."
    ), n_excl_dca, n_agi, 100*n_agi/16, bp_agi, n_prj, 100*n_prj/16, bp_prj, p_agi, p_prj),
    pt = BASE * 0.80, budget_in = FIGW - 0.3),
    # margin(b=...) on the overall title: on the first render, the 2-line
    # wrapped composition title and panel A's own title crowded together
    # with no visible gap (and panel A's y-axis title text intersected the
    # composition title's left edge) -- the annotation title's grob did not
    # reserve extra vertical room for its second wrapped line. An explicit
    # bottom margin fixes it without depending on plot_layout() height
    # allocation (which governs PANEL rows, not the annotation title area).
    theme = theme(plot.title = element_text(size = BASE * 1.00, face = "bold", margin = margin(b = BASE * 0.9)),
                  plot.caption = element_text(size = BASE * 0.80, hjust = 0))
  )

out <- save_manuscript_fig(combined, "Lean_FigS4_external_transportability_v2", width = FIGW, height = 20)
cat(sprintf("Wrote %s / %s / %s\n", out$svg, out$pdf, out$png))
cat("[21_lean_figS4_external_transportability_v2.R done]\n")
