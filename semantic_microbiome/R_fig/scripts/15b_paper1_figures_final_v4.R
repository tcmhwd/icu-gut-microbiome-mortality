#!/usr/bin/env Rscript
# 15b_paper1_figures_final_v3.R -- M3 fold-local-preprocessing-correction
# repoint (2026-10-02, FINAL_LEAN_SUBMISSION_2026-10-02_v3 package). Copied
# from 15b_paper1_figures_final_v2.R (DO NOT edit that script -- the v2
# package depends on its current output). See
# reports/paper1/FINAL_LEAN_SUBMISSION_2026-10-02_v3/QC/M3_FOLD_LOCAL_METHOD_AUDIT_v3.md
# for the full old-vs-new numbers. This copy only:
#   (a) repoints Fig2 panel B (DCA) and panel C (Brier) at the v3 nested-CV
#       outputs (script 55_v3, script 59_v3);
#   (b) repoints Fig3 panel A's C0-referenced block at the v3 bootstrap-delta
#       output (script 45_v3; C1 row still comes from its only variant, V0,
#       since the literature-derived model is unaffected by the M3
#       correction), and its SOFA-referenced block at script 61_v3;
#   (c) reorders Fig3 panel B so the SOFA-referenced row renders above the
#       APACHE-referenced row (SOFA is the principal external severity
#       comparator), and relabels the panel B legend "Primary (SOFA)" /
#       "Secondary (APACHE)" -> "SOFA reference" / "APACHE II reference"
#       (panel B's own data, PRJNA842425, is M1-only and unaffected by the
#       M3 correction -- this is a pure relabel/reorder);
#   (d) repoints Fig4 panels A/C (survival, script 42_v3 -> phase_x_survival_models_v4.csv)
#       and panel B's own source file (phase_x_survival_time_split_v4.csv).
# No other statistic, CI, p-value, data point, axis range, or label is
# changed. Fig1 (M1/clinical/SOFA cohort-flow content only, no M3
# dependency) is deliberately NOT rebuilt here -- that whole block is
# removed from this copy; the v3 submission package reuses v2's
# Final_Fig1_cohort_flow_v2.{svg,pdf,png} unchanged (copied, not re-rendered).
# Writes NEW files only, with a _v3 suffix so nothing under v2's
# figures_{svg,pdf,png}/ is overwritten: Final_Fig2_discrimination_dca_brier_v4,
# Final_Fig3_incremental_discrimination_v4, Final_Fig4_time_to_event_v4.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(patchwork)
  library(readr); library(stringr); library(forcats); library(tibble)
})
source("/home/tcmhwd/microbiome/ICU_public/semantic_microbiome/R_fig/scripts/00_setup_theme.R")
stage_source <- function(path) invisible(NULL)  # this script never re-stages source copies
# enlarge the value labels drawn inside hs_forest() (p labels) without touching any data mark
bump_text <- function(p, size) { for (i in seq_along(p$layers)) if (inherits(p$layers[[i]]$geom, "GeomText")) p$layers[[i]]$aes_params$size <- size; p }
`%||%` <- function(a, b) if (is.null(a)) b else a

CLIN_REP <- file.path(CLIN, "reports")
MRV2 <- file.path(SEM, "reports", "methodological_revision_v2")

log_con <- file(file.path(DIR_LOG, "15b_paper1_figures_final_v4.log"), open = "wt")
sink(log_con, split = TRUE)
cat("=== 15b_paper1_figures_final_v4.R started", format(Sys.time()), "===\n")

flow_box <- function(x, y, w, h, label, fill = "white", border = "grey30", size = 4.5) {
  list(
    geom_rect(aes(xmin = x - w/2, xmax = x + w/2, ymin = y - h/2, ymax = y + h/2),
              fill = fill, colour = border, linewidth = 0.5),
    geom_text(aes(x = x, y = y, label = label), size = size, lineheight = 0.95)
  )
}
flow_arrow <- function(x, y1, y2) {
  geom_segment(aes(x = x, xend = x, y = y1, yend = y2), arrow = arrow(length = unit(0.12, "cm")), colour = "grey30", linewidth = 0.4)
}
# strwrap-based title/label wrapping idiom (established convention, reused
# from R_fig/scripts/11_pairwise_and_group_timing_figure.R) -- used across
# this script's figures to fix edge-clipped titles/legends (verification
# audit C3).
wrap <- function(x, w = 60) paste(strwrap(x, width = w), collapse = "\n")

# ---------------------------------------------------------------------------
# House-style font-size fix (2026-10-01, FONT_SIZE_FIX_main_figs.md; see
# ~/.claude/playbooks/r_ggplot_house_style.md section "180mm 폭에서 실제
# 몇 pt인가"). PASS requires every text element to render >=11.5pt (target
# 12pt) AFTER the saved figure is shrunk to the 180mm submission width:
# final_pt = configured_pt * (180 / (ggsave_width_in * 25.4)).
#
# hs_text_bump() is a PARTIAL theme() override (not a complete theme like
# theme_manuscript()/theme_bw()), applied to a whole patchwork via the `&`
# operator so it broadcasts to every subplot without wiping out any other
# theme()/legend.position/panel.grid tweak already layered onto that panel.
# This matters because hs_forest() (~/R/house_style.R, used for several
# panels below) applies its OWN complete theme, theme_house(base_size=11)
# fixed, internally -- that call is never reached by this project's global
# theme_set(theme_manuscript()) in 00_setup_theme.R, so raising
# theme_manuscript()'s base_size would do nothing for those panels, and
# appending a second COMPLETE theme (theme_manuscript(base_size=X)) after a
# panel's own theme()-tweaks would discard those tweaks (e.g. legend.position
# = "top"). A partial, absolute-pt override avoids both problems and gives
# every panel in a figure -- hs_forest-built or plain ggplot -- IDENTICAL
# pt sizes per element type, computed from theme_manuscript()'s own rel()
# ratios (00_setup_theme.R) so the numbers verify against the same formula
# used for the rest of the house style.
#
# plot.caption is the one deliberate deviation: theme_manuscript()'s own
# ratio for it is rel(0.65), the smallest in the hierarchy, meant for
# de-emphasised footnotes. Final_FigS_mdro_detection's panel A caption is a
# one-line marker-legend note (not a footnote) and enforcing the full 0.65
# ratio at a layout-safe width would require base_size ~29 (see that
# figure's row in FONT_SIZE_FIX_main_figs.md), which breaks its dense
# 14-row forest panel. Raised to match legend.text's rel(0.80) instead --
# a readability choice, not an oversight; recorded here because it is the
# only figure in this pass that uses plot.caption at all.
hs_text_bump <- function(base_size) {
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

# P1_Fig1's 3 panels are theme_void() flow diagrams (deliberately no axis/
# legend/strip at all -- the diagram is entirely geom_rect()/geom_text()
# boxes and arrows). hs_text_bump() unconditionally sets axis.text/
# axis.title/legend.text/legend.title/strip.text to element_text(...), which
# would UN-BLANK theme_void()'s element_blank() for those slots and print
# spurious x/y axis ticks and labels over the diagram (confirmed by a first
# render: 23:46 job 640054's Final_Fig1_cohort_flow.png showed axis numbers
# and an "x"/"y" axis title bleeding through all 3 panels). Fig1 only ever
# uses plot.title/plot.subtitle, so it gets this narrower helper instead.
hs_title_bump <- function(base_size) {
  theme(
    plot.title    = element_text(size = base_size * 1.00, face = "bold"),
    plot.subtitle = element_text(size = base_size * 0.85)
  )
}

# ===========================================================================
# P1_Fig2: clinical vs microbiome discrimination + clinical utility
#          (adapted from Fig4)
# ===========================================================================
cat("\n--- P1_Fig2_clinical_microbiome_utility ---\n")
f_primary <- file.path(MRV2, "primary_results_table.csv")
# Propagation fix (2026-09-02, two independent reviewers): the DCA claim was
# recomputed correctly by script 55 (PAIRED nested-CV-corrected bootstrap
# delta-NB) but the corrected result was never propagated into this figure --
# Panel B kept sourcing the superseded marginal-CI-overlap statistic built
# from script 18's pre-nested-CV OOF (`phase_x_decision_curve_data_with_CI.csv`,
# `phase_x_microbiome_over_sofa_decision_curve.csv`), annotated "0/25". Both
# are replaced by the single corrected source below; the SOFA-referenced
# panel is dropped entirely (no nested-CV OOF exists for any SOFA-referenced
# S0-S3 candidate -- see script 55's docstring).
# M3 fold-local correction repoint (2026-10-02, v3 package): script 55 was
# rerun as 55_v3 against the fold-local M3 outputs (job 640326,
# M3_FOLD_LOCAL_METHOD_AUDIT_v3.md section 8). Same column layout as the old
# file; 4 cells flip (C2a loses significance at 3 thresholds, C2b gains it
# at the 30% threshold) and the full-grid exclude-zero count moves 9/75 ->
# 7/75 -- both read live from this file below, not hardcoded, so no caption
# text needs separate editing.
f_dca_paired <- file.path(CLIN_REP, "phase_x_decision_curve_nested_paired_v4.csv")
# Panel C repointed (owner decision 2026-09-05) from the pre-nested,
# no-CI single-split-layer point estimates (phase_x_calibration_summary.csv,
# script 31) to script 59's nested cross-validated Brier scores WITH
# percentile-bootstrap 95% CIs (phase_x_calibration_metrics.csv) -- the same
# nested layer as panels A/B, and TRIPOD+AI expects calibration reported
# with uncertainty.
# M3 fold-local correction repoint (2026-10-02, v3 package): script 59 was
# rerun as 59_v3 (job 640324); Brier values are essentially unchanged
# (C2a 0.1504->0.1504, C2b 0.1491->0.1493), same column layout.
f_calib302 <- file.path(CLIN_REP, "phase_x_calibration_metrics_v4.csv")
# Panel A: SOFA, Clinical AI (C0), Microbiome M1 alone -- AUROC with 95% CI
# on every bar. All three now come from the single-source, single-bootstrap-
# protocol CI file (script 56), which reproduces script 29's published SOFA
# and C0-direct intervals exactly as a precondition for writing its own C0
# out-of-fold and M1 values (see phase_x_panelA_discrimination_ci.md).
f_panelA <- file.path(CLIN_REP, "phase_x_panelA_discrimination_ci.csv")
for (f in c(f_primary, f_panelA, f_dca_paired, f_calib302)) stage_source(f)

panelA_src <- read_csv(f_panelA, show_col_types = FALSE)
stopifnot(setequal(panelA_src$bar, c("SOFA", "C0", "M1")))
panelA_dat <- panelA_src %>%
  mutate(model = recode(bar, SOFA = "SOFA",
                             C0   = "Clinical model",
                             M1   = "Literature-derived microbiome score"),
         CI_low = AUROC_CI_low, CI_high = AUROC_CI_high,
         # SOFA given its own "kind" (house-style fix, FIGURE_HOUSE_STYLE_AUDIT.md):
         # previously fell through to "clinical" and shared PAL$clinical with
         # Clinical AI (C0) on this same panel/axis -- colour only, no data change.
         kind = case_when(bar == "M1" ~ "microbiome", bar == "SOFA" ~ "sofa", TRUE ~ "clinical")) %>%
  arrange(AUROC) %>% mutate(model = factor(model, levels = model))

sofa_row <- panelA_src %>% filter(bar == "SOFA")
c0_row   <- panelA_src %>% filter(bar == "C0")
m1_row   <- panelA_src %>% filter(bar == "M1")

# House-style geom swap (2026-09-06, FIGURE_HOUSE_STYLE_AUDIT.md follow-up):
# hs_forest() replaces the hand-built vline+errorbarh+point stack. Same 3
# rows, same AUROC/CI values, same ascending order (hs_forest's default
# order="effect" reproduces panelA_dat's existing arrange(AUROC)); no p
# column exists in phase_x_panelA_discrimination_ci.csv, so hs_forest omits
# a p label automatically, exactly as before. The PAL$clinical/microbiome/
# sofa colour mapping (the house-style SOFA-collision fix) is preserved by
# overriding hs_forest's own group palette after the call, rather than
# accepting its default hs_colors() cycle -- content/semantics unchanged,
# only the geometry primitive is swapped.
panelA2 <- hs_forest(
    data.frame(label = panelA_dat$model, estimate = panelA_dat$AUROC,
               ci_low = panelA_dat$CI_low, ci_high = panelA_dat$CI_high,
               kind = panelA_dat$kind),
    xlab = "AUROC", ref = 0.5, group = "kind") +
  scale_colour_manual(values = c(clinical = PAL$clinical, microbiome = PAL$microbiome, sofa = PAL$sofa), guide = "none") +
  coord_cartesian(xlim = c(0.45, 0.85)) +
  labs(title = wrap("A. Same-cohort discrimination: AUROC (95% CI), n = 302, 69 deaths", w = 40))

dca_paired <- read_csv(f_dca_paired, show_col_types = FALSE) %>%
  mutate(model_short = case_when(
    str_detect(model, "^C1_")  ~ "Clinical + literature-derived microbiome",
    str_detect(model, "^C2a_") ~ "Clinical + data-driven microbiome, LR",
    str_detect(model, "^C2b_") ~ "Clinical + data-driven microbiome, CatBoost",
    TRUE ~ model))
# str_wrap()-ed (v2 render follow-up, job 640296): facet STRIP text does not
# auto-reflow the way an axis title does -- left unwrapped, these new longer
# descriptive strings ran off the right edge of the device (confirmed on
# that render). Embedding the line breaks here, before the level order is
# locked in, keeps the wrapped string as the actual factor level/strip label.
dca_paired$model_short <- str_wrap(dca_paired$model_short, width = 26)
dca_paired$model_short <- factor(dca_paired$model_short,
  levels = str_wrap(c("Clinical + literature-derived microbiome", "Clinical + data-driven microbiome, LR",
             "Clinical + data-driven microbiome, CatBoost"), width = 26))

n_thr_total <- nrow(dca_paired)
n_excl_total <- sum(dca_paired$CI_excludes_zero)
n_fav_cand <- sum(dca_paired$direction == "favors_candidate", na.rm = TRUE)
n_fav_c0 <- sum(dca_paired$direction == "favors_C0", na.rm = TRUE)

# Holm-Bonferroni step-down across the 6 prespecified cells (3 models x
# {10%, 30%}) -- same algorithm as script 55/build_tables.py, computed live
# here rather than hardcoded so the figure caption cannot drift from its source.
holm_adjust <- function(p) {
  m <- length(p); o <- order(p); adj <- numeric(m); running_max <- 0
  for (rnk in seq_along(o)) {
    idx <- o[rnk]
    running_max <- max(running_max, (m - rnk + 1) * p[idx])
    adj[idx] <- min(running_max, 1)
  }
  adj
}
pre_thr <- dca_paired %>% filter(prespecified_primary) %>%
  mutate(holm_p = holm_adjust(delta_NB_boot_p))
holm_p_min <- pre_thr %>% arrange(delta_NB_boot_p) %>% slice(1) %>% pull(holm_p)

cat(sprintf("Paired decision-curve delta-NB: %d/%d threshold-contrast evaluations exclude zero (%d favor candidate, %d favor C0); smallest prespecified nominal p=0.012 -> Holm p=%.3f\n",
            n_excl_total, n_thr_total, n_fav_cand, n_fav_c0, holm_p_min))

# Panel B redesigned (propagation fix, 2026-09-02): a paired delta_NB-vs-
# threshold plot with a zero reference line and CI ribbon, per model, so the
# mixed-direction result (low-threshold band favoring the candidate AND the
# single reversal at 0.24 favoring clinical-alone) is legible rather than
# averaged away. Significant points (CI excludes zero) are marked and
# coloured by direction; dotted vertical lines mark the two prespecified
# thresholds (10%, 30%).
panelB2 <- ggplot(dca_paired, aes(x = threshold, y = delta_NB)) +
  geom_ribbon(aes(ymin = delta_NB_CI_low, ymax = delta_NB_CI_high), fill = PAL$combined, alpha = 0.15) +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = c(0.10, 0.30), linetype = "dotted", colour = PAL$neutral, linewidth = 0.4) +
  geom_line(colour = PAL$combined, linewidth = 0.6) +
  geom_point(data = filter(dca_paired, CI_excludes_zero), aes(colour = direction, shape = direction), size = 2.6) +
  scale_colour_manual(values = c(favors_candidate = PAL$combined, favors_C0 = PAL$leaked),
                       labels = c(favors_candidate = "favors\nmicrobiome-\naugmented",
                                  favors_C0 = "favors\nclinical-alone\n(reversal)"),
                       name = "CI excludes zero") +
  # greyscale-safety fix (2026-09-14, house-style follow-up pass): the two
  # directions were colour-only (both circles); shape now carries the same
  # distinction redundantly (triangle = reversal) so the panel survives a
  # greyscale print run, matching the filled/open convention used elsewhere
  # in this figure set.
  scale_shape_manual(values = c(favors_candidate = 16, favors_C0 = 17),
                      labels = c(favors_candidate = "favors\nmicrobiome-\naugmented",
                                 favors_C0 = "favors\nclinical-alone\n(reversal)"),
                      name = "CI excludes zero") +
  facet_wrap(~model_short, ncol = 1) +
  # font-size-fix follow-up: title/subtitle wrap()-ed, legend labels above
  # str_wrap()-ed -- at base_size 18 these ran off the right edge of the
  # 8in-wide image (confirmed on job 640060's render). v2 render follow-up
  # (job 640296): the v2 label-pass strings are longer still, and
  # "microbiome-augmented" is one unbreakable (no-space) token that
  # str_wrap() cannot split -- it was overflowing the device edge uncut.
  # Replaced with a manual hard break (favors\nmicrobiome-\naugmented) rather
  # than relying on str_wrap() for this one token. y-axis title shortened
  # "clinical model" -> "clinical" (unambiguous given the panel's own title/
  # x-axis already say "candidate" vs the comparator) to pull its rotated-
  # text footprint back down near the v1 length, after the full-length
  # phrase was found overlapping Panel C's "Brier score" label in the shared
  # left margin on that same render.
  labs(title = wrap("B. Paired decision-curve net-benefit difference (95% CI)", w = 40),
       subtitle = wrap("Markers: 95% CI excludes zero. Dotted lines: 10% and 30% thresholds", w = 45),
       x = "Threshold probability", y = "Delta NB (candidate - clinical), net true positives per patient") +
  theme(legend.position = "top")

# C6 (22-panel) was dropped from Panel C (readiness-audit Fig2 item 3,
# 2026-09-05): it is not defined anywhere in Paper 1 and is not otherwise
# used in this manuscript (the model set discussed throughout is
# C0/C1/C2a/C2b, plus M1 alone as reported below); the nested-CV source file
# does not carry a C6 row at all, so no filter is needed for it here.
#
# Panel C repoint (owner decision 2026-09-05): source is now the nested
# cross-validated Brier scores WITH percentile-bootstrap 95% CIs (script 59,
# phase_x_calibration_metrics.csv) instead of the pre-nested single-split
# point estimates without CIs (script 31, phase_x_calibration_summary.csv)
# that panel A/B are not on. The ranking of C1 and C2a SWAPS versus the old
# panel (old: C2b < C0 < C1 < C2a i.e. C1 better than C2a; new nested: C2b <
# C2a < C0 < C1 i.e. C2a better than C1) -- expected, not an error: script 59
# documents C0/C1/C2a/C2b and M1-alone as different fitted objects on
# different designs, with no guarantee that a single-split ranking is
# preserved once refit under nested CV.
#
# M1 alone is INCLUDED (the file's fifth row) rather than dropped, because
# (a) panel A of this same figure already reports M1 alone alongside SOFA
# and C0, so its calibration is directly relevant scientific content here
# (Brier 0.174 vs ~0.15 for the nested C0/C1/C2a/C2b cluster, slope 0.50 --
# M1 alone calibrates markedly worse, which is worth showing); (b) this
# project's own house style already mixes primary (nested) and secondary
# (single-split) estimates on one panel with an explicit visual/legend
# distinction rather than omission (see P1_Fig3's nested-vs-single-split
# colour split). M1 alone is visually distinguished as its own legend
# category: open circle marker + dashed CI whisker (vs filled circle + solid
# CI for the four nested-design models), and is named as a different design
# ("single-split OOF; different design") in the point-shape legend and in
# the caption below -- it must not be read as a fifth point on the same
# nested design as C0/C1/C2a/C2b.
# model_code keeps the short internal code purely for logical comparisons
# downstream (e.g. the C0 colour highlight below); model_short is the
# reader-facing display label actually plotted on the x-axis -- decoupled so
# relabeling model_short can never silently break a code-string comparison.
calib302 <- read_csv(f_calib302, show_col_types = FALSE) %>%
  mutate(model_code = case_when(
    str_detect(model, "C2b") ~ "C2b",
    str_detect(model, "C0")  ~ "C0", str_detect(model, "C1")  ~ "C1",
    str_detect(model, "C2a") ~ "C2a",
    str_detect(model, "M1_alone") ~ "M1 alone", TRUE ~ model),
    model_short = case_when(
    model_code == "C2b" ~ "+ Data-driven CatBoost",
    model_code == "C0"  ~ "Clinical",
    model_code == "C1"  ~ "+ Lit.-derived",
    model_code == "C2a" ~ "+ Data-driven LR",
    model_code == "M1 alone" ~ "Lit.-derived (alone)", TRUE ~ model_code),
    is_single_split = str_detect(design, "single-split")) %>%
  arrange(Brier)
calib302$model_short <- factor(calib302$model_short, levels = calib302$model_short)
nested_only <- filter(calib302, !is_single_split)

# C1 fix (verification audit): the previous coord_cartesian(ylim=c(0.14,0.156))
# bar chart visually exaggerated Brier differences of <=0.005 into large
# apparent bar-height ratios (a standard truncated-bar-chart reviewer
# complaint). Replaced with a Cleveland dot plot: points only (no bar/lollipop
# stem, so no length encodes magnitude), a non-zero-based but honestly-labeled
# axis, and the exact value printed at every point so the small differences
# are legible without visual exaggeration. Error bars now show the
# percentile-bootstrap 95% CI (script 59) rather than a bare point estimate.
brier_span <- max(nested_only$Brier) - min(nested_only$Brier)
panelC2 <- ggplot(calib302, aes(x = model_short, y = Brier, colour = model_code == "C0")) +
  geom_errorbar(aes(ymin = Brier_CI_low, ymax = Brier_CI_high, linetype = is_single_split),
                width = 0.15, linewidth = 0.55, colour = "grey40", show.legend = FALSE) +
  geom_point(aes(shape = is_single_split), size = 4) +
  geom_text(aes(y = Brier_CI_high, label = sprintf("%.4f", Brier)), vjust = -0.9, size = 5.1, colour = "grey20", show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = PAL$clinical, `FALSE` = PAL$combined), guide = "none") +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), name = NULL,
                      labels = c(`FALSE` = str_wrap("nested CV, 5x5 (4 combined models)", 18),
                                 `TRUE` = str_wrap("single-split out-of-fold (literature-derived score, alone)", 18))) +
  scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "22"), guide = "none") +
  scale_y_continuous(limits = c(min(calib302$Brier_CI_low) - 0.006, max(calib302$Brier_CI_high) + 0.012)) +
  labs(title = wrap("C. Brier score with 95% bootstrap CI (lower = better)", w = 40), x = NULL, y = "Brier score") +
  # x-axis categories angled (v2 render follow-up, job 640297): the v2
  # descriptive category text ("+ Data-driven CatBoost" etc.), even wrapped
  # to 2 lines, was still wider than its per-category horizontal slot across
  # these 5 categories and visibly overlapped its neighbours on that render.
  # Angling single-line labels (the standard ggplot fix for this) uses the
  # diagonal instead of forcing more lines/width into the same per-category
  # slot.
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1, size = rel(1.0)),
        panel.grid.major.y = element_line(colour = "grey90"),
        legend.position = "bottom")

# Panel B now facets 3 models (vs the old 2-facet SOFA/C0 layout), so it is
# stacked full-width below Panel A rather than squeezed side-by-side with
# Panel C (propagation fix, 2026-09-02); figure heightened accordingly.
# Font-size fix (2026-10-01, FONT_SIZE_FIX_main_figs.md): width 10->8in,
# base_size 11->18 (hs_text_bump(), broadcast with `&` so it reaches
# panelA2, which is built via hs_forest()/theme_house() and is NOT reached
# by this project's global theme_set(theme_manuscript())). Brier-value
# labels (line above) scaled 3.1->5.1mm, ratio 18/11 = 1.636. Height NOT
# scaled proportionally with width (a first attempt at 10.08in -- the plain
# 8/10 ratio -- left adjacent panels' y-axis titles ("Brier score" / "Delta
# NB (candidate - C0)...") overlapping each other in the shared left margin,
# confirmed on job 640060's render) -- raised to 13.5in instead for margin
# room; titles/subtitle/legend labels in all 3 panels also wrap()-ed /
# str_wrap()-ed (they ran off the right edge of the image at base_size 18,
# also confirmed on that render).
# Width nudged 8.0->8.5in (v2 render follow-up, job 640296): matches the
# width already used for Fig3's own forest panels and gives the longer v2
# descriptive strings (wrapped facet strips, legend) a bit more horizontal
# room; still clears the 11.5pt floor at base_size 18 (legend.text, the
# smallest ratio used here = 18*0.80*180/(8.5*25.4) = 11.97pt).
fig2 <- (panelA2 / panelB2 / panelC2 + plot_layout(heights = c(0.42, 1.75, 0.80))) & hs_text_bump(18)
out2 <- save_manuscript_fig(fig2, "Final_Fig2_discrimination_dca_brier_v4", width = 8.5, height = 14.5)
cat("[P1_Fig2 done]\n")

# ===========================================================================
# P1_Fig3: incremental value (the primary null)
# ===========================================================================
cat("\n--- P1_Fig3_incremental_value ---\n")
# M3 fold-local correction repoint (2026-10-02, v3 package): script 45 was
# rerun as 45_v3 (jobs 640315/640316, M3_FOLD_LOCAL_METHOD_AUDIT_v3.md
# sections 1-7). The v3 output carries ALL preprocessing variants (V0/V1/V2/
# V2S/V3) in one long file, one row per (preprocessing_variant, model); V3 is
# the pre-declared primary variant (every outcome-blind preprocessing step
# fold-local). C1 (literature-derived, M1) is unaffected by the M3
# correction and was not rerun under V1-V3 -- its only row is V0, which is
# kept as that contrast's value (identical to the old single-variant file).
f_nested <- file.path(CLIN_REP, "phase_x_c1_c2_nested_bootstrap_delta_v4.csv")
# Repointed 2026-09-06 (script41_nested_rebuild_DESIGN.md section 6): the
# SOFA-referenced contrasts now come from script 61's nested-CV rebuild of
# 41_phase_x_microbiome_over_sofa.py, not the superseded single-split
# stacked-OOF design (TS1 row 15 / revision_log.md item 64).
# M3 fold-local correction repoint (2026-10-02, v3 package): script 61 was
# rerun as 61_v3 (job 640322, audit section 9) on top of 45_v3's FeatureSource
# construction, so this file is already the single primary-variant output --
# same column layout as the old file, no variant filter needed here.
f_sofa_boot <- file.path(CLIN_REP, "phase_x_microbiome_over_sofa_nestedcv_paired_bootstrap_v4.csv")
f_mde_md <- file.path(CLIN_REP, "phase_x_minimum_detectable_effect.md")
for (f in c(f_nested, f_sofa_boot, f_mde_md)) stage_source(f)

# Reader-facing label pass (2026-10-02, v2 package): row labels now name
# ONLY the microbiome strategy being added (the block/facet heading already
# establishes the reference model), per the nomenclature table -- "C1 vs C0"
# phrasing is dropped from the row text itself, not from the data/ordering.
# Primary-variant filter, v4 FINAL CLOSURE PASS FIX (previously this block still filtered
# C2a to "V3", silently plotting the superseded pre-strict-LR estimate under a v4 legend --
# caught by independent Codex/Fable audit, both file:line-cited; see
# QC/FINAL_CHANGELOG_v4.md and QC/DATA_DRIVEN_LR_STRICT_METHOD_AUDIT_v4.md). C1 keeps
# its only variant (V0, unaffected by the M3 correction); C2a takes the FINAL headline
# variant V4_STRICT (fully tuning-local data-driven LR); C2b (CatBoost, unaffected by the
# strict-LR change) keeps V3.
nested_raw <- read_csv(f_nested, show_col_types = FALSE)
nested <- nested_raw %>%
  filter((model == "C1_clinical_plus_M1_LiteratureCore" & preprocessing_variant == "V0") |
         (model == "C2a_clinical_plus_M3_LR" & preprocessing_variant == "V4_STRICT") |
         (model == "C2b_clinical_plus_M3_CatBoost" & preprocessing_variant == "V3")) %>%
  transmute(contrast = recode(model,
                              C1_clinical_plus_M1_LiteratureCore = "Literature-derived microbiome",
                              C2a_clinical_plus_M3_LR = "Data-driven microbiome, LR",
                              C2b_clinical_plus_M3_CatBoost = "Data-driven microbiome, CatBoost"),
            delta_AUROC, delta_AUROC_CI_low, delta_AUROC_CI_high, delta_AUROC_boot_p,
            source = "Added to clinical model (nested cross-validation)")
stopifnot(nrow(nested) == 3)
# Grouped, not effect-size, row order (readiness-audit Fig3 item 5, 2026-09-14
# follow-up pass): within each comparator family the three models always
# appear in the fixed literature-derived -> data-driven LR -> data-driven
# CatBoost order, regardless of which CSV row order the source file happens
# to use -- explicit factor levels rather than trusting file order.
#
# VISUAL top-to-bottom order fix (coordinator instruction, 2026-10-02): the
# task spec names literature-derived/LR/CatBoost as the on-screen top-to-
# bottom reading order, not merely a data/insertion order. ggplot's
# categorical y-axis draws the FIRST factor level at the BOTTOM and later
# levels upward, and hs_forest() (order="none") derives its level order from
# unique(label) in first-APPEARANCE order across the row sequence actually
# fed to it (not from any pre-set factor levels() attribute, which R's
# unique() on a factor ignores in favour of appearance order) -- so getting
# "Literature-derived microbiome" to render at the TOP requires it to be the
# LAST of the 3 labels first encountered, i.e. rows arranged
# CatBoost -> LR -> Literature-derived here (reverse of the data/family
# narrative order above). This arrange()-reversal is cosmetic only: it does
# not change which family a row belongs to, nor any plotted value.
nested$contrast <- factor(nested$contrast,
  levels = c("Data-driven microbiome, CatBoost", "Data-driven microbiome, LR", "Literature-derived microbiome"))
nested <- arrange(nested, contrast)
nested$family <- "Added to clinical model"

# Row labels typeset by hand (readiness-audit Fig3 item 2, 2026-09-05) rather
# than derived from the raw model string, which otherwise surfaces
# underscore-joined variable names ("SOFA_plus_M3CatBoost vs SOFA alone") on
# the figure. Same three microbiome-strategy labels as the clinical-model
# block above, plus a 4th "Clinical model" row (the SOFA+clinical-model
# contrast, S3ref) as its own added-information row within this block.
sofa_boot <- read_csv(f_sofa_boot, show_col_types = FALSE) %>%
  filter(str_detect(reference, "S0_SOFA_alone")) %>%
  mutate(contrast = recode(model,
                           S1_SOFA_plus_M1 = "Literature-derived microbiome",
                           S2a_SOFA_plus_M3LR = "Data-driven microbiome, LR",
                           S2b_SOFA_plus_M3CatBoost = "Data-driven microbiome, CatBoost",
                           S3ref_SOFA_plus_clinicalAI = "Clinical model")) %>%
  transmute(contrast, delta_AUROC, delta_AUROC_CI_low, delta_AUROC_CI_high, delta_AUROC_boot_p,
            source = "Added to SOFA (nested cross-validation)")
# Visual order fix (coordinator instruction, 2026-10-02): desired top-to-
# bottom is Literature-derived / LR / CatBoost / Clinical model (bottom).
# "Clinical model" is unique to this block (not present in `nested` at all),
# so for it to land at the BOTTOM -- i.e. get the LOWEST global first-
# appearance index once bind_rows() combines both blocks -- it must be the
# very first row of the WHOLE combined dataset (see forest_dat below, where
# sofa_boot is now bound BEFORE nested). Internal row order here is
# therefore Clinical model -> CatBoost -> LR -> Literature-derived (reverse
# reading order), matching nested's own reversed arrangement above so the 3
# shared labels get consistent global indices either way.
sofa_boot$contrast <- factor(sofa_boot$contrast,
  levels = c("Clinical model", "Data-driven microbiome, CatBoost",
             "Data-driven microbiome, LR", "Literature-derived microbiome"))
sofa_boot <- arrange(sofa_boot, contrast)
sofa_boot$family <- "Added to SOFA"

# Grouped by comparator family (readiness-audit Fig3 item 5, resolved
# 2026-09-14): rows no longer interleave the two families by effect size.
# Row GROUPING (which family each row belongs to) is unaffected by bind
# order -- within each family, rows keep the fixed literature-derived/data-
# driven-LR/data-driven-CatBoost(/clinical-model) order set above, not
# ranked by delta_AUROC, and facet_grid(rows=vars(family)) always draws
# "Added to clinical model" above "Added to SOFA" because that panel order
# is set explicitly below via forest_dat$family's own factor levels --
# NEITHER of those depends on bind_rows() argument order.
# hs_forest() is called below with order="none" so it does not re-sort rows
# by effect size.
# NOTE (v2 label pass): the row-label TEXT is now intentionally identical
# across the two families for the 3 shared microbiome strategies (e.g.
# "Literature-derived microbiome" appears once in each block) -- this is
# safe and deliberate, not a collision: hs_forest()'s panel="family" facets
# each family into its own strip, so each label only ever renders its own
# family's row in its own panel; the pre-hs_forest contrast-relabeling step
# that used to lock in a single GLOBAL factor order (needed when every row
# label was a unique "X vs Y" string) is dropped here because duplicate
# level strings are not permitted by factor(levels=); hs_forest's own
# order="none" + internal unique(label) level construction reproduces the
# same fixed row order inside each facet without that step.
#
# bind_rows() ARGUMENT ORDER fix (coordinator instruction, 2026-10-02): this
# is the one place where bind order DOES matter. hs_forest()'s row-position
# logic (unique(label) in first-appearance order across the full row
# sequence, see comment above) assigns the GLOBAL bottom-to-top index by
# when a label is FIRST seen -- and "Clinical model" (unique to sofa_boot)
# only gets the lowest index (bottom position, as the coordinator specified)
# if it is literally the first row of the combined dataset. sofa_boot is
# therefore bound FIRST (not nested): its already-reversed internal row
# order (Clinical model -> CatBoost -> LR -> Literature-derived, set above)
# fixes the global index sequence 1-4 in that order; nested's own 3 rows
# then contribute no NEW labels (all already claimed), so its internal
# order is inert for positioning purposes, but is kept in the same reversed
# arrangement for readability. Expected on-screen result, both blocks,
# top-to-bottom: Literature-derived / LR / CatBoost (/ Clinical model,
# bottom, SOFA block only) -- confirm against the rendered PNG after this
# script runs.
forest_dat <- bind_rows(sofa_boot, nested)
forest_dat$family <- factor(forest_dat$family, levels = c("Added to clinical model", "Added to SOFA"))

# MDE reference lines removed from the figure (owner decision 2026-09-05,
# FIGURE_LEGENDS_DRAFT.md Part 6 note 3 / wording rule 4): the MDE is an
# algebraic restatement of each contrast's own CI width (implied_SE = CI
# width / 3.92), not independent evidence of underpowering, and a legend
# that has to explain that a plotted reference line "is not independent
# evidence" is a sign the line should not be on the figure. In its place,
# the power-inversion sentence (what cohort size the definitive study would
# need) is read live from phase_x_minimum_detectable_effect.md's own
# power-inversion table -- the CSV (phase_x_minimum_detectable_effect.csv)
# carries only the MDE columns, not the N_required/events_required columns,
# which exist solely in the .md's second table -- and is carried in the
# figure's manifest note and the manuscript legend, not plotted.
mde_md_lines <- read_lines(f_mde_md)
pwr_row <- mde_md_lines[str_detect(mde_md_lines,
  fixed("| C1 vs C0 | nested-CV-corrected (script 46) | 0.0106 | 302 | 69 | 0.02 |"))]
stopifnot(length(pwr_row) == 1)
pwr_fields <- str_trim(str_split(pwr_row, "\\|")[[1]])
pwr_fields <- pwr_fields[pwr_fields != ""]
n_required_c1_002 <- as.numeric(pwr_fields[7])
events_required_c1_002 <- as.numeric(pwr_fields[8])
cat(sprintf("Power inversion (C1 vs C0, nested-CV-corrected, target dAUROC=0.02, 80%% power): N_required=%d, events_required=%d\n",
            n_required_c1_002, events_required_c1_002))

# House-style geom swap (2026-09-06): hs_forest() replaces the hand-built
# vline+errorbarh+point+geom_text stack. Same rows/values/CI/p; the printed p
# wording moves from "p=0.338" to house style's "p = 0.338" (hs_fmt_p), same
# underlying value. The PAL$combined/clinical primary-vs-secondary colour
# mapping is preserved by overriding hs_forest's group palette after the
# call. Zero line kept at the same position (ref=0); hs_forest draws it as
# its own standard dashed grey line rather than the previous prominent solid
# leaked-red line -- a chrome-only restyling, the line's value is unchanged.
# Grouped-row-order fix (2026-09-14, readiness-audit Fig3 item 5): order was
# "effect" (ascending delta_AUROC), which interleaved the primary
# (C0-referenced) and secondary (SOFA-referenced) families by effect size.
# order="none" now respects forest_dat's own row order (set above: the three
# "Added to clinical model" contrasts, then the four "Added to SOFA" ones,
# each family in its fixed literature-derived/data-driven-LR/data-driven-
# CatBoost(/clinical-model) order, never re-ranked). panel="family"
# additionally facets the two families into separate strips with free y-space
# -- a visible divider between them, and a second, greyscale-safe channel
# (the strip's text label) for the family distinction that colour alone
# provided before. (family itself is now tagged directly on nested/sofa_boot
# above, not parsed back out of `source` text.)
p3 <- hs_forest(
    data.frame(label = forest_dat$contrast, estimate = forest_dat$delta_AUROC,
               ci_low = forest_dat$delta_AUROC_CI_low, ci_high = forest_dat$delta_AUROC_CI_high,
               p = forest_dat$delta_AUROC_boot_p, source = forest_dat$source, family = forest_dat$family),
    # Shortened to "ΔAUROC after addition" (v2 render follow-up, job
    # 640297): the longer "Change in AUROC after addition" phrasing was
    # clipped at the device edge on that render -- this panel also has TWO
    # different references (clinical model and SOFA, via the family facet),
    # so the task brief's alternate wording ("...versus clinical reference")
    # would be inaccurate for the SOFA-referenced block; this compact form
    # keeps the addition-not-replication framing without naming either
    # single reference.
    xlab = "ΔAUROC after addition", ref = 0, order = "none",
    group = "source", panel = "family")
# x-axis right margin widened past hs_forest()'s own internal xlim_hi (font-
# size fix follow-up, job 640060): bump_text() only resizes the p-label
# GeomText layer, it does not touch hs_forest()'s xlim_hi, which is a fixed
# fraction of the data span (span*0.32) sized for hs_forest's own hardcoded
# 2.7mm p-label text -- at our 5.1mm bump the labels ("p = 0.164" etc.) were
# clipped mid-digit by the panel edge. Recomputes hs_forest()'s own rng_hi/
# span formula and scales the margin fraction up by the same font ratio
# (5.1/2.7), plus a 1.4x safety factor (text-width-to-data-unit conversion
# is only approximate here, so this errs generous rather than re-clipping).
span3 <- max(c(forest_dat$delta_AUROC_CI_high, forest_dat$delta_AUROC, 0), na.rm = TRUE) -
         min(c(forest_dat$delta_AUROC_CI_low, forest_dat$delta_AUROC, 0), na.rm = TRUE)
rng_hi3 <- max(c(forest_dat$delta_AUROC_CI_high, forest_dat$delta_AUROC, 0), na.rm = TRUE)
# Safety factor raised 1.4->2.2 (v2 render follow-up, job 640297): even
# with the top legend removed, the p-value labels ("p = 0.107" etc.) were
# still clipped mid-digit on that render -- empirically widened further
# rather than re-deriving hs_forest()'s internal text-width-to-data-unit
# approximation exactly.
xlim_hi3 <- rng_hi3 + span3 * 0.32 * (5.1 / 2.7) * 2.2
# Colour legend dropped (v2 render follow-up, job 640296): with the v2
# descriptive family/source strings, the top-row legend ("Added to clinical
# model (nested cross-validation)" / "Added to SOFA (nested cross-
# validation)", two keys side by side) overflowed the right edge of the
# device -- confirmed on that render (legend text and p-value labels both
# clipped). The colour/source grouping is already fully disclosed by the
# panel="family" facet strip text immediately next to each block of rows
# (same "Added to clinical model" / "Added to SOFA" wording), so colour here
# is a redundant, purely decorative confirmation, not the reader's only way
# to tell the two blocks apart -- dropping its guide (not its colouring)
# removes the overflow without losing information, the same redundant-
# legend-suppression pattern already used elsewhere in this script (e.g.
# Panel B's "predictor family" legend in the MDRO figure).
p3 <- bump_text(p3, 5.1) +
  scale_colour_manual(values = c("Added to clinical model (nested cross-validation)" = PAL$combined,
                                 "Added to SOFA (nested cross-validation)" = PAL$clinical), guide = "none") +
  scale_x_continuous(limits = c(NA, xlim_hi3)) +
  labs(title = wrap("A. Internal: nested cross-validation (N = 302; 69 in-hospital deaths)", w = 40)) +
  theme(legend.position = "none")

# ---------------------------------------------------------------------------
# P1_Fig3 Panel B (NEW, manuscript freeze 2026-09-14, owner-decision default
# already fixed by the task brief itself: "integrate Fig_external_delta_AUROC
# into Main Figure 3", not a standalone 6th main figure): external
# replication in PRJNA842425. SOFA+M1 (primary, in-hospital-death endpoint,
# 27 events) is visually prominent; APACHE+M1 (secondary) is muted grey, not
# just a smaller point, so the primary/secondary distinction survives
# greyscale. Source: critical_care_external/prjna842425_incremental_results.csv
# (already reviewed by fable-auditor + Codex; values NOT recomputed here,
# only read and plotted -- this script never fits a model).
# ---------------------------------------------------------------------------
CC_EXT <- file.path(SEM, "reports", "paper1", "critical_care_external")
f_ext <- file.path(CC_EXT, "prjna842425_incremental_results.csv")
stage_source(f_ext)
ext_prim <- read_csv(f_ext, show_col_types = FALSE) %>%
  filter(endpoint == "in_hospital_death_PRIMARY")
# "(PRJNA842425)" dropped from the row text itself (v2 render follow-up, job
# 640297): it duplicates the panel B title, which already names the cohort
# ("B. External application: PRJNA842425..."); the v2 descriptive row text
# is already longer than v1's "SOFA + M1 vs SOFA alone" short form, and the
# redundant cohort tag on every row was widening the shared left-label
# column (patchwork aligns p3/p3b's left margins) enough to push p3b's own
# top legend ("Secondary (APACHE)") off the right edge of the device on
# that render.
# Row order / legend wording fix (coordinator instruction, 2026-10-02): SOFA
# is the principal external severity comparator, so the SOFA-referenced row
# must render ABOVE the APACHE-referenced row, and the "Primary"/"Secondary"
# legend labels are replaced with "SOFA reference"/"APACHE II reference" --
# this is a pure relabel/reorder of the existing PRJNA842425 values (ext_prim
# above), not a data change; the underlying HR/CI/p values are untouched.
ext_dat <- ext_prim %>%
  transmute(label = if_else(severity_score == "SOFA", "SOFA + literature-derived microbiome",
                             "APACHE + literature-derived microbiome"),
            estimate = delta_AUROC, ci_low = delta_AUROC_lo, ci_high = delta_AUROC_hi,
            p = delta_AUROC_boot_p,
            priority = if_else(severity_score == "SOFA", "SOFA reference", "APACHE II reference"))
# hs_forest(order="none") places the FIRST-appearing row at the BOTTOM of the
# categorical y-axis (same mechanism documented above for panel A's
# bind_rows() order) -- so APACHE must be the first row fed to hs_forest
# (bottom) and SOFA the last (top).
ext_dat <- ext_dat[order(match(ext_dat$priority, c("APACHE II reference", "SOFA reference"))), ]
ext_dat$label <- factor(ext_dat$label, levels = rev(ext_dat$label))
ext_dat$priority <- factor(ext_dat$priority, levels = c("SOFA reference", "APACHE II reference"))

# Same p-label margin widening as p3 above, using ext_dat's own estimate/
# ci_low/ci_high/ref=0.
span3b <- max(c(ext_dat$ci_high, ext_dat$estimate, 0), na.rm = TRUE) -
          min(c(ext_dat$ci_low, ext_dat$estimate, 0), na.rm = TRUE)
rng_hi3b <- max(c(ext_dat$ci_high, ext_dat$estimate, 0), na.rm = TRUE)
xlim_hi3b <- rng_hi3b + span3b * 0.32 * (5.1 / 2.7) * 2.2
p3b <- bump_text(hs_forest(ext_dat, xlab = "ΔAUROC after addition", ref = 0, order = "none",
                  group = "priority"), 5.1) +
  scale_colour_manual(values = c("SOFA reference" = PAL$microbiome, "APACHE II reference" = "grey55"), name = NULL) +
  scale_x_continuous(limits = c(NA, xlim_hi3b)) +
  labs(title = wrap("B. External application: PRJNA842425 (N = 71; 27 in-hospital deaths)", w = 40),
       subtitle = "In-sample two-term model, not out-of-fold") +
  theme(legend.position = "top")

# Font-size fix (2026-10-01, FONT_SIZE_FIX_main_figs.md): width 10.5->8.0in,
# base_size 11->18 (hs_text_bump(), broadcast with `&`; both p3 and p3b are
# hs_forest()/theme_house()-built, so the global theme_set() bump would not
# reach either without this). The hs_forest() internal p-value labels
# (bump_text() calls above) scaled 3.1->5.1mm, ratio 18/11 = 1.636, matching
# the base_size ratio; their x-axis margin widened separately, see above.
# Height NOT scaled proportionally with width (a first attempt at 6.32in --
# the plain 8/10.5 ratio -- clipped p3's facet strip labels ("SOFA
# refer[ence]") and left its 7 forest rows + legend + title cramped,
# confirmed on job 640060's render) -- raised to 11in for row/strip room.
# Width nudged 8.0->8.5in (second follow-up, job 640067's render): the
# legend ("...SOFA-referenced...") and facet strip labels ("Clinical-model
# refe[rence]") on the far right were still clipped by the device edge even
# after the margin/wrap fixes above. 8.5in still clears 11.5pt for every
# element (legend.text, the smallest used ratio here, = 18*0.80*scale =
# 12.01pt at this width -- see FONT_SIZE_FIX_main_figs.md).
fig3 <- (p3 / p3b + plot_layout(heights = c(1, 0.4))) & hs_text_bump(18)
# height nudged 11->13in (third follow-up, job 640076's render): the
# rotated facet strip label "Clinical-model reference" (its row-group has
# only 3 rows, less vertical room than "SOFA reference"'s 4 rows) was still
# clipped at the top of its own facet group. More total height gives that
# smaller group more room without touching width (which is already near
# the 11.5pt floor for legend.text at base_size 18).
out3 <- save_manuscript_fig(fig3, "Final_Fig3_incremental_discrimination_v4", width = 8.5, height = 13)
cat("[P1_Fig3 done]\n")

# ===========================================================================
# P1_Fig4: time-to-event competing risk (reuse Fig7 content, re-rendered
#          under P1 naming)
# ===========================================================================
cat("\n--- P1_Fig4_time_to_event_competing_risk ---\n")
# M3 fold-local correction repoint (2026-10-02, v3 package): script 42 was
# rerun as 42_v3 (job 640323, audit section 8). Panels A/C's M3-CatBoost (and
# M3-LR) hazard ratios move modestly; qualitative conclusions unchanged
# (Fine-Gray M3-CatBoost alone stays significant, +SOFA stays non-significant
# in both panels; M1/SOFA rows unaffected). Same column layout as the old
# files.
f_surv <- file.path(CLIN_REP, "phase_x_survival_models_v4.csv")
f_split <- file.path(CLIN_REP, "phase_x_survival_time_split_v4.csv")
for (f in c(f_surv, f_split)) stage_source(f)
surv  <- read_csv(f_surv, show_col_types = FALSE)
split <- read_csv(f_split, show_col_types = FALSE)
PRIMARY <- "Spec B-CS CORRECTED (PRIMARY): cause-specific, in-hospital death"

pick_focal <- function(d) {
  d %>% filter(if_else(str_detect(covariate_set, "^SOFA alone"), term == "SOFA", str_detect(term, "^logit_")))
}
# Reader-facing term substitution (v2 label pass, 2026-10-02): replaces the
# M1/M3-LR/M3-CatBoost model-code tokens embedded in covariate_set strings
# with the nomenclature-table descriptive forms. Order matters only in that
# "M3-CatBoost"/"M3-LR" are matched before the bare "M1" token (M1 is not a
# substring of either, so the order is not actually load-bearing, but kept
# defensive/explicit). str_wrap()-ed afterwards so the long combined strings
# (e.g. "Data-driven microbiome, CatBoost + SOFA-stratified (tertiles)") wrap
# to multiple lines rather than forcing the row-label column wide.
# Shortened to a compact parenthetical form (v2 render follow-up, job
# 640296): the full "Literature-derived microbiome"/"Data-driven microbiome,
# LR"/"Data-driven microbiome, CatBoost" phrasing, combined with Panel A's
# "alone"/"+ SOFA"/"+ SOFA-stratified (tertiles)" suffixes, wrapped to 3
# lines per row and badly overlapped adjacent rows across this panel's ~10
# forest rows on that render. "microbiome" is dropped from the term itself
# (every row in this HR forest is already a microbiome- or SOFA-term row,
# per the panel's own legend/"microbiome term" key) -- this is the compact
# fallback form the task brief explicitly allows when full descriptive text
# does not fit.
descr_term <- function(x) {
  x <- str_replace(x, fixed("M3-CatBoost"), "Data-driven (CatBoost)")
  x <- str_replace(x, fixed("M3-LR"), "Data-driven (LR)")
  x <- str_replace(x, fixed("M1"), "Lit.-derived")
  x
}
# Scale fix (coordinator instruction, 2026-10-02): microbiome-term HRs
# switched from HR_per_1_unit (1 unit of the constructed logit score) to
# HR_per_1_SD/HR_per_1_SD_CI_low/HR_per_1_SD_CI_high -- already-computed
# columns in this same source CSV, already used for the same quantity in
# Table S3, not a new analysis. SOFA stays on HR_per_1_unit (identical to
# "per 1-point" for SOFA, an integer score, so no column switch needed for
# that term). HR_plot/CI_low_plot/CI_high_plot are the values actually
# fed to hs_forest below; p_value is untouched because a Wald test on a
# linear-model coefficient is scale-invariant (rescaling the covariate by a
# constant does not change whether its coefficient is zero), confirmed by
# this source file carrying a single p_value column shared by both the
# per-unit and per-SD HR columns for the same row.
s_dat <- surv %>% filter(specification == PRIMARY, !is.na(HR_per_1_unit)) %>%
  pick_focal() %>%
  mutate(kind = ifelse(term == "SOFA", "clinical", "microbiome"),
         HR_plot = ifelse(term == "SOFA", HR_per_1_unit, HR_per_1_SD),
         CI_low_plot = ifelse(term == "SOFA", HR_CI_low, HR_per_1_SD_CI_low),
         CI_high_plot = ifelse(term == "SOFA", HR_CI_high, HR_per_1_SD_CI_high),
         # Row-label clarity fix (readiness-audit Fig4 item 2, 2026-09-14):
         # the raw covariate_set string "SOFA(stratified)" names a modelling
         # choice (SOFA entered as a stratification factor -- tertiles, per
         # script 42's SE_type string) that was previously undefined on the
         # figure itself. Display-label substitution only; HR/CI values and
         # row identity are unchanged.
         covariate_set = str_replace(covariate_set, fixed("SOFA(stratified)"), "SOFA-stratified (tertiles)"),
         covariate_set = str_wrap(descr_term(covariate_set), width = 24),
         covariate_set = factor(covariate_set, levels = rev(unique(covariate_set))))
stopifnot(!any(duplicated(s_dat$covariate_set)))
stopifnot(!any(is.na(s_dat$HR_plot)), !any(is.na(s_dat$CI_low_plot)), !any(is.na(s_dat$CI_high_plot)))
n_deaths <- surv %>% filter(specification == PRIMARY) %>% pull(N_events) %>% unique() %>% na.omit()
n_death_evt <- if (length(n_deaths) >= 1) n_deaths[1] else NA

# House-style geom swap (2026-09-06): hs_forest() replaces the hand-built
# vline+errorbar+point stack for panels A/B. Neither panel has a p column
# (audit-noted gap, unchanged), so hs_forest omits a p label automatically,
# exactly as before. Panel A's row order (SOFA alone -> alone models ->
# +SOFA models -> +SOFA(stratified) models) is a deliberate grouping, not an
# effect-size sort, so hs_forest is called with order="none" on rows
# pre-arranged to match the existing factor's own level order (covariate_set
# already carries the correct order via levels=rev(unique(...)) above) --
# this reproduces the identical axis order, not an approximation of it.
s_dat_lvls <- levels(s_dat$covariate_set)
s_dat_ord  <- s_dat[match(s_dat_lvls, as.character(s_dat$covariate_set)), ]
p4a <- hs_forest(
    data.frame(label = as.character(s_dat_ord$covariate_set), estimate = s_dat_ord$HR_plot,
               ci_low = s_dat_ord$CI_low_plot, ci_high = s_dat_ord$CI_high_plot, kind = s_dat_ord$kind),
    # Trailing "-- scale differs by term, see caption" clause dropped (render
    # follow-up, job 640299): it was clipped at the device edge. The
    # figure-wide caption below (plot_annotation) already states this
    # explicitly ("Panels A and C: microbiome terms are HR per 1-SD
    # increase; SOFA is HR per 1-point increase"), so no information is lost.
    xlab = "Hazard ratio (log scale, 95% CI)", ref = 1, log_x = TRUE,
    group = "kind", order = "none") +
  scale_colour_manual(values = c(clinical = PAL$clinical, microbiome = PAL$microbiome), name = NULL,
                      labels = c(clinical = "SOFA term", microbiome = "microbiome term")) +
  labs(title = wrap("A. Cause-specific Cox (primary): 69 in-hospital deaths, 233 competing", w = 38),
       subtitle = wrap("one focal coefficient per model; adjusted rows show the microbiome term", w = 45))

# NOT switched to per-SD (checked, coordinator instruction 2026-10-02):
# unlike phase_x_survival_models.csv (panels A/C's source), this file
# (phase_x_survival_time_split.csv) has NO HR_per_1_SD/CI columns at all --
# confirmed by reading its header and by a bounded grep for "HR_per_1_SD"
# across clinical_ai/reports/*.csv, which returns only
# phase_x_survival_models.csv. Computing a per-SD HR for this
# early/late-split specification would require fitting a new model (the SD
# of each period's own risk set), which is out of this task's scope ("not a
# new analysis"). Panel B therefore stays on HR_per_1_unit for both terms --
# flagged explicitly in its own x-axis title below so it is not read as
# being on the same scale as panels A/C.
sp_dat <- split %>% filter(specification == PRIMARY, !is.na(HR_per_1_unit)) %>%
  pick_focal() %>% mutate(kind = ifelse(term == "SOFA", "clinical", "microbiome"),
                           covariate_set = str_wrap(descr_term(covariate_set), width = 24))

# Panel B's covariate_set was never releveled (plain character), so its
# original axis order was ggplot's own default: alphabetical. hs_forest's
# order="label" reproduces that default exactly.
p4b <- hs_forest(
    data.frame(label = sp_dat$covariate_set, estimate = sp_dat$HR_per_1_unit,
               ci_low = sp_dat$HR_CI_low, ci_high = sp_dat$HR_CI_high, kind = sp_dat$kind,
               period = sp_dat$period),
    xlab = "Hazard ratio per 1 unit (log scale, 95% CI)", ref = 1, log_x = TRUE,
    group = "kind", order = "label") +
  scale_colour_manual(values = c(clinical = PAL$clinical, microbiome = PAL$microbiome), name = NULL,
                      labels = c(clinical = "SOFA term", microbiome = "microbiome term")) +
  facet_wrap(~ period) +
  # Title/subtitle made descriptive, not interpretive (readiness-audit Fig4
  # item 1, 2026-09-14): "SOFA attenuates; microbiome estimates remain
  # imprecise" and "not evidence of a time trend" are conclusions, which
  # belong in the external legend (already stated there) rather than printed
  # on the image per house-style rule 4.
  labs(title = wrap("B. Cause-specific HRs before and after the median time to death (19 d)", w = 38),
       # Trailing "-- not per-SD, see caption" clause dropped (render
       # follow-up, job 640299): clipped at the device edge. The
       # figure-wide caption already states Panel B's per-unit exception
       # explicitly, so this shorter axis title loses no information.
       x = "Hazard ratio per 1 unit of logit score (log scale, 95% CI)",
       y = NULL)

# ---------------------------------------------------------------------------
# Panel C (NEW, owner decision 2026-09-05): the secondary Fine-Gray
# subdistribution-hazard specification, drawn rather than carried only in a
# long legend paragraph -- wording rule 9 requires four facts to travel with
# this figure wherever it is presented, and a legend that must hold all four
# reads as text-in-a-caption (FIGURE_LEGENDS_DRAFT.md Part 6 note 2). Source
# is the same phase_x_survival_models.csv panel A/B already read (`surv`),
# filtered to specification "Spec B-FG CORRECTED (SECONDARY): Fine-Gray
# subdistribution" -- no new file, no new analysis.
# ---------------------------------------------------------------------------
FG_SECONDARY <- "Spec B-FG CORRECTED (SECONDARY): Fine-Gray subdistribution"
# BUG FIX (2026-10-02, v4 final sweep): "M3-LR + SOFA" was missing from this
# filter list (carried over from an earlier v3 panel-C design where it may
# not yet have been computed) -- the manuscript's outlier-diagnostic
# reframing (QC/DATA_DRIVEN_LR_STRICT_METHOD_AUDIT_v4.md, Table S10) now
# explicitly discusses this row's value (HR 0.447, p=0.172), so the figure
# must show it too. stopifnot bound updated 5->6 accordingly.
fg_rows <- surv %>%
  filter(specification == FG_SECONDARY,
         covariate_set %in% c("M3-CatBoost alone", "M3-CatBoost + SOFA",
                               "M1 alone", "M1 + SOFA",
                               "M3-LR alone", "M3-LR + SOFA"),
         str_detect(term, "^logit_"))
stopifnot(nrow(fg_rows) == 6)

# Visual encoding: colour carries the specification (alone = microbiome-green,
# + SOFA = combined-purple, matching this script's PAL semantics elsewhere);
# a connecting arrow between a model's "alone" and "+ SOFA" points makes the
# CatBoost attenuation (fact i) legible as one pair rather than two unrelated
# rows; shape + linetype (both driven by the source file's own
# `bootstrap_stability` flag) together mark the two M1 rows (fact ii) as
# UNSTABLE -- open marker AND a dashed, not solid, CI whisker -- a redundant,
# greyscale-safe encoding so they cannot be mistaken for the stable rows.
# Reader-facing axis labels (v2 pass): model_grp's display text is now the
# descriptive form (embedded "\n" line breaks, since these are categorical
# y-axis tick labels, not a wrappable title/legend string) instead of the
# M1/M3-LR/M3-CatBoost codes; factor level ORDER is unchanged (CatBoost,
# literature-derived, LR -- the pre-existing row order, not re-sorted).
fg_dat <- fg_rows %>%
  transmute(
    model_grp = case_when(str_detect(covariate_set, "^M3-CatBoost") ~ "Data-driven\nmicrobiome,\nCatBoost",
                           str_detect(covariate_set, "^M1")         ~ "Literature-\nderived\nmicrobiome",
                           str_detect(covariate_set, "^M3-LR")      ~ "Data-driven\nmicrobiome, LR"),
    adjustment = factor(if_else(str_detect(covariate_set, "SOFA"), "+ SOFA", "alone"),
                        levels = c("alone", "+ SOFA")),
    # Scale fix (coordinator instruction, 2026-10-02): all 5 rows here are
    # microbiome terms (no SOFA row in this panel), so all switch to
    # HR_per_1_SD/CI -- same already-computed columns as panel A, confirmed
    # present and non-missing for this Fine-Gray specification too. p_value
    # is unchanged (Wald test on the coefficient is scale-invariant; this
    # source file carries one p_value column shared by both HR scales).
    HR = HR_per_1_SD, CI_low = HR_per_1_SD_CI_low, CI_high = HR_per_1_SD_CI_high, p_value,
    unstable = bootstrap_stability == "UNSTABLE") %>%
  mutate(model_grp = factor(model_grp, levels = c("Data-driven\nmicrobiome,\nCatBoost",
                                                   "Literature-\nderived\nmicrobiome",
                                                   "Data-driven\nmicrobiome, LR")),
         # offset widened 0.14->0.20 (font-size fix follow-up): at the bigger
         # annotation font the HR=.../p=... row labels collided between
         # adjacent rows, confirmed on job 640060's render.
         y_pos = as.numeric(model_grp) + if_else(adjustment == "alone", 0.20, -0.20),
         stability_lab = factor(if_else(unstable,
                                         "UNSTABLE (sandwich-SE fallback, not a bootstrap CI)",
                                         "stable (bootstrap 95% CI)"),
                                levels = c("stable (bootstrap 95% CI)",
                                           "UNSTABLE (sandwich-SE fallback, not a bootstrap CI)")))

fg_seg <- fg_dat %>%
  group_by(model_grp) %>%
  filter(n() == 2) %>%
  summarise(y1 = y_pos[adjustment == "alone"], y2 = y_pos[adjustment == "+ SOFA"],
            x1 = HR[adjustment == "alone"], x2 = HR[adjustment == "+ SOFA"], .groups = "drop")

# (these 5 objects are defined but not otherwise referenced downstream in
# this script -- updated to match the new model_grp strings for consistency
# only; dead code carried over unchanged in behaviour from v1.)
fg_catboost_alone <- fg_dat %>% filter(model_grp == "Data-driven\nmicrobiome,\nCatBoost", adjustment == "alone")
fg_catboost_sofa  <- fg_dat %>% filter(model_grp == "Data-driven\nmicrobiome,\nCatBoost", adjustment == "+ SOFA")
fg_m1_alone <- fg_dat %>% filter(model_grp == "Literature-\nderived\nmicrobiome", adjustment == "alone")
fg_m1_sofa  <- fg_dat %>% filter(model_grp == "Literature-\nderived\nmicrobiome", adjustment == "+ SOFA")
fg_m3lr <- fg_dat %>% filter(model_grp == "Data-driven\nmicrobiome, LR")

p4c <- ggplot(fg_dat, aes(x = HR, y = y_pos)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = PAL$neutral, linewidth = 0.4) +
  geom_segment(data = fg_seg, aes(x = x1, xend = x2, y = y1, yend = y2), inherit.aes = FALSE,
               colour = "grey55", linewidth = 0.5,
               arrow = arrow(length = unit(0.09, "cm"), type = "closed")) +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high, colour = adjustment, linetype = stability_lab),
                 height = 0.1, linewidth = 0.6) +
  geom_point(aes(colour = adjustment, shape = stability_lab), size = 3, stroke = 1.2) +
  geom_text(aes(x = CI_high, label = sprintf("HR=%.3f, p%s", HR, ifelse(p_value < 0.001, "<0.001", sprintf("=%.3f", p_value)))),
            hjust = -0.08, size = 4.6, colour = "grey30") +
  scale_x_log10(expand = expansion(mult = c(0.08, 0.80))) +
  scale_y_continuous(breaks = 1:3, labels = levels(fg_dat$model_grp), limits = c(0.5, 3.5)) +
  scale_colour_manual(values = c("alone" = PAL$microbiome, "+ SOFA" = PAL$combined), name = "specification") +
  scale_shape_manual(values = c("stable (bootstrap 95% CI)" = 16,
                                "UNSTABLE (sandwich-SE fallback, not a bootstrap CI)" = 1),
                     # wrap width narrowed 22->13 (fourth follow-up, job 640076's render):
                     # this legend sits at the very bottom-right of an 8in-wide figure and
                     # was still clipped by the device edge at width=22.
                     name = "bootstrap stability", labels = function(x) str_wrap(x, width = 13)) +
  scale_linetype_manual(values = c("stable (bootstrap 95% CI)" = "solid",
                                   "UNSTABLE (sandwich-SE fallback, not a bootstrap CI)" = "22"),
                        guide = "none") +
  labs(title = wrap("C. Secondary specification: Fine-Gray subdistribution hazard", w = 28),
       subtitle = wrap("Arrows link each model without and with SOFA adjustment", w = 45),
       x = "Hazard ratio per 1-SD increase (log scale, 95% CI)", y = NULL) +
  theme(legend.position = "bottom", legend.box = "vertical")

# Widened 9.0->10in and title wrapped via wrap() (verification audit C3) --
# panel B's title was previously cut off ("...remain impre"). Heightened
# 8.6->11.6in for the new panel C (owner decision 2026-09-05).
# Font-size fix (2026-10-01, FONT_SIZE_FIX_main_figs.md): width 10.0->7.5in,
# base_size 11->17 (hs_text_bump(), broadcast with `&`; p4a/p4b are
# hs_forest()/theme_house()-built). Panel C's HR/p annotation (geom_text
# above) scaled 3.0->4.6mm, ratio 17/11 = 1.545, matching the base_size
# ratio; its row y-offset widened 0.14->0.20 (above) for the same reason.
# Height NOT scaled proportionally with width (a first attempt at 8.25in --
# the plain 7.5/10 ratio -- badly overlapped panel A's ~10 categorical row
# labels on top of each other and collided panel C's HR=.../p=... text
# labels between rows, confirmed on job 640060's render) -- raised to 18in;
# panel A alone has up to 10 forest rows and needs real vertical room at a
# 17pt axis-text font. Width nudged 7.5->8.0in (second follow-up, job
# 640067's render): panel C's HR=.../p=... labels and panel A/B's legends
# were still clipped by the device's right edge -- widened the panel C
# x-expansion margin too (0.55->0.80 mult, above) and this still clears
# 11.5pt for legend.text (the smallest ratio used, 0.80) = 17*0.8*scale =
# 12.05pt at 8.0in (see FONT_SIZE_FIX_main_figs.md).
# Panel A height share raised further, and total figure height raised 18->26in
# (v2 render follow-up, job 640296): even with the pre-existing height=18in
# tuning (comment above), the v2 descriptive row labels (str_wrap()-ed,
# 1-3 lines each after descr_term() substitution) badly overlapped adjacent
# rows across panel A's ~10-row forest on that render -- these labels are
# longer than the v1 "M1 alone"/"M1 + SOFA" style single-line strings that
# height=18in with heights=c(1,...) was originally tuned for. Height is free
# to raise without affecting the 180mm-print-width point-size floor (that
# formula depends only on WIDTH, not height), so panel A's share is raised
# substantially rather than shrinking the label text back down.
fig4 <- (p4a / p4b / p4c + plot_layout(heights = c(1.6, 0.7, 0.7))) & hs_text_bump(17)
# Effect-scale disclosure, REVISED (coordinator instruction, 2026-10-02,
# supersedes the original v2 label-pass note that used to be here). The
# original pass found this script plotted HR_per_1_unit for every row and,
# correctly per a prior R1-05 review finding, declined to mislabel that
# per-unit-of-logit-score value as "per-SD". The coordinator's resolution,
# after reviewing that flag, is to actually SWITCH the microbiome rows onto
# the already-computed HR_per_1_SD/CI columns (not a new analysis -- the
# same columns already used for this quantity in Table S3) rather than
# leave the mislabel risk in place. Implemented above: panel A's microbiome
# rows (s_dat, via HR_plot/CI_low_plot/CI_high_plot) and panel C's rows
# (fg_dat, all 5 rows are microbiome terms) now plot HR_per_1_SD; SOFA stays
# on HR_per_1_unit (equivalent to "per 1-point" for an integer score, so no
# column switch needed there). p_value is unchanged in both panels -- a
# Wald test on a linear coefficient is invariant to rescaling the covariate,
# and this source file carries one shared p_value column for both HR scales
# per row.
#
# Panel B (sp_dat, from phase_x_survival_time_split.csv) is the ONE
# exception: that file has NO HR_per_1_SD columns at all (checked directly
# -- its header has only HR_per_1_unit/HR_CI_low/HR_CI_high; a bounded grep
# for "HR_per_1_SD" across clinical_ai/reports/*.csv returns only
# phase_x_survival_models.csv, panels A/C's source). Computing a per-SD HR
# for the early/late time-split specification would mean fitting a new
# model, outside this task's scope ("not a new analysis"). Panel B
# therefore still plots HR_per_1_unit for its microbiome rows, flagged
# explicitly in its own x-axis title ("...of logit score...-- not per-SD,
# see caption") rather than silently left inconsistent with panels A/C.
fig4 <- fig4 + plot_annotation(
  caption = wrap("Panels A and C: microbiome terms are HR per 1-SD increase; SOFA is HR per 1-point increase. Panel B: both terms remain HR per 1 unit of the plotted term (1 point for SOFA; 1 unit of the constructed microbiome logit score) -- per-SD HRs are not available for this early/late time-split specification.", w = 68),
  theme = theme(plot.caption = element_text(size = 17 * 0.80, colour = "grey30", hjust = 0))
)
out4 <- save_manuscript_fig(fig4, "Final_Fig4_time_to_event_v4", width = 8.0, height = 26)
cat("[P1_Fig4 done]\n")

# ---------------------------------------------------------------------------
# Fig S (MDRO/VRE detection, Final_FigS_mdro_detection) is out of scope for
# this pass and is intentionally NOT built by this copy. Fig1 is also NOT
# built by this copy (unaffected by the M3 correction; see header comment) --
# the v3 submission package reuses v2's Final_Fig1_cohort_flow_v2 unchanged.
cat("\n=== 15b_paper1_figures_final_v3.R finished: 3 main figures (Fig2/3/4) ===\n")
sink()
