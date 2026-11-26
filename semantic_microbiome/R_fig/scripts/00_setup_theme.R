#!/usr/bin/env Rscript
# 00_setup_theme.R -- shared ggplot2 theme, palette, and save helper for all
# manuscript figure regeneration scripts. Sourced by 01_/02_, never run alone.
# Does NOT read or write any analysis data.

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(patchwork)
  library(svglite)
})

# house-style redirect (2026-09-06, FIGURE_HOUSE_STYLE_AUDIT.md): pulls in
# hs_font()/HS_PAL from the machine-wide ~/R/house_style.R so this project's
# theme_manuscript()/PAL can borrow the house style's purely presentational
# chrome (Arial, dotted edge-coloured gridlines, no panel border) without
# renaming theme_manuscript()/PAL themselves or touching their semantics
# (PAL's clinical/microbiome/combined/... keys are unchanged). Does not
# switch this project onto hs_box()/hs_scatter()/etc or hs_save() -- those
# have different call signatures and this is a chrome-only migration.
source("~/R/house_style.R")

# ---------------------------------------------------------------------------
# paths (all outputs go under R_fig/, nothing outside is ever written)
# ---------------------------------------------------------------------------
ROOT       <- "/home/tcmhwd/microbiome/ICU_public"
SEM        <- file.path(ROOT, "semantic_microbiome")
CLIN       <- file.path(ROOT, "clinical metadata", "clinical_ai")
OUT        <- file.path(SEM, "R_fig")
DIR_SVG    <- file.path(OUT, "figures_svg")
DIR_PDF    <- file.path(OUT, "figures_pdf")
DIR_PNG    <- file.path(OUT, "figures_png")
DIR_SRC    <- file.path(OUT, "source_data_used")
DIR_LOG    <- file.path(OUT, "logs")
DIR_MAN    <- file.path(OUT, "manifest")

# ---------------------------------------------------------------------------
# journal-ready theme: white background, consistent sizing, minimal chartjunk
# Chrome migrated onto ~/R/house_style.R (2026-09-06, FIGURE_HOUSE_STYLE_AUDIT.md):
# base_family -> hs_font() (Arial; fontconfig substitutes metric-compatible
# Liberation Sans), gridlines -> HS_PAL$edge dotted, panel border replaced by
# an HS_PAL$edge axis line (house style's "no panel border" convention).
# Sizing, legend/strip styling, and the white background are unchanged --
# those are this project's own choices, not part of the presentational
# migration.
# ---------------------------------------------------------------------------
theme_manuscript <- function(base_size = 11) {
  theme_bw(base_size = base_size, base_family = hs_font()) +
    theme(
      panel.background  = element_rect(fill = "white", colour = NA),
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = HS_PAL$edge, linewidth = 0.3, linetype = "dotted"),
      panel.border      = element_blank(),
      axis.line         = element_line(colour = HS_PAL$edge, linewidth = 0.4),
      axis.text         = element_text(colour = "black", size = rel(0.85)),
      axis.title        = element_text(colour = "black", size = rel(0.95)),
      plot.title        = element_text(size = rel(1.0), face = "bold", hjust = 0),
      plot.subtitle     = element_text(size = rel(0.85), colour = "grey30"),
      plot.caption      = element_text(size = rel(0.65), colour = "grey40", hjust = 0),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.key        = element_rect(fill = "white", colour = NA),
      legend.text       = element_text(size = rel(0.8)),
      legend.title      = element_text(size = rel(0.85)),
      strip.background  = element_rect(fill = "grey92", colour = "grey40", linewidth = 0.3),
      strip.text        = element_text(colour = "black", size = rel(0.85), face = "bold")
    )
}
theme_set(theme_manuscript())

# ---------------------------------------------------------------------------
# consistent, non-saturated palette used across ALL figures
# ---------------------------------------------------------------------------
PAL <- list(
  clinical    = "#2C5F8A",   # muted blue  -- clinical-only models/variables
  microbiome  = "#3E8E5A",   # muted green -- microbiome-only models/markers
  combined    = "#7A4F9C",   # muted purple -- clinical+microbiome combined
  leaked      = "#B03A2E",   # muted red   -- flagged/leaked/invalid results
  corrected   = "#3E8E5A",   # same muted green -- corrected/leakage-free
  neutral     = "#707070",   # grey        -- reference lines, chance level
  survivor    = "#4C78A8",
  death       = "#B0342D",
  highlight   = "#D4A017",   # muted gold -- annotations
  sofa        = "#D98A4A",   # muted orange (HS_PAL$groups[2]) -- SOFA specifically, added
                             # 2026-09-06 (FIGURE_HOUSE_STYLE_AUDIT.md): SOFA had no key of its
                             # own and fell through to PAL$clinical, colliding with Clinical AI
                             # (C0) in P1_Fig2 Panel A, where both are plotted on one axis
  seq5        = c("#2C5F8A", "#3E8E5A", "#D4A017", "#7A4F9C", "#B03A2E")
)

# ---------------------------------------------------------------------------
# save a ggplot (or patchwork) object as svg + pdf + png, verify non-empty
# ---------------------------------------------------------------------------
save_manuscript_fig <- function(plot, name, width = 7, height = 5, dpi = 300) {
  svg_path <- file.path(DIR_SVG, paste0(name, ".svg"))
  pdf_path <- file.path(DIR_PDF, paste0(name, ".pdf"))
  png_path <- file.path(DIR_PNG, paste0(name, ".png"))
  ggsave(svg_path, plot = plot, width = width, height = height, device = svglite::svglite)
  ggsave(pdf_path, plot = plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  for (p in c(svg_path, pdf_path, png_path)) {
    sz <- file.info(p)$size
    if (is.na(sz) || sz == 0) stop(sprintf("save_manuscript_fig: %s was not written or is empty", p))
  }
  invisible(list(svg = svg_path, pdf = pdf_path, png = png_path))
}

# helper: copy a source data file into source_data_used/ (never modifies original)
stage_source <- function(path) {
  if (!file.exists(path)) {
    warning(sprintf("stage_source: MISSING source file: %s", path))
    return(invisible(NULL))
  }
  dest <- file.path(DIR_SRC, basename(path))
  file.copy(path, dest, overwrite = TRUE)
  invisible(dest)
}

# ---------------------------------------------------------------------------
# readable-label lookup for raw microbiome/feature variable codes (added
# 2026-09-14, submission-figure follow-up pass: house-style rule "remove raw
# programming variable names where the mapping is unambiguous"). Covers the
# specific codes found on FigS2 panel b, FigS3 panels a/b, FigS5 panel c,
# FigS8 and FigS9 panel b during that pass -- deliberately a plain lookup
# (exact match, case-sensitive), not a regex/heuristic transform, so a name
# not in the table is left untouched and visibly flaggable rather than
# guessed. Genus/species names are not italicised here (would need ggtext,
# a new dependency); plain text only.
READABLE_MARKER <- c(
  Enterococcus_genus_relabund = "Enterococcus relative abundance",
  Enterococcus_rel_abundance = "Enterococcus relative abundance",
  enterococcus_relabund = "Enterococcus relative abundance",
  facultative_to_obligate_anaerobe_balance = "Facultative-to-obligate anaerobe balance",
  obligate_anaerobe_preservation = "Obligate-anaerobe preservation",
  alpha_diversity_shannon = "Shannon diversity",
  shannon = "Shannon diversity",
  species_richness = "Observed species richness",
  observed_richness = "Observed species richness",
  Prevotella_genus_relabund = "Prevotella relative abundance",
  pathobiont_burden = "Pathobiont burden",
  pathobiont_MAG_burden = "Pathobiont MAG burden",
  Faecalibacterium_genus_relabund = "Faecalibacterium relative abundance",
  Proteobacteria_phylum_relabund = "Proteobacteria relative abundance",
  inv_simpson = "Inverse Simpson index",
  dominance = "Dominance (max. single-species abundance)",
  top5_cumulative_relabund = "Top-5 cumulative relative abundance",
  enterobacterales_proteobacteria_relabund = "Enterobacterales/Proteobacteria relative abundance",
  anaerobic_metabolism_score = "Anaerobic metabolism score",
  Klebsiella_genus_relabund = "Klebsiella relative abundance",
  Erysipelotrichaceae_family_relabund = "Erysipelotrichaceae relative abundance",
  Enterococcus_faecium_species_relabund = "Enterococcus faecium relative abundance",
  Christensenellaceae_family_relabund = "Christensenellaceae relative abundance",
  Bacteroides_genus_relabund = "Bacteroides relative abundance",
  Escherichia_genus_relabund = "Escherichia relative abundance",
  Bacteroidetes_Firmicutes_ratio = "Bacteroidetes:Firmicutes ratio",
  Erysipelotrichaceae_family_relabund = "Erysipelotrichaceae relative abundance",
  Clostridium_leptum_species_relabund = "Clostridium leptum relative abundance",
  observed_species_richness = "Observed species richness"
)
#' Map a vector of raw feature/marker codes to manuscript-readable labels.
#' Unrecognised codes pass through unchanged (never guessed).
hs_readable_marker <- function(x) {
  hit <- x %in% names(READABLE_MARKER)
  x[hit] <- READABLE_MARKER[x[hit]]
  x
}

cat(sprintf("[00_setup_theme] theme/palette/helpers loaded. Output root: %s\n", OUT))
