# -*- coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
#  house_style.R — 모든 프로젝트 공용 ggplot2 그림 스타일
#
#  사용:  source("~/R/house_style.R")
#
#  규약 (2026-09-04 확정):
#    1. 그림은 R / ggplot2로 그린다.
#    2. 글꼴은 Arial.
#    3. 가능한 경우 통계 수치(R²·p·ρ·n)를 그림 안에 함께 넣는다.
#
#  팔레트와 그림 타입은 PD 코호트 프로젝트에서 정착시킨 house style을 계승한 것.
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ggplot2); library(ragg); library(systemfonts); library(scales)
})

#' 선택 패키지 존재 확인. 없으면 base 구현으로 폴백한다.
hs_has <- function(pkg) requireNamespace(pkg, quietly = TRUE)

# ── 팔레트 ───────────────────────────────────────────────────────────────────
HS_PAL <- list(
  red   = "#C0504D",   # 양(+) 방향 / 강조
  blue  = "#4A7FC1",   # 음(−) 방향
  cream = "#F1F0E9",   # 배경 톤
  ink   = "#16150F",   # 본문 글자
  gray  = "#8A8578",   # 보조 글자 (통계 라벨, 캡션)
  edge  = "#DAD6C9",   # 축선 / 격자
  track = "#E6E2D6",   # ranked bar 트랙
  mid   = "#EFE9DD",   # 발산형 스케일 중앙 / 신뢰띠
  groups = c("#4A7FC1", "#D98A4A", "#4AAE8A", "#B25FA8",
             "#C0504D", "#7E9BC4", "#A38B6B", "#5F8A7A")
)

#' 군 색상 벡터. n개를 순환해서 돌려준다.
hs_colors <- function(n) rep(HS_PAL$groups, length.out = n)

# ── 글꼴 ─────────────────────────────────────────────────────────────────────
#' 본문 글꼴.
#' 이 기계에는 2026-09-22부터 Arial 실물이 있다 (~/.fonts/Arial*.ttf, 4종).
#' 없는 기계에서도 fontconfig가 Liberation Sans(metric-compatible, 글자 폭이 동일)로
#' 대체하므로 "Arial"을 그대로 쓰면 된다 — 레이아웃은 어느 쪽이든 같다.
#' @param cjk TRUE면 한글용 폰트. Arial에는 한글 글리프가 없다. 다만 ragg는
#'   글리프 단위 fallback을 하므로 FALSE로 둬도 대개 한글이 깨지지는 않는다 —
#'   한글 비중이 크면 명시하는 편이 렌더링이 안정적이다.
hs_font <- function(cjk = FALSE) {
  if (cjk) {
    fams <- unique(systemfonts::system_fonts()$family)
    for (f in c("Noto Sans CJK KR", "Malgun Gothic", "NanumGothic", "AppleGothic")) {
      if (f %in% fams) return(f)
    }
    warning("한글 폰트를 못 찾았다 — 그림 안 한글이 두부(□)로 나올 수 있다")
  }
  "Arial"
}

# ── 테마 ─────────────────────────────────────────────────────────────────────
theme_house <- function(base_size = 11, cjk = FALSE) {
  theme_minimal(base_size = base_size, base_family = hs_font(cjk)) +
    theme(
      text             = element_text(colour = HS_PAL$ink),
      plot.title       = element_text(face = "bold", hjust = 0, size = rel(1.0)),
      plot.subtitle    = element_text(colour = HS_PAL$gray, size = rel(0.85)),
      plot.caption     = element_text(colour = HS_PAL$gray, size = rel(0.75), hjust = 0),
      axis.title       = element_text(size = rel(0.85)),
      axis.text        = element_text(colour = HS_PAL$ink, size = rel(0.8)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = HS_PAL$edge, linewidth = 0.3, linetype = "dotted"),
      panel.border     = element_blank(),
      axis.line        = element_line(colour = HS_PAL$edge, linewidth = 0.4),
      strip.text       = element_text(face = "bold", hjust = 0, size = rel(0.9)),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.key.size  = unit(0.8, "lines"),
      plot.title.position = "plot"
    )
}

# ── 통계 헬퍼 ────────────────────────────────────────────────────────────────
hs_fmt_p <- function(p) {
  ifelse(is.na(p), "p = —", ifelse(p < 0.001, "p < 0.001", sprintf("p = %.3f", p)))
}

#' 회귀/상관 요약 — 산점도용. n, R², Spearman ρ, p.
#' @note R²는 선형 적합의 설명력일 뿐 '두 방법이 일치한다'는 뜻이 아니다.
#'   방법 비교(method comparison)에서는 Bland–Altman(편향·일치한계)을 병기할 것.
hs_stats_fit <- function(x, y) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  if (length(x) < 3) return(list(n = length(x), label = sprintf("n = %d", length(x))))
  fit <- stats::lm(y ~ x); s <- summary(fit)
  pv  <- stats::pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail = FALSE)
  rho <- suppressWarnings(stats::cor(x, y, method = "spearman"))
  list(n = length(x), r2 = s$r.squared, slope = unname(coef(fit)[2]),
       intercept = unname(coef(fit)[1]), p = unname(pv), rho = rho,
       label = sprintf("n = %d\nR² = %.3f\nSpearman ρ = %.2f\n%s",
                       length(x), s$r.squared, rho, hs_fmt_p(unname(pv))))
}

#' 군간 비교 — 2군이면 Mann–Whitney, 3군 이상이면 Kruskal–Wallis.
hs_stats_group <- function(value, group) {
  ok <- is.finite(value) & !is.na(group)
  value <- value[ok]; group <- droplevels(as.factor(group[ok]))
  if (nlevels(group) < 2 || length(value) < 4) return(list(label = "", p = NA_real_))
  if (nlevels(group) == 2) {
    tt <- suppressWarnings(stats::wilcox.test(value ~ group)); nm <- "Mann–Whitney"
  } else {
    tt <- stats::kruskal.test(value ~ group); nm <- "Kruskal–Wallis"
  }
  list(p = unname(tt$p.value), test = nm,
       label = sprintf("%s %s", nm, hs_fmt_p(unname(tt$p.value))))
}

#' 짝지은 전후 비교 — Wilcoxon signed-rank.
hs_stats_paired <- function(before, after) {
  ok <- is.finite(before) & is.finite(after)
  b <- before[ok]; a <- after[ok]
  if (length(b) < 5) return(list(label = sprintf("n = %d (부족)", length(b)), p = NA_real_))
  d <- a - b
  pv <- if (any(d != 0)) suppressWarnings(stats::wilcox.test(b, a, paired = TRUE)$p.value) else NA_real_
  list(n = length(b), delta_median = stats::median(d), p = pv,
       label = sprintf("n = %d pairs\nΔ median = %+.2f\nWilcoxon %s",
                       length(b), stats::median(d), hs_fmt_p(pv)))
}

#' 패널 모서리에 통계 라벨을 얹는다. 축 범위와 무관하게 ±Inf로 고정.
#' @param corner "tl"/"tr"/"bl"/"br"
hs_annotate <- function(label, corner = "tl", size = 3, cjk = FALSE) {
  if (is.null(label) || !nzchar(label)) return(NULL)
  pos <- switch(corner,
    tl = list(x = -Inf, y =  Inf, h = -0.08, v =  1.15),
    tr = list(x =  Inf, y =  Inf, h =  1.08, v =  1.15),
    bl = list(x = -Inf, y = -Inf, h = -0.08, v = -0.25),
    br = list(x =  Inf, y = -Inf, h =  1.08, v = -0.25))
  annotate("text", x = pos$x, y = pos$y, label = label, hjust = pos$h, vjust = pos$v,
           size = size, colour = HS_PAL$gray, family = hs_font(cjk), lineheight = 1.1)
}

# ── 그림 타입 ────────────────────────────────────────────────────────────────

#' ① boxplot + jitter — 군간 비교의 기본형.
#' x축 라벨에 군별 n을 붙이고, 검정 결과를 패널 안쪽에 넣는다.
#' @param direction_hint 제목에 붙일 방향 설명 (예: "↓ = 값이 클수록 풍부")
hs_box <- function(df, group, value, title = NULL, ylab = value,
                   direction_hint = NULL, stats = TRUE, cjk = FALSE) {
  d <- df[is.finite(df[[value]]) & !is.na(df[[group]]), c(group, value)]
  names(d) <- c("g", "v"); d$g <- droplevels(as.factor(d$g))
  st  <- if (stats) hs_stats_group(d$v, d$g) else list(label = "")
  cnt <- tapply(d$v, d$g, length)
  levels(d$g) <- sprintf("%s\nn=%d", levels(d$g), cnt[levels(d$g)])
  ttl <- if (is.null(direction_hint)) title else paste0(title, " (", direction_hint, ")")
  ggplot(d, aes(g, v, fill = g, colour = g)) +
    geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.20, linewidth = 0.6) +
    geom_jitter(width = 0.16, size = 1.1, alpha = 0.75, stroke = 0) +
    scale_fill_manual(values = hs_colors(nlevels(d$g))) +
    scale_colour_manual(values = hs_colors(nlevels(d$g))) +
    hs_annotate(st$label, cjk = cjk) +
    labs(title = ttl, x = NULL, y = ylab) +
    theme_house(cjk = cjk) +
    theme(legend.position = "none", panel.grid.major.x = element_blank())
}

#' ② 산점도 + 회귀선 + R²/p.
#' @param identity TRUE면 완전일치선(y=x)을 점선으로 함께 (방법 비교용)
hs_scatter <- function(df, x, y, colour = NULL, title = NULL,
                       identity = FALSE, stats = TRUE, cjk = FALSE) {
  d  <- df[is.finite(df[[x]]) & is.finite(df[[y]]), ]
  st <- if (stats) hs_stats_fit(d[[x]], d[[y]]) else list(label = "")
  p  <- ggplot(d, aes(.data[[x]], .data[[y]]))
  if (!is.null(colour)) p <- p + aes(colour = .data[[colour]])
  if (identity) p <- p + geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                                     colour = HS_PAL$gray, linewidth = 0.4)
  p + geom_point(size = 1.3, alpha = 0.7, stroke = 0) +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
                colour = HS_PAL$red, fill = HS_PAL$mid, linewidth = 0.6) +
    scale_colour_manual(values = hs_colors(64), na.value = HS_PAL$gray) +
    hs_annotate(st$label, cjk = cjk) +
    labs(title = title, x = x, y = y) + theme_house(cjk = cjk)
}

#' ③ 구획 heatmap — 변수 × 지표 상관.
#' @param d long-format: row_label, col_label, rho, p, section
hs_heatmap <- function(d, title = NULL, sig = 0.05, legend_name = "Spearman ρ", cjk = FALSE) {
  d$row_label <- factor(d$row_label, levels = rev(unique(d$row_label)))
  d$col_label <- factor(d$col_label, levels = unique(d$col_label))
  ggplot(d, aes(col_label, row_label, fill = rho)) +
    geom_tile(colour = "white", linewidth = 1.2) +
    geom_point(data = subset(d, !is.na(p) & p < sig), size = 1.1, colour = HS_PAL$ink) +
    scale_fill_gradient2(low = HS_PAL$blue, mid = HS_PAL$mid, high = HS_PAL$red,
                         midpoint = 0, limits = c(-1, 1), na.value = "white",
                         name = legend_name) +
    { if ("section" %in% names(d))
        facet_grid(rows = vars(section), scales = "free_y", space = "free_y", switch = "y") } +
    labs(title = title, x = NULL, y = NULL,
         caption = sprintf("● p < %.2f (uncorrected)", sig)) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line = element_blank(), axis.text.x = element_text(face = "bold"),
          strip.placement = "outside",
          strip.text.y.left = element_text(angle = 0, hjust = 0, colour = HS_PAL$gray),
          legend.title = element_text(size = rel(0.8)),
          legend.key.width = unit(2.2, "lines"))
}

#' ④ ranked effect bar — 무엇이 가장 강하게 붙는지 한 장으로.
#' @param d label, rho, p (+ 선택 sublabel)
hs_ranked <- function(d, title = NULL, xlim = c(-1, 1), cjk = FALSE) {
  d <- d[order(d$p), ]
  d$label <- factor(d$label, levels = rev(d$label))
  d$dir   <- ifelse(d$rho >= 0, "pos", "neg")
  pad <- diff(xlim) * 0.22
  ggplot(d, aes(y = label)) +
    geom_segment(aes(x = xlim[1], xend = xlim[2], yend = label),
                 colour = HS_PAL$track, linewidth = 2.6, lineend = "round") +
    geom_segment(aes(x = 0, xend = rho, yend = label, colour = dir),
                 linewidth = 2.6, lineend = "round") +
    geom_text(aes(x = xlim[2] + pad * 0.12, label = hs_fmt_p(p)), hjust = 0,
              size = 2.7, colour = HS_PAL$ink, family = "mono") +
    scale_colour_manual(values = c(pos = HS_PAL$red, neg = HS_PAL$blue), guide = "none") +
    scale_x_continuous(limits = c(xlim[1], xlim[2] + pad), expand = c(0, 0)) +
    labs(title = title, x = NULL, y = NULL) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line = element_blank(), axis.text.x = element_blank(),
          axis.ticks = element_blank())
}

#' ⑤ 종단 spaghetti — 같은 개체를 선으로 잇는 전후 비교.
#' @param time_levels 시점 순서. 주지 않으면 factor()가 **알파벳순으로 정렬**해
#'   B/M/E가 B/E/M이 되는 식의 사고가 난다 — 시점에 순서가 있으면 반드시 지정할 것.
hs_paired <- function(df, id, time, value, group = NULL, title = NULL,
                      ylab = value, time_levels = NULL, stats = TRUE, cjk = FALSE) {
  d <- df[is.finite(df[[value]]), ]
  d$.id <- d[[id]]
  d$.t  <- if (is.null(time_levels)) factor(d[[time]]) else factor(d[[time]], levels = time_levels)
  d$.v  <- d[[value]]
  lab <- ""
  if (stats && nlevels(d$.t) >= 2) {
    w  <- stats::reshape(d[, c(".id", ".t", ".v")], idvar = ".id", timevar = ".t",
                         direction = "wide")
    st <- hs_stats_paired(w[[2]], w[[ncol(w)]])
    lab <- st$label
  }
  p <- ggplot(d, aes(.t, .v))
  if (!is.null(group)) p <- p + aes(colour = .data[[group]])
  p + geom_line(aes(group = .id), colour = HS_PAL$gray, linewidth = 0.4, alpha = 0.45) +
    geom_point(size = 1.4, alpha = 0.75, stroke = 0) +
    stat_summary(fun = stats::median, geom = "crossbar", width = 0.45,
                 colour = HS_PAL$ink, linewidth = 0.4) +
    scale_colour_manual(values = hs_colors(64), na.value = HS_PAL$gray) +
    hs_annotate(lab, cjk = cjk) +
    labs(title = title, x = NULL, y = ylab) + theme_house(cjk = cjk) +
    theme(panel.grid.major.x = element_blank())
}

#' ⑥ forest plot — 점추정치 + 신뢰구간, 기준선 대비.
#' 효과크기(차이는 ref=0, 비율/OR/HR은 ref=1)와 95% CI를 행마다 그리고, 오른쪽에
#' p값을 monospace로 붙인다(house style 통계-표기 규약, hs_ranked와 동일 관례).
#' 유의성은 색만이 아니라 점 모양(채움=유의 / 빈 원=비유의)으로도 이중 인코딩한다
#' — 흑백 인쇄에서도 구분되도록.
#' @param d label, estimate, ci_low, ci_high (+ 선택 p, group, panel 컬럼)
#' @param ref 기준선 위치 (차이=0, 비율/OR/HR=1)
#' @param log_x TRUE면 x축 로그스케일 (OR/HR 등 비율 지표일 때)
#' @param order "effect"(기본, 효과크기 오름차순) | "label"(라벨 알파벳순) |
#'   "none"(입력 순서 그대로 -- 이미 정렬된 데이터를 넘길 때, 또는 facet마다
#'   같은 라벨 순서를 유지하고 싶을 때)
#' @param group 점 색상을 나눌 컬럼명 (예: 1차 대 2차 대비, 예측자군 구분).
#'   생략하면 hs_ranked와 같은 관례로 기준선 기준 양(RED)/음(BLUE) 자동 채색.
#' @param panel facet_grid(rows=)로 쪼갤 컬럼명 (다중 forest 패널일 때, 예: 여러
#'   층화 부분군을 한 그림에 쌓을 때). label 순서는 patch 전체에서 공유된다.
#' @param show_p 오른쪽에 p값을 monospace로 붙일지 (기본 TRUE; p 컬럼이 없으면
#'   자동으로 꺼진다)
hs_forest <- function(d, title = NULL, xlab = NULL, ref = 0, log_x = FALSE,
                      order = c("effect", "label", "none"), group = NULL, panel = NULL,
                      show_p = TRUE, cjk = FALSE) {
  order <- match.arg(order)
  stopifnot(all(c("label", "estimate", "ci_low", "ci_high") %in% names(d)))
  d <- as.data.frame(d)
  if (!("p" %in% names(d))) { d$p <- NA_real_; show_p <- FALSE }
  d$.sig <- ifelse(is.na(d$ci_low) | is.na(d$ci_high), NA,
                   !(d$ci_low <= ref & d$ci_high >= ref))
  d$.grp <- if (!is.null(group)) as.factor(d[[group]]) else
    factor(ifelse(d$estimate >= ref, "pos", "neg"), levels = c("pos", "neg"))

  ord <- switch(order,
    effect = order(d$estimate),
    label  = order(as.character(d$label)),
    none   = seq_len(nrow(d)))
  d <- d[ord, ]
  d$label <- factor(d$label, levels = unique(d$label))

  pal <- if (!is.null(group)) stats::setNames(hs_colors(nlevels(d$.grp)), levels(d$.grp))
         else c(pos = HS_PAL$red, neg = HS_PAL$blue)

  # p 열을 위한 오른쪽 여백 -- 데이터 범위 기준으로 동적 계산 (로그축은 곱셈 여백)
  rng_lo <- min(c(d$ci_low, d$estimate, ref), na.rm = TRUE)
  rng_hi <- max(c(d$ci_high, d$estimate, ref), na.rm = TRUE)
  if (log_x) {
    l_lo <- log10(max(rng_lo, 1e-6)); l_hi <- log10(max(rng_hi, 1e-6))
    span <- max(l_hi - l_lo, 1e-6)
    d$.p_x   <- 10^(log10(pmax(d$ci_high, ref)) + span * 0.06)
    xlim_hi  <- if (show_p) 10^(l_hi + span * 0.32) else NA
  } else {
    span <- max(rng_hi - rng_lo, 1e-6)
    d$.p_x  <- pmax(d$ci_high, ref) + span * 0.06
    xlim_hi <- if (show_p) rng_hi + span * 0.32 else NA
  }

  p_obj <- ggplot(d, aes(x = estimate, y = label)) +
    geom_vline(xintercept = ref, colour = HS_PAL$gray, linetype = "dashed", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, colour = .grp), height = 0, linewidth = 0.7) +
    geom_point(aes(colour = .grp, shape = .sig), size = 2.4, stroke = 1.0, fill = "white") +
    scale_colour_manual(values = pal, na.value = HS_PAL$gray,
                        guide = if (!is.null(group)) "legend" else "none") +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), na.value = 4, guide = "none") +
    labs(title = title, x = xlab, y = NULL, colour = NULL) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major.y = element_blank())

  if (show_p) {
    p_obj <- p_obj + geom_text(aes(x = .p_x, label = hs_fmt_p(p)), hjust = 0,
                                size = 2.7, colour = HS_PAL$ink, family = "mono")
  }
  if (log_x) {
    p_obj <- p_obj + scale_x_log10(limits = c(NA, if (is.na(xlim_hi)) NA else xlim_hi))
  } else if (!is.na(xlim_hi)) {
    p_obj <- p_obj + scale_x_continuous(limits = c(NA, xlim_hi))
  }
  if (!is.null(panel)) {
    p_obj <- p_obj + facet_grid(rows = vars(.data[[panel]]), scales = "free_y", space = "free_y")
  }
  p_obj
}

# ── 선택 패키지 편의 레이어 ──────────────────────────────────────────────────

#' 산점도에 회귀식 + R² + p를 수식으로. facet마다 자동 재계산 (ggpmisc).
hs_poly_eq <- function(size = 3) {
  if (!hs_has("ggpmisc")) { warning("ggpmisc 없음 — hs_annotate(hs_stats_fit(...)) 사용"); return(NULL) }
  ggpmisc::stat_poly_eq(
    mapping = aes(label = paste(after_stat(eq.label), after_stat(rr.label),
                                after_stat(p.value.label), sep = "~~~")),
    formula = y ~ x, parse = TRUE, size = size, colour = HS_PAL$gray,
    label.x = "left", label.y = "top")
}

#' 군간 유의성 브래킷 (ggpubr). comparisons 없으면 전체 검정 p만.
hs_signif <- function(comparisons = NULL, size = 3, method = "wilcox.test") {
  if (!hs_has("ggpubr")) { warning("ggpubr 없음 — hs_annotate(hs_stats_group(...)) 사용"); return(NULL) }
  if (is.null(comparisons)) {
    ggpubr::stat_compare_means(size = size, colour = HS_PAL$gray)
  } else {
    ggpubr::stat_compare_means(comparisons = comparisons, method = method, size = size,
                               colour = HS_PAL$ink, tip.length = 0.01, bracket.size = 0.3)
  }
}

# ── 저장 ─────────────────────────────────────────────────────────────────────
#' ragg::agg_png로 저장. 기본 png()보다 폰트 처리가 정확하고 글리프 fallback을 한다.
hs_save <- function(plot, path, width = 8, height = 5, dpi = 200) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  ragg::agg_png(path, width = width, height = height, units = "in",
                res = dpi, background = "white")
  print(plot); invisible(grDevices::dev.off())
  path
}

#' 벡터 출력(논문 제출용). svglite 또는 pdf.
#'
#' 확장자와 내용이 어긋나지 않는다. 예전 판은 svglite가 없으면 조용히 cairo_pdf로
#' 폴백하면서 **파일명은 .svg로 유지**했다 — 그래서 저자가 편집하려고 연 파일이
#' 사실은 PDF였다. 2026-09-14에 실제로 일어났고, 11개 그림이 전부 그 상태로
#' 납품됐다. 파일 안에 무엇이 들었는지는 확장자가 말해야 하므로, .svg를 요청했는데
#' svglite가 없으면 폴백하지 않고 **멈춘다**.
hs_save_vector <- function(plot, path, width = 8, height = 5) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (grepl("\\.svg$", path)) {
    if (!hs_has("svglite"))
      stop("hs_save_vector(): svglite is not installed, so '", basename(path),
           "' cannot be written as SVG. Falling back to cairo_pdf under a .svg ",
           "name would hand the author a file that does not open in an SVG ",
           "editor. Install svglite, or ask for a .pdf path.")
    svglite::svglite(path, width = width, height = height)
    on.exit({
      # 제출용 글꼴 스택. SVG 에 기록되는 이름은 Arial 을 앞에 둔다. 이 기계에는 실물
      # Arial 이 있으므로 그대로 렌더되고, 없는 기계에서는 Liberation Sans 로 떨어지는데
      # metric 호환이라 글자 폭·줄바꿈이 동일하다.
      #
      # ⚠️ **홑따옴표를 쓰지 말 것.** svglite 는 style 속성을 style='...' 로 감싼다.
      #    그 안에 'Liberation Sans' 를 넣으면 속성이 거기서 끊겨 SVG 전체가 깨진다
      #    (실측 2026-09-22 — 모든 그림의 SVG 가 not well-formed 가 됐다).
      #    홑따옴표 속성 안의 겹따옴표는 적법하므로 겹따옴표를 쓴다.
      if (file.exists(path)) {
        txt <- readLines(path, warn = FALSE)
        txt <- gsub('font-family: *"?(Liberation Sans|DejaVu Sans|Arial)"?',
                    'font-family: Arial, "Liberation Sans", Helvetica, sans-serif',
                    txt)
        writeLines(txt, path)
        # 고쳐 쓴 뒤 **반드시 파싱해 본다.** 위 치환이 SVG 를 깨뜨린 적이 있다.
        ok <- tryCatch({ xml2::read_xml(path); TRUE }, error = function(e) conditionMessage(e))
        if (!isTRUE(ok))
          stop("hs_save_vector(): 글꼴 치환 후 SVG 가 깨졌다 — ", basename(path), ": ", ok)
      }
    }, add = TRUE)
  } else {
    grDevices::cairo_pdf(path, width = width, height = height)
  }
  print(plot); invisible(grDevices::dev.off())
  path
}
