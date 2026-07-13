const BG = "#F7F3EC";
const INK = "#102132";
const MUTED = "#5B6770";
const BLUE = "#2B7BB9";
const TEAL = "#2FA79B";
const GOLD = "#D9A441";
const RED = "#C84646";
const GREEN = "#4E9A51";
const RULE = "#C8BFAF";
const WHITE = "#FFFFFF";

export const paths = {
  toySetup: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/toy_model_ferm/01_synthetic_network_and_true_sigma.png",
  toyCalibration: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/toy_model_ferm/02_calibration_curves.png",
  toyTest: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/toy_model_ferm/03_test_summary.png",
  toyRobustness: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/toy_model_ferm/05_robustness.png",
  empiricalCalibration: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/presentation_figures/02_calibration_sigma_tuning.png",
  empiricalRank: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/clean_figures/presentation_all_features/02_test_ranking_three_metrics.png",
  empiricalScatter: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/clean_figures/presentation_all_features/04_route_error_scatter_top_models.png",
  normalization: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/presentation_figures/02a_normalization_sensitivity_all_features.png",
  sciDistribution: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/combined_gdp_sci_figures/01a_sci_raw_vs_minmax_distribution.png",
  mdl: "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python/outputs/combined_gdp_sci_figures/11_description_length_mdl_vs_rm_strict_sigma.png",
};

export const C = { BG, INK, MUTED, BLUE, TEAL, GOLD, RED, GREEN, RULE, WHITE };

export function canvas(slide, ctx, page) {
  ctx.addShape(slide, { x: 0, y: 0, width: ctx.W, height: ctx.H, fill: BG });
  ctx.addShape(slide, { x: 0, y: 0, width: ctx.W, height: 13, fill: INK });
  ctx.addText(slide, {
    text: String(page).padStart(2, "0"),
    x: 1198, y: 668, width: 42, height: 20,
    fontSize: 12, color: MUTED, align: "right",
  });
}

export function kicker(slide, ctx, text, color = BLUE) {
  ctx.addShape(slide, { x: 58, y: 44, width: 22, height: 4, fill: color });
  ctx.addText(slide, {
    text,
    x: 90, y: 34, width: 360, height: 24,
    fontSize: 13, bold: true, color: color,
    typeface: ctx.fonts.body,
  });
}

export function title(slide, ctx, text, y = 72, size = 38, width = 1000) {
  ctx.addText(slide, {
    text,
    x: 58, y, width, height: 96,
    fontSize: size, bold: true, color: INK,
    typeface: ctx.fonts.title,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function subtitle(slide, ctx, text, x = 60, y = 172, width = 850) {
  ctx.addText(slide, {
    text,
    x, y, width, height: 78,
    fontSize: 20, color: MUTED,
    typeface: ctx.fonts.body,
  });
}

export function footer(slide, ctx, text = "FERM project") {
  ctx.addText(slide, {
    text,
    x: 58, y: 668, width: 520, height: 18,
    fontSize: 10, color: MUTED,
  });
}

export function note(slide, ctx, text, x, y, width, color = INK) {
  ctx.addText(slide, {
    text,
    x, y, width, height: 52,
    fontSize: 18, bold: true, color,
    typeface: ctx.fonts.body,
  });
}

export function body(slide, ctx, text, x, y, width, height = 160, size = 19, color = INK) {
  ctx.addText(slide, {
    text,
    x, y, width, height,
    fontSize: size, color,
    typeface: ctx.fonts.body,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function pill(slide, ctx, text, x, y, width, color) {
  ctx.addShape(slide, {
    x, y, width, height: 34,
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
  ctx.addText(slide, {
    text,
    x: x + 12, y: y + 7, width: width - 24, height: 18,
    fontSize: 13, bold: true, color: WHITE, align: "center",
  });
}

export function railItem(slide, ctx, n, label, text, x, y, color = BLUE) {
  ctx.addShape(slide, { x, y: y + 4, width: 30, height: 30, fill: color });
  ctx.addText(slide, {
    text: n,
    x, y: y + 10, width: 30, height: 16,
    fontSize: 13, bold: true, color: WHITE, align: "center",
  });
  ctx.addText(slide, {
    text: label,
    x: x + 44, y, width: 350, height: 24,
    fontSize: 20, bold: true, color: INK,
  });
  ctx.addText(slide, {
    text,
    x: x + 44, y: y + 30, width: 420, height: 66,
    fontSize: 16, color: MUTED,
  });
}

export function callout(slide, ctx, headline, text, x, y, width, height, color = BLUE) {
  ctx.addShape(slide, {
    x, y, width, height,
    fill: WHITE,
    line: { style: "solid", fill: RULE, width: 1 },
  });
  ctx.addShape(slide, { x, y, width: 7, height, fill: color });
  ctx.addText(slide, {
    text: headline,
    x: x + 22, y: y + 18, width: width - 42, height: 28,
    fontSize: 19, bold: true, color: INK,
  });
  ctx.addText(slide, {
    text,
    x: x + 22, y: y + 52, width: width - 42, height: height - 66,
    fontSize: 16, color: MUTED,
  });
}

export function equation(slide, ctx, text, x, y, width, color = INK) {
  ctx.addShape(slide, { x, y, width, height: 54, fill: "#EDE7DC" });
  ctx.addText(slide, {
    text,
    x: x + 16, y: y + 14, width: width - 32, height: 24,
    fontSize: 20, bold: true, color,
    typeface: ctx.fonts.mono,
    align: "center",
  });
}

export async function image(slide, ctx, path, x, y, width, height, fit = "contain") {
  ctx.addShape(slide, {
    x, y, width, height,
    fill: WHITE,
    line: { style: "solid", fill: RULE, width: 1 },
  });
  return ctx.addImage(slide, {
    path,
    x: x + 8, y: y + 8, width: width - 16, height: height - 16,
    fit,
  });
}
