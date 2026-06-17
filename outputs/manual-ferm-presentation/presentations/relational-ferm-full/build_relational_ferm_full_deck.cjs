const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const ROOT = "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python";
const FIG = path.join(ROOT, "outputs", "presentation_figures");
const OUT_DIR = path.join(ROOT, "outputs", "presentation_deck");
const FINAL = path.join(OUT_DIR, "Relational_FERM_full_presentation.pptx");

fs.mkdirSync(OUT_DIR, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Vittoria Bettio";
pptx.subject = "Relational FERM model presentation";
pptx.title = "Relational FERM: Model, Normalization, and Empirical Evidence";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

const C = {
  bg: "FBFAF6",
  ink: "17202A",
  slate: "65727C",
  rule: "CBD0D2",
  faint: "EEF1EF",
  teal: "2F6F73",
  tealSoft: "DCEBE8",
  clay: "B65C2A",
  claySoft: "F1DDD2",
  red: "B54848",
  redSoft: "F4DADA",
  dark: "28313B",
  white: "FFFFFF",
};

const selectedNorms = [
  ["GDP", "log zscore", "8"],
  ["Abel stock", "expected-stock log-ratio min-max", "8"],
  ["Social connectedness", "min-max", "5"],
  ["Common religion", "min-max", "8"],
  ["Diplomatic disagreement", "log rank", "8"],
];

const testRows = [
  ["Traditional FERM: GDP", "0.037", "0.568", "0.689", "0.152"],
  ["Relational FERM: SCI", "0.026", "0.562", "0.658", "0.157"],
  ["Relational FERM: Common religion", "0.002", "0.502", "0.654", "0.197"],
  ["Abel stock", "-0.003", "0.446", "0.643", "0.207"],
  ["Relational FERM: Diplo disagreement", "-0.008", "0.413", "0.626", "0.228"],
  ["RM", "baseline", "baseline", "0.646", "0.000"],
];

function bg(slide) {
  slide.background = { color: C.bg };
}

function footer(slide, n, source = "Sources: FERM model note; src/ferm/model.py; presentation_ferm_results.ipynb.") {
  slide.addShape(pptx.ShapeType.line, { x: 0.6, y: 7.02, w: 12.1, h: 0, line: { color: C.rule, width: 0.6 } });
  slide.addText(source, { x: 0.63, y: 7.08, w: 9.8, h: 0.18, fontFace: "Aptos", fontSize: 7.5, color: C.slate, margin: 0 });
  slide.addText(String(n).padStart(2, "0"), { x: 12.25, y: 7.06, w: 0.45, h: 0.2, fontFace: "Aptos", fontSize: 8.5, bold: true, color: C.slate, align: "right", margin: 0 });
}

function kicker(slide, text, x = 0.72, y = 0.42, color = C.teal) {
  slide.addShape(pptx.ShapeType.rect, { x, y: y + 0.045, w: 0.15, h: 0.15, fill: { color }, line: { color } });
  slide.addText(text.toUpperCase(), { x: x + 0.26, y, w: 5.5, h: 0.24, fontFace: "Aptos", fontSize: 8.4, bold: true, color, charSpace: 1.15, margin: 0 });
}

function title(slide, text, subtitle = "", opts = {}) {
  const y = opts.y ?? 0.78;
  slide.addText(text, {
    x: opts.x ?? 0.72,
    y,
    w: opts.w ?? 11.3,
    h: opts.h ?? 0.85,
    fontFace: "Aptos Display",
    fontSize: opts.size ?? 30,
    bold: true,
    color: C.ink,
    fit: "shrink",
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: opts.x ?? 0.74,
      y: y + (opts.subY ?? 0.92),
      w: opts.subW ?? 10.8,
      h: 0.5,
      fontFace: "Aptos",
      fontSize: opts.subSize ?? 13.3,
      color: C.slate,
      fit: "shrink",
      margin: 0,
    });
  }
}

function callout(slide, head, body, x, y, w, h, color = C.teal, fill = C.tealSoft) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: fill, transparency: 5 }, line: { color: fill } });
  slide.addText(head, { x: x + 0.16, y: y + 0.14, w: w - 0.32, h: 0.25, fontFace: "Aptos", fontSize: 10.2, bold: true, color, margin: 0 });
  slide.addText(body, { x: x + 0.16, y: y + 0.46, w: w - 0.32, h: h - 0.58, fontFace: "Aptos", fontSize: 11.8, color: C.ink, fit: "shrink", margin: 0 });
}

function bulletList(slide, items, x, y, w, h, size = 14) {
  slide.addText(items.map(t => ({ text: t, options: { bullet: { type: "ul" }, hanging: 3, breakLine: true } })), {
    x, y, w, h, fontFace: "Aptos", fontSize: size, color: C.ink, fit: "shrink", margin: 0.04, paraSpaceAfterPt: 7,
  });
}

function metric(slide, value, label, x, y, w, color) {
  slide.addText(value, { x, y, w, h: 0.48, fontFace: "Aptos Display", fontSize: 25, bold: true, color, align: "center", margin: 0 });
  slide.addText(label, { x, y: y + 0.52, w, h: 0.36, fontFace: "Aptos", fontSize: 8.6, bold: true, color: C.slate, align: "center", margin: 0, fit: "shrink" });
}

async function imageBox(slide, file, x, y, w, h, border = true) {
  const full = path.join(FIG, file);
  const meta = await sharp(full).metadata();
  const ratio = meta.width / meta.height;
  const box = w / h;
  let iw = w, ih = h, ix = x, iy = y;
  if (ratio > box) {
    ih = w / ratio;
    iy = y + (h - ih) / 2;
  } else {
    iw = h * ratio;
    ix = x + (w - iw) / 2;
  }
  if (border) slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: C.white }, line: { color: C.rule, width: 0.5 } });
  slide.addImage({ path: full, x: ix, y: iy, w: iw, h: ih });
}

function pill(slide, text, x, y, w, color, fill) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.34, fill: { color: fill }, line: { color: fill }, radius: 0.05 });
  slide.addText(text, { x: x + 0.07, y: y + 0.075, w: w - 0.14, h: 0.12, fontFace: "Aptos", fontSize: 8.5, bold: true, color, align: "center", margin: 0, fit: "shrink" });
}

function arrow(slide, x1, y1, x2, y2, color = C.rule) {
  slide.addShape(pptx.ShapeType.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color, width: 1.2, beginArrowType: "none", endArrowType: "triangle" } });
}

function simpleBox(slide, text, x, y, w, h, color, fill, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: fill, transparency: opts.transparency ?? 0 }, line: { color: opts.line ?? fill, width: opts.lineWidth ?? 0.7 } });
  slide.addText(text, { x: x + 0.12, y: y + 0.12, w: w - 0.24, h: h - 0.24, fontFace: "Aptos", fontSize: opts.size ?? 13, bold: opts.bold ?? false, color, fit: "shrink", valign: "mid", align: opts.align ?? "center", margin: 0 });
}

function modelFamilyTable(slide) {
  const rows = [
    [{ text: "Variant", options: { bold: true, color: C.white } }, { text: "Sigma structure", options: { bold: true, color: C.white } }, { text: "Interpretation", options: { bold: true, color: C.white } }],
    ["RM", "Sigma_ii = 0, Sigma_ij = 0", "No feature-based attractiveness."],
    ["Traditional FERM", "Sigma_ii = mu_i, Sigma_ij = mu_j", "Destination-specific attractiveness; rows are identical."],
    ["Relational FERM", "Sigma_ii = 0, Sigma_ij = delta_ij", "Thresholds neutral; offers become corridor-specific."],
    ["Combined model", "Sigma_ii = mu_i, Sigma_ij = f(mu_j, delta_ij)", "Destination and corridor information together."],
  ];
  slide.addTable(rows, {
    x: 0.78, y: 1.78, w: 11.85, h: 3.35,
    border: { type: "solid", color: C.rule, pt: 0.6 },
    margin: 0.07,
    fontFace: "Aptos",
    fontSize: 11.5,
    color: C.ink,
    valign: "mid",
    colW: [2.15, 3.75, 5.9],
    rowH: [0.38, 0.48, 0.58, 0.58, 0.58],
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.rect, { x: 0.78, y: 1.78, w: 11.85, h: 0.38, fill: { color: C.ink }, line: { color: C.ink } });
}

function selectedNormTable(slide, x, y, w, h) {
  const rows = [
    [{ text: "Feature", options: { bold: true, color: C.white } }, { text: "Selected normalization", options: { bold: true, color: C.white } }, { text: "sigma", options: { bold: true, color: C.white, align: "center" } }],
    ...selectedNorms,
  ];
  slide.addTable(rows, {
    x, y, w, h,
    border: { type: "solid", color: C.rule, pt: 0.6 },
    margin: 0.055,
    fontFace: "Aptos",
    fontSize: 9.7,
    color: C.ink,
    valign: "mid",
    colW: [1.75, 2.85, 0.65],
    rowH: [0.34, 0.36, 0.36, 0.36, 0.36, 0.36],
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.34, fill: { color: C.ink }, line: { color: C.ink } });
}

function testTable(slide) {
  const rows = [
    [{ text: "Model", options: { bold: true, color: C.white } }, { text: "Median improvement", options: { bold: true, color: C.white } }, { text: "Share better", options: { bold: true, color: C.white } }, { text: "Pearson log", options: { bold: true, color: C.white } }],
    ...testRows.map(r => [r[0], r[1], r[2], r[3]]),
  ];
  slide.addTable(rows, {
    x: 0.78, y: 1.7, w: 8.8, h: 3.75,
    border: { type: "solid", color: C.rule, pt: 0.6 },
    margin: 0.06,
    fontFace: "Aptos",
    fontSize: 10.3,
    color: C.ink,
    valign: "mid",
    colW: [4.0, 1.65, 1.45, 1.3],
    rowH: [0.37, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.rect, { x: 0.78, y: 1.7, w: 8.8, h: 0.37, fill: { color: C.ink }, line: { color: C.ink } });
}

async function build() {
  let s, n = 1;

  s = pptx.addSlide(); bg(s);
  kicker(s, "RELATIONAL FERM", 0.75, 0.52, C.teal);
  s.addText("Relational FERM keeps the radiation mechanism and changes the attractiveness matrix.", {
    x: 0.75, y: 1.08, w: 10.95, h: 1.4, fontFace: "Aptos Display", fontSize: 33, bold: true, color: C.ink, fit: "shrink", margin: 0,
  });
  s.addText("Model formulation, Sigma construction, normalization, and out-of-sample evidence", { x: 0.78, y: 2.58, w: 9.6, h: 0.35, fontFace: "Aptos", fontSize: 14, color: C.slate, margin: 0 });
  metric(s, "Sigma_ij", "corridor attractiveness", 0.8, 4.05, 2.4, C.teal);
  metric(s, "0.037", "GDP median improvement", 3.45, 4.05, 2.4, C.clay);
  metric(s, "0.562", "SCI share better", 6.1, 4.05, 2.4, C.teal);
  metric(s, "24.7%", "routes won by SCI", 8.75, 4.05, 2.4, C.teal);
  callout(s, "Core claim", "The relational extension is not a different mobility process. It is a different parameterization of attractiveness within the same radiation-style absorption framework.", 0.78, 5.4, 11.65, 0.95, C.teal, C.tealSoft);
  footer(s, n++, "Sources: FERM model note and presentation_ferm_results.ipynb.");

  s = pptx.addSlide(); bg(s);
  kicker(s, "MOTIVATION", 0.72, 0.42, C.clay);
  title(s, "Migration attractiveness is not only destination-specific.", "The same destination can be more or less attractive depending on the origin corridor.");
  simpleBox(s, "Destination-level attractiveness\nmu_j", 0.9, 2.15, 2.75, 1.25, C.clay, C.claySoft, { bold: true });
  simpleBox(s, "Destination j looks the same\nfrom every origin", 0.9, 3.75, 2.75, 1.15, C.ink, C.white, { line: C.rule });
  simpleBox(s, "Corridor-specific attractiveness\ndelta_ij", 5.0, 2.15, 2.95, 1.25, C.teal, C.tealSoft, { bold: true });
  simpleBox(s, "Destination j can look different\nfrom origin i versus origin k", 5.0, 3.75, 2.95, 1.15, C.ink, C.white, { line: C.rule });
  arrow(s, 3.9, 2.76, 4.7, 2.76, C.rule);
  callout(s, "Examples of bilateral signals", "social connectedness, common language or religion, visa openness, diaspora networks, diplomatic relations, historical ties", 8.65, 2.0, 3.55, 2.55, C.teal, C.tealSoft);
  callout(s, "Research question", "Can corridor-specific attractiveness improve route-level predictions without changing the radiation mechanism?", 8.65, 4.85, 3.55, 1.1, C.clay, C.claySoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "MECHANISM", 0.72, 0.42, C.teal);
  title(s, "The radiation-style mechanism is sequential absorption by distance.", "Particles leave an origin, inspect destinations in increasing distance order, and stop at the first successful offer.");
  simpleBox(s, "Origin i\nparticle emitted", 0.9, 2.45, 1.75, 0.82, C.ink, C.white, { line: C.rule, bold: true });
  simpleBox(s, "Threshold T_i", 0.9, 3.55, 1.75, 0.65, C.teal, C.tealSoft, { bold: true });
  const xs = [4.0, 6.05, 8.1, 10.15];
  const labels = ["nearest k1", "k2", "destination j", "farther k3"];
  for (let i = 0; i < xs.length; i++) {
    simpleBox(s, `${labels[i]}\noffer B_i${i+1}`, xs[i], 2.45, 1.45, 0.92, i === 2 ? C.clay : C.ink, i === 2 ? C.claySoft : C.white, { line: i === 2 ? C.claySoft : C.rule, bold: i === 2 });
    if (i > 0) arrow(s, xs[i] - 0.55, 2.9, xs[i] - 0.12, 2.9, C.rule);
  }
  arrow(s, 2.78, 2.9, 3.65, 2.9, C.rule);
  s.addText("P(i -> j) = p_ij × product over closer k of (1 - p_ik)", { x: 3.95, y: 4.35, w: 6.4, h: 0.35, fontFace: "Aptos", fontSize: 17, bold: true, color: C.ink, align: "center", margin: 0 });
  callout(s, "Important", "Traditional FERM and relational FERM do not change this scanning process. They change the centers of the latent threshold and offer distributions.", 1.0, 5.35, 11.4, 0.8, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "LATENT VARIABLES", 0.72, 0.42, C.clay);
  title(s, "Thresholds and offers are latent maxima centered by Sigma.", "Population enters through the maximum over many Gaussian draws; attractiveness shifts the mean of those draws.");
  simpleBox(s, "Threshold side\nT_i ~ max over m_i draws\nN(Sigma_ii, sigma^2)", 1.0, 2.0, 4.65, 1.45, C.teal, C.tealSoft, { bold: true, size: 15 });
  simpleBox(s, "Offer side\nB_ij ~ max over n_j draws\nN(Sigma_ij, sigma^2)", 7.0, 2.0, 4.65, 1.45, C.clay, C.claySoft, { bold: true, size: 15 });
  arrow(s, 5.85, 2.72, 6.75, 2.72, C.rule);
  callout(s, "Population", "m_i and n_j affect the distributions through maxima: larger populations generate more extreme thresholds/offers.", 1.0, 4.2, 5.2, 1.1, C.teal, C.tealSoft);
  callout(s, "Shared sigma", "The model uses one sigma as the shared spread of the Gaussian sampling kernel. Larger sigma dilutes the effect of Sigma centers.", 6.85, 4.2, 5.2, 1.1, C.clay, C.claySoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "SIGMA MATRIX", 0.72, 0.42, C.teal);
  title(s, "One attractiveness matrix unifies RM, traditional FERM, and relational FERM.", "The diagonal controls origin thresholds; off-diagonal entries control origin-destination offers.");
  const x0 = 1.1, y0 = 1.95, cell = 0.72;
  const countries = ["A", "B", "C", "D", "E"];
  for (let i = 0; i < 5; i++) {
    s.addText(countries[i], { x: x0 - 0.42, y: y0 + i * cell + 0.2, w: 0.25, h: 0.15, fontSize: 9, bold: true, color: C.slate, margin: 0, align: "center" });
    s.addText(countries[i], { x: x0 + i * cell + 0.24, y: y0 - 0.32, w: 0.25, h: 0.15, fontSize: 9, bold: true, color: C.slate, margin: 0, align: "center" });
    for (let j = 0; j < 5; j++) {
      const diag = i === j;
      s.addShape(pptx.ShapeType.rect, { x: x0 + j * cell, y: y0 + i * cell, w: cell - 0.03, h: cell - 0.03, fill: { color: diag ? C.tealSoft : C.claySoft }, line: { color: C.white } });
      s.addText(diag ? "Σii" : "Σij", { x: x0 + j * cell, y: y0 + i * cell + 0.24, w: cell - 0.03, h: 0.12, fontSize: 8.2, bold: true, color: diag ? C.teal : C.clay, align: "center", margin: 0 });
    }
  }
  callout(s, "Diagonal", "Sigma_ii centers the threshold distribution at origin i.", 5.45, 2.0, 3.05, 1.15, C.teal, C.tealSoft);
  callout(s, "Off-diagonal", "Sigma_ij centers the offer distribution for corridor i -> j.", 8.85, 2.0, 3.05, 1.15, C.clay, C.claySoft);
  callout(s, "Relational view", "A destination is not intrinsically attractive in the same way from every origin. The row matters.", 5.45, 4.05, 6.5, 1.05, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "MODEL FAMILY", 0.72, 0.42, C.clay);
  title(s, "The variants differ only by the structure imposed on Sigma.", "This is the conceptual clarity of the matrix formulation.");
  modelFamilyTable(s);
  callout(s, "Takeaway", "Relational FERM is the corridor-specific case: Sigma_ii is kept neutral and Sigma_ij encodes bilateral attraction or repulsion.", 1.0, 5.65, 11.3, 0.8, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "RELATIONAL FERM", 0.72, 0.42, C.teal);
  title(s, "In the relational specification, all bilateral information lives on the offer side.", "Thresholds are neutral; corridors shift destination offers.");
  simpleBox(s, "Origin threshold\nSigma_ii = 0", 0.95, 2.45, 2.3, 0.9, C.teal, C.tealSoft, { bold: true });
  simpleBox(s, "Negative corridor\nSigma_ij < 0", 4.2, 1.85, 2.25, 0.9, C.red, C.redSoft, { bold: true });
  simpleBox(s, "Neutral corridor\nSigma_ij = 0", 4.2, 3.0, 2.25, 0.9, C.ink, C.faint, { bold: true, line: C.rule });
  simpleBox(s, "Positive corridor\nSigma_ij > 0", 4.2, 4.15, 2.25, 0.9, C.clay, C.claySoft, { bold: true });
  arrow(s, 3.35, 2.9, 4.0, 2.28, C.rule);
  arrow(s, 3.35, 2.9, 4.0, 3.45, C.rule);
  arrow(s, 3.35, 2.9, 4.0, 4.6, C.rule);
  callout(s, "Interpretation", "A positive corridor shifts the offer distribution upward and increases the chance of absorption before farther destinations are reached.", 7.25, 1.75, 4.65, 1.3, C.clay, C.claySoft);
  callout(s, "Mechanism unchanged", "The particle still scans destinations in distance order. Relational FERM only changes the mean of B_ij.", 7.25, 3.55, 4.65, 1.15, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "INTERPRETATION", 0.72, 0.42, C.clay);
  title(s, "Sigma controls direction; sigma controls how sharply the feature matters.", "This distinction is crucial for interpreting validation curves.");
  simpleBox(s, "Sigma_ij > 0\nmore attractive than neutral", 0.9, 1.95, 3.2, 1.0, C.clay, C.claySoft, { bold: true });
  simpleBox(s, "Sigma_ij = 0\nneutral corridor", 0.9, 3.25, 3.2, 0.9, C.ink, C.faint, { bold: true, line: C.rule });
  simpleBox(s, "Sigma_ij < 0\nmore repulsive than neutral", 0.9, 4.5, 3.2, 1.0, C.red, C.redSoft, { bold: true });
  callout(s, "Low sigma", "Centers matter strongly; Sigma has a stronger effect on absorption.", 5.15, 2.0, 3.1, 1.05, C.teal, C.tealSoft);
  callout(s, "High sigma", "Gaussian spread dominates center differences; the model moves closer to RM-like behavior.", 8.65, 2.0, 3.1, 1.05, C.clay, C.claySoft);
  s.addShape(pptx.ShapeType.line, { x: 5.25, y: 4.45, w: 5.9, h: 0, line: { color: C.rule, width: 1.4 } });
  s.addText("feature influence", { x: 5.05, y: 4.75, w: 1.45, h: 0.2, fontSize: 9, color: C.slate, margin: 0, align: "center" });
  s.addText("RM-like dilution", { x: 9.95, y: 4.75, w: 1.7, h: 0.2, fontSize: 9, color: C.slate, margin: 0, align: "center" });
  s.addText("small sigma", { x: 4.9, y: 4.12, w: 1.4, h: 0.18, fontSize: 10, bold: true, color: C.teal, align: "center", margin: 0 });
  s.addText("large sigma", { x: 10.15, y: 4.12, w: 1.4, h: 0.18, fontSize: 10, bold: true, color: C.clay, align: "center", margin: 0 });
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "EMPIRICAL SIGMA", 0.72, 0.42, C.teal);
  title(s, "The empirical task is to build a meaningful Sigma matrix.", "The model is only as good as the feature and normalization used to construct Sigma.");
  const features = [
    ["GDP", "destination-level", "traditional FERM", C.clay, C.claySoft],
    ["Abel stock", "historical corridor stock", "relational FERM", C.teal, C.tealSoft],
    ["Social connectedness", "bilateral social tie", "relational FERM", C.teal, C.tealSoft],
    ["Common religion", "bilateral similarity", "relational FERM", C.teal, C.tealSoft],
    ["Diplo disagreement", "bilateral friction", "relational FERM", C.red, C.redSoft],
  ];
  for (let i = 0; i < features.length; i++) {
    const y = 1.75 + i * 0.72;
    simpleBox(s, features[i][0], 0.95, y, 2.25, 0.46, features[i][3], features[i][4], { bold: true, size: 11 });
    s.addText(features[i][1], { x: 3.55, y: y + 0.12, w: 3.3, h: 0.14, fontSize: 10.5, color: C.ink, margin: 0 });
    pill(s, features[i][2], 7.4, y + 0.04, 2.4, features[i][3], features[i][4]);
  }
  callout(s, "Distance", "Distance is not a feature in this comparison. It is already used to order destinations in the radiation mechanism.", 9.95, 2.05, 2.6, 1.6, C.clay, C.claySoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "NORMALIZATION", 0.72, 0.42, C.clay);
  title(s, "Bilateral features cannot be inserted raw into Sigma.", "Zero should mean neutral; signs should mean attraction versus repulsion; heavy tails need compression.");
  const normItems = [
    "sign orientation: high values can mean attraction or repulsion",
    "centering: Sigma > 0 attractive, Sigma = 0 neutral, Sigma < 0 repulsive",
    "heavy tails: log compression can stop extreme corridors dominating",
    "missing values: zero is a substantive neutrality assumption",
    "split comparability: validation and test must use consistent transformations",
  ];
  bulletList(s, normItems, 0.9, 1.85, 6.0, 3.65, 14.2);
  selectedNormTable(s, 7.25, 1.85, 5.25, 2.55);
  callout(s, "Validation rule", "The notebook tests normalizations within each feature group and carries forward one selected version per group into the test split.", 7.25, 4.9, 5.25, 0.9, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "PIPELINE", 0.72, 0.42, C.teal);
  title(s, "The empirical workflow separates tuning from reporting.", "The final comparison is out-of-sample on test_2019_h2.");
  const steps = [
    ["Build nodes and flows", "Asia, 2019 validation/test periods"],
    ["Build distance matrix", "gravity dist used for scanning order"],
    ["Construct Sigma variants", "GDP, Abel, SCI, religion, diplomacy"],
    ["Tune sigma on validation", "same grid, same metric hierarchy"],
    ["Freeze and test", "one selected model per feature group"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const x = 0.85 + i * 2.43;
    simpleBox(s, String(i + 1), x, 2.0, 0.42, 0.42, C.white, i < 3 ? C.teal : C.clay, { bold: true });
    simpleBox(s, steps[i][0], x, 2.65, 1.92, 0.7, C.ink, C.white, { line: C.rule, bold: true, size: 11.5 });
    s.addText(steps[i][1], { x, y: 3.5, w: 1.95, h: 0.55, fontFace: "Aptos", fontSize: 9.5, color: C.slate, fit: "shrink", align: "center", margin: 0 });
    if (i < steps.length - 1) arrow(s, x + 1.95, 3.0, x + 2.28, 3.0, C.rule);
  }
  callout(s, "Metric hierarchy", "Primary: median route-level improvement over RM. Tie-breakers: share of routes better than RM, then Pearson correlation in log space.", 1.0, 5.15, 11.3, 0.85, C.teal, C.tealSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "NORMALIZATION RESULTS", 0.72, 0.42, C.clay);
  title(s, "Normalization changes results, so the selected specification is chosen feature-by-feature.", "This is a robustness step, not an extra model parameter.");
  await imageBox(s, "02a_normalization_sensitivity_all_features.png", 0.85, 1.52, 11.65, 5.25, true);
  footer(s, n++, "Validation split: 2019 H1. Each row is one feature group; each line is one normalization.");

  s = pptx.addSlide(); bg(s);
  kicker(s, "TEST RESULTS", 0.72, 0.42, C.clay);
  title(s, "GDP is strongest globally; SCI is the strongest relational feature.", "The relational extension shows value, but only when the bilateral feature contains useful signal.");
  testTable(s);
  callout(s, "Main result", "GDP leads on median improvement, share better than RM, and Pearson log correlation.", 10.05, 1.88, 2.35, 1.15, C.clay, C.claySoft);
  callout(s, "Relational result", "SCI is meaningfully positive: median improvement 0.026 and share better 0.562.", 10.05, 3.55, 2.35, 1.18, C.teal, C.tealSoft);
  callout(s, "Weak features", "Abel and diplomatic disagreement underperform RM on median route-level error.", 10.05, 5.2, 2.35, 0.95, C.red, C.redSoft);
  footer(s, n++, "Test split: 2019 H2.");

  s = pptx.addSlide(); bg(s);
  kicker(s, "ROUTE EVIDENCE", 0.72, 0.42, C.teal);
  title(s, "Route-level plots show GDP and SCI improve many OD pairs, but GDP is more stable.", "Below the diagonal means the model has lower absolute log-ratio error than RM.");
  await imageBox(s, "05_route_error_scatter.png", 0.72, 1.48, 11.95, 5.25, true);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "HETEROGENEITY", 0.72, 0.42, C.teal);
  title(s, "Relational information helps specific routes, not the whole system uniformly.", "SCI wins almost a quarter of all OD pairs, even though GDP is the best global model.");
  await imageBox(s, "08_route_winners.png", 0.88, 1.75, 7.1, 3.65, true);
  callout(s, "Route winners", "GDP wins 31.7% of OD pairs. SCI wins 24.7%. RM still wins 11.4%.", 8.45, 1.85, 3.65, 1.25, C.teal, C.tealSoft);
  callout(s, "Interpretation", "This supports a feature-dependent relational extension rather than a universal relational improvement.", 8.45, 3.55, 3.65, 1.25, C.clay, C.claySoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "ABEL DIAGNOSTIC", 0.72, 0.42, C.red);
  title(s, "Abel stock is unstable here because it zero-predicts too many real corridors.", "The problem is not only tiny routes; some large South Asia-Gulf corridors receive zero predictions.");
  metric(s, "20.7%", "zero-prediction share", 0.9, 2.0, 2.5, C.red);
  metric(s, "57,919", "observed migrants on zero routes", 3.65, 2.0, 2.9, C.red);
  metric(s, "7,424", "largest observed zero route", 6.85, 2.0, 2.8, C.red);
  const zeroRows = [
    [{ text: "Zero-predicted route", options: { bold: true, color: C.white } }, { text: "Observed", options: { bold: true, color: C.white, align: "center" } }],
    ["Sri Lanka -> Qatar", "7,424"],
    ["Qatar -> Sri Lanka", "6,239"],
    ["Bahrain -> Philippines", "6,137"],
    ["Sri Lanka -> Kuwait", "4,500"],
    ["Nepal -> Cyprus", "2,815"],
  ];
  s.addTable(zeroRows, { x: 1.1, y: 3.5, w: 5.6, h: 2.25, colW: [3.9, 1.3], rowH: [0.36, 0.34, 0.34, 0.34, 0.34, 0.34], border: { type: "solid", color: C.rule, pt: 0.6 }, margin: 0.06, fontFace: "Aptos", fontSize: 11, color: C.ink, fit: "shrink" });
  s.addShape(pptx.ShapeType.rect, { x: 1.1, y: 3.5, w: 5.6, h: 0.36, fill: { color: C.ink }, line: { color: C.ink } });
  callout(s, "Takeaway", "Abel remains conceptually relevant, but this construction is empirically too punitive or too sparse for important labor-migration corridors.", 7.55, 3.65, 4.55, 1.35, C.red, C.redSoft);
  footer(s, n++);

  s = pptx.addSlide(); bg(s);
  kicker(s, "CONCLUSION", 0.72, 0.42, C.teal);
  title(s, "Relational FERM is viable, but the empirical Sigma must be built carefully.", "The model contribution is conceptual clarity; the empirical contribution is identifying which bilateral features actually carry signal.");
  bulletList(s, [
    "The radiation mechanism is unchanged across RM, traditional FERM, and relational FERM.",
    "Relational FERM moves attractiveness from destination-only to corridor-specific offers.",
    "Normalization is part of the model specification because Sigma has sign and scale semantics.",
    "Out of sample, GDP is strongest globally; SCI is the most promising relational feature.",
    "The next natural step is a combined destination-plus-relational Sigma.",
  ], 0.92, 1.95, 7.1, 3.65, 16);
  callout(s, "Suggested final sentence", "The relational extension does not fail; it tells us that route-specific attractiveness matters only when the bilateral feature is strong enough and normalized coherently.", 8.45, 2.05, 3.75, 1.8, C.teal, C.tealSoft);
  callout(s, "Next model", "Sigma_ii = mu_i and Sigma_ij combines destination attractiveness with corridor-specific relational structure.", 8.45, 4.55, 3.75, 1.15, C.clay, C.claySoft);
  footer(s, n++);

  await pptx.writeFile({ fileName: FINAL });
  console.log(FINAL);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
