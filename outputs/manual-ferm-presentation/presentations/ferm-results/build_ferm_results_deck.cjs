const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const ROOT = "/Users/vittoriabettio/Desktop/PhD/Research/FERM/Github/FERM_python";
const FIG = path.join(ROOT, "outputs", "presentation_figures");
const OUT_DIR = path.join(ROOT, "outputs", "presentation_deck");
const FINAL = path.join(OUT_DIR, "FERM_results_presentation.pptx");

fs.mkdirSync(OUT_DIR, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Vittoria Bettio";
pptx.company = "FERM";
pptx.subject = "FERM results presentation";
pptx.title = "FERM Results: Feature Normalization and Relational Evidence";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

const W = 13.333;
const H = 7.5;
const C = {
  bg: "FBFAF6",
  ink: "18202A",
  slate: "64717D",
  rule: "CBD0D2",
  teal: "2F6F73",
  tealSoft: "DCEBE8",
  clay: "B65C2A",
  claySoft: "F1DDD2",
  green: "4A7C59",
  red: "B54848",
  white: "FFFFFF",
};

const metrics = [
  { model: "Traditional FERM: GDP", med: 0.037, share: 0.568, pearson: 0.689, zero: 0.152, color: C.clay },
  { model: "Relational FERM: Social connectedness", med: 0.026, share: 0.562, pearson: 0.658, zero: 0.157, color: C.teal },
  { model: "Relational FERM: Common religion", med: 0.002, share: 0.502, pearson: 0.654, zero: 0.197, color: C.green },
  { model: "Abel stock", med: -0.003, share: 0.446, pearson: 0.643, zero: 0.207, color: C.slate },
  { model: "Relational FERM: Diplomatic disagreement", med: -0.008, share: 0.413, pearson: 0.626, zero: 0.228, color: C.red },
  { model: "RM", med: null, share: null, pearson: 0.646, zero: 0.000, color: "111111" },
];

const selectedNorms = [
  ["GDP", "log zscore", "8"],
  ["Abel stock", "expected-stock log-ratio min-max", "8"],
  ["Social connectedness", "min-max", "5"],
  ["Common religion", "min-max", "8"],
  ["Diplomatic disagreement", "log rank", "8"],
];

function addBg(slide) {
  slide.background = { color: C.bg };
}

function addFooter(slide, n, source = "Source: presentation_ferm_results.ipynb; test split = 2019 H2.") {
  slide.addShape(pptx.ShapeType.line, { x: 0.62, y: 7.02, w: 12.1, h: 0, line: { color: C.rule, width: 0.6 } });
  slide.addText(source, { x: 0.65, y: 7.08, w: 8.9, h: 0.18, fontFace: "Aptos", fontSize: 7.8, color: C.slate, margin: 0 });
  slide.addText(String(n).padStart(2, "0"), { x: 12.25, y: 7.06, w: 0.45, h: 0.2, fontFace: "Aptos", fontSize: 8.5, bold: true, color: C.slate, align: "right", margin: 0 });
}

function addKicker(slide, text, x = 0.72, y = 0.42, color = C.teal) {
  slide.addShape(pptx.ShapeType.rect, { x, y: y + 0.04, w: 0.16, h: 0.16, fill: { color }, line: { color } });
  slide.addText(text.toUpperCase(), { x: x + 0.28, y, w: 5.2, h: 0.25, fontFace: "Aptos", fontSize: 8.5, bold: true, color, charSpace: 1.2, margin: 0 });
}

function addTitle(slide, title, subtitle = "", opts = {}) {
  const y = opts.y ?? 0.78;
  slide.addText(title, {
    x: opts.x ?? 0.72,
    y,
    w: opts.w ?? 11.2,
    h: opts.h ?? 0.82,
    fontFace: "Aptos Display",
    fontSize: opts.size ?? 30,
    bold: true,
    color: C.ink,
    fit: "shrink",
    margin: 0,
    breakLine: false,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: opts.x ?? 0.74,
      y: y + (opts.subY ?? 0.9),
      w: opts.subW ?? 10.5,
      h: 0.55,
      fontFace: "Aptos",
      fontSize: opts.subSize ?? 13.5,
      color: C.slate,
      fit: "shrink",
      margin: 0,
    });
  }
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  for (const item of items) {
    runs.push({
      text: item,
      options: {
        bullet: { type: "ul" },
        hanging: 3,
        breakLine: true,
      },
    });
  }
  slide.addText(runs, {
    x, y, w, h,
    fontFace: "Aptos",
    fontSize: opts.size ?? 14,
    color: opts.color ?? C.ink,
    breakLine: false,
    fit: "shrink",
    valign: "mid",
    margin: 0.04,
    paraSpaceAfterPt: 6,
  });
}

function addCallout(slide, heading, body, x, y, w, h, color = C.teal, fill = C.tealSoft) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, rectRadius: 0.04, fill: { color: fill, transparency: 6 }, line: { color: fill } });
  slide.addText(heading, { x: x + 0.16, y: y + 0.15, w: w - 0.32, h: 0.28, fontFace: "Aptos", fontSize: 10.5, bold: true, color, margin: 0 });
  slide.addText(body, { x: x + 0.16, y: y + 0.48, w: w - 0.32, h: h - 0.6, fontFace: "Aptos", fontSize: 12.3, color: C.ink, fit: "shrink", margin: 0 });
}

function addMetric(slide, value, label, x, y, w, color) {
  slide.addText(value, { x, y, w, h: 0.48, fontFace: "Aptos Display", fontSize: 26, bold: true, color, margin: 0, align: "center" });
  slide.addText(label, { x, y: y + 0.53, w, h: 0.34, fontFace: "Aptos", fontSize: 8.8, bold: true, color: C.slate, margin: 0, align: "center", fit: "shrink" });
}

async function imageBox(slide, file, x, y, w, h, opts = {}) {
  const full = path.join(FIG, file);
  const meta = await sharp(full).metadata();
  const imgRatio = meta.width / meta.height;
  const boxRatio = w / h;
  let iw = w, ih = h, ix = x, iy = y;
  if (imgRatio > boxRatio) {
    ih = w / imgRatio;
    iy = y + (h - ih) / 2;
  } else {
    iw = h * imgRatio;
    ix = x + (w - iw) / 2;
  }
  if (opts.backdrop) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: C.white, transparency: 0 }, line: { color: C.rule, width: 0.5 } });
  }
  slide.addImage({ path: full, x: ix, y: iy, w: iw, h: ih });
}

function addModelTable(slide) {
  const rows = [
    [
      { text: "Model", options: { bold: true, color: C.white } },
      { text: "Sigma structure", options: { bold: true, color: C.white } },
      { text: "Interpretation", options: { bold: true, color: C.white } },
    ],
    ["RM", "All Sigma = 0", "Neutral benchmark: population and distance only."],
    ["Traditional FERM", "Sigma_ij = destination attractiveness", "Tests broad destination absorption, here proxied by GDP."],
    ["Relational FERM", "Off-diagonal Sigma_ij from bilateral feature", "Tests whether corridor-specific affinity improves routes."],
  ];
  slide.addTable(rows, {
    x: 0.8, y: 2.1, w: 11.75, h: 2.2,
    border: { type: "solid", color: C.rule, pt: 0.7 },
    fill: { color: C.white },
    margin: 0.08,
    fontFace: "Aptos",
    fontSize: 12.2,
    color: C.ink,
    valign: "mid",
    colW: [2.2, 3.9, 5.65],
    autoFit: false,
    rowH: [0.42, 0.58, 0.62, 0.62],
    fit: "shrink",
    fill: { color: C.white },
    bandRow: false,
  });
  slide.addShape(pptx.ShapeType.rect, { x: 0.8, y: 2.1, w: 11.75, h: 0.42, fill: { color: C.ink }, line: { color: C.ink } });
}

function addMetricTable(slide) {
  const rows = [
    [
      { text: "Model", options: { bold: true, color: C.white } },
      { text: "Median improvement", options: { bold: true, color: C.white, align: "center" } },
      { text: "Share better", options: { bold: true, color: C.white, align: "center" } },
      { text: "Pearson log", options: { bold: true, color: C.white, align: "center" } },
      { text: "Zero pred.", options: { bold: true, color: C.white, align: "center" } },
    ],
    ...metrics.map(m => [
      m.model,
      m.med === null ? "baseline" : m.med.toFixed(3),
      m.share === null ? "baseline" : m.share.toFixed(3),
      m.pearson.toFixed(3),
      m.zero.toFixed(3),
    ]),
  ];
  slide.addTable(rows, {
    x: 0.78, y: 1.8, w: 11.8, h: 3.9,
    border: { type: "solid", color: C.rule, pt: 0.6 },
    margin: 0.07,
    fontFace: "Aptos",
    fontSize: 11,
    color: C.ink,
    valign: "mid",
    colW: [4.7, 1.8, 1.55, 1.55, 1.35],
    rowH: [0.42, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48],
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.rect, { x: 0.78, y: 1.8, w: 11.8, h: 0.42, fill: { color: C.ink }, line: { color: C.ink } });
  addCallout(
    slide,
    "Reading rule",
    "Median improvement is RM absolute log error minus model absolute log error. Positive values mean the model improves on RM.",
    0.78, 6.05, 11.8, 0.68, C.teal, C.tealSoft
  );
}

function addSelectedNormTable(slide) {
  const rows = [
    [
      { text: "Feature group", options: { bold: true, color: C.white } },
      { text: "Selected normalization", options: { bold: true, color: C.white } },
      { text: "sigma", options: { bold: true, color: C.white, align: "center" } },
    ],
    ...selectedNorms,
  ];
  slide.addTable(rows, {
    x: 7.15, y: 1.75, w: 5.35, h: 2.65,
    border: { type: "solid", color: C.rule, pt: 0.6 },
    margin: 0.06,
    fontFace: "Aptos",
    fontSize: 10.3,
    color: C.ink,
    valign: "mid",
    colW: [1.8, 2.75, 0.7],
    rowH: [0.35, 0.38, 0.38, 0.38, 0.38, 0.38],
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.rect, { x: 7.15, y: 1.75, w: 5.35, h: 0.35, fill: { color: C.ink }, line: { color: C.ink } });
}

async function build() {
  let s, n = 1;

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "FERM RESULTS", 0.75, 0.52, C.clay);
  s.addText("GDP wins globally; SCI is the relational signal worth taking seriously.", {
    x: 0.75, y: 1.1, w: 10.9, h: 1.35, fontFace: "Aptos Display", fontSize: 34, bold: true, color: C.ink, fit: "shrink", margin: 0,
  });
  s.addText("Feature normalization, relational FERM, and out-of-sample route-level evidence", {
    x: 0.78, y: 2.58, w: 9.2, h: 0.35, fontFace: "Aptos", fontSize: 14, color: C.slate, margin: 0,
  });
  addMetric(s, "0.037", "GDP median improvement", 0.8, 4.08, 2.2, C.clay);
  addMetric(s, "0.568", "GDP share better", 3.2, 4.08, 2.2, C.clay);
  addMetric(s, "0.562", "SCI share better", 5.6, 4.08, 2.2, C.teal);
  addMetric(s, "24.7%", "routes won by SCI", 8.0, 4.08, 2.2, C.teal);
  addCallout(s, "Thesis", "The relational extension is informative, but feature-dependent. GDP captures the dominant global destination-attractiveness gradient; SCI is the most promising bilateral feature.", 0.78, 5.4, 11.65, 0.95, C.teal, C.tealSoft);
  addFooter(s, n++, "Generated from presentation_ferm_results.ipynb outputs.");

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "SETUP", 0.72, 0.42, C.teal);
  addTitle(s, "The experiment compares one benchmark with two ways of injecting attractiveness.", "Distance is not a feature here: it is already part of the RM/FERM mechanism through the distance matrix.");
  addModelTable(s);
  addCallout(s, "What changes", "Only Sigma changes across FERM variants. RM keeps Sigma neutral; traditional FERM makes destinations attractive; relational FERM makes specific OD corridors attractive or unattractive.", 0.8, 5.0, 5.65, 1.05, C.clay, C.claySoft);
  addCallout(s, "Evaluation", "Tune normalization and sigma on validation_2019_h1. Report final results on test_2019_h2.", 6.9, 5.0, 5.65, 1.05, C.teal, C.tealSoft);
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "NORMALIZATION", 0.72, 0.42, C.clay);
  addTitle(s, "Normalization is a specification choice, so it is selected on validation.", "Every feature has a different scale and skew; the final test carries forward one selected version per feature group.");
  addSelectedNormTable(s);
  const steps = [
    ["1", "Build Sigma variants", "GDP, Abel, SCI, religion, diplomacy"],
    ["2", "Tune sigma on validation", "Same sigma grid for each variant"],
    ["3", "Select one per feature group", "Median improvement, then share better, then Pearson"],
    ["4", "Freeze and test", "Report only the selected feature specifications"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const y = 1.65 + i * 1.05;
    s.addShape(pptx.ShapeType.ellipse, { x: 0.9, y, w: 0.42, h: 0.42, fill: { color: i < 2 ? C.clay : C.teal }, line: { color: i < 2 ? C.clay : C.teal } });
    s.addText(steps[i][0], { x: 0.9, y: y + 0.095, w: 0.42, h: 0.16, fontSize: 9, bold: true, color: C.white, align: "center", margin: 0 });
    s.addText(steps[i][1], { x: 1.48, y: y - 0.03, w: 4.2, h: 0.28, fontSize: 14, bold: true, color: C.ink, margin: 0 });
    s.addText(steps[i][2], { x: 1.48, y: y + 0.27, w: 4.8, h: 0.25, fontSize: 10.5, color: C.slate, margin: 0 });
    if (i < steps.length - 1) s.addShape(pptx.ShapeType.line, { x: 1.11, y: y + 0.48, w: 0, h: 0.5, line: { color: C.rule, width: 1.0 } });
  }
  addFooter(s, n++, "Validation selection uses median route-level improvement as the primary criterion.");

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "VALIDATION", 0.72, 0.42, C.teal);
  addTitle(s, "Validation says GDP is strongest, while SCI is the relational feature worth carrying forward.", "High selected sigma means dilution toward RM-like behavior, not stronger feature influence.", { subW: 11.2 });
  await imageBox(s, "02_validation_sigma_tuning.png", 0.6, 1.65, 12.15, 4.55, { backdrop: true });
  addFooter(s, n++, "Validation split: 2019 H1.");

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "NORMALIZATION CHECK", 0.72, 0.42, C.clay);
  addTitle(s, "Across normalization variants, GDP remains the clearest global winner.", "The normalization sensitivity figure is the robustness story: each feature group is tested fairly before final test evaluation.");
  await imageBox(s, "02a_normalization_sensitivity_all_features.png", 0.85, 1.55, 11.65, 5.1, { backdrop: true });
  addFooter(s, n++, "Each row is one feature group; each line is one normalization variant.");

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "TEST RESULT", 0.72, 0.42, C.clay);
  addTitle(s, "Out of sample, GDP leads every main metric; SCI is second and meaningfully positive.", "Common religion is marginal; Abel and diplomatic disagreement underperform RM on median route-level error.", { subW: 11.2 });
  await imageBox(s, "03_test_summary_bars.png", 0.8, 1.65, 11.75, 4.65, { backdrop: true });
  addFooter(s, n++, "Test split: 2019 H2.");

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "EXACT METRICS", 0.72, 0.42, C.teal);
  addTitle(s, "The ranking is not subtle: GDP is globally best; SCI is the only strong relational candidate.", "Positive median improvement means the model reduced RM's absolute log-ratio error.");
  addMetricTable(s);
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "PREDICTION STRUCTURE", 0.72, 0.42, C.teal);
  addTitle(s, "All variants remain close to RM, but GDP and SCI make useful route-level shifts.", "This matters because high sigma values push the model back toward RM-like distributions.");
  await imageBox(s, "04_prediction_structure_vs_rm.png", 0.72, 1.52, 11.95, 5.2, { backdrop: true });
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "ROUTE ERRORS", 0.72, 0.42, C.clay);
  addTitle(s, "GDP and SCI improve many routes, but GDP shifts errors more consistently below the diagonal.", "Points below the diagonal mean the FERM variant has lower absolute log-ratio error than RM.");
  await imageBox(s, "05_route_error_scatter.png", 0.72, 1.52, 11.95, 5.2, { backdrop: true });
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "HETEROGENEITY", 0.72, 0.42, C.teal);
  addTitle(s, "The gains are not uniform: GDP and SCI help most on larger observed-flow routes.", "This is the strongest argument against a single headline metric: different features win different OD pairs.");
  await imageBox(s, "07_where_features_work_bins.png", 0.75, 1.42, 10.7, 5.2, { backdrop: true });
  addCallout(s, "Read", "Blue/positive cells indicate median route-level improvement over RM within each observed-flow or distance bin.", 11.65, 2.3, 1.05, 2.45, C.teal, C.tealSoft);
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "ROUTE WINNERS", 0.72, 0.42, C.clay);
  addTitle(s, "GDP wins the largest route share, but SCI wins a substantial subset of OD pairs.", "This is the cleanest evidence that relational information has value, even if it does not dominate globally.");
  await imageBox(s, "08_route_winners.png", 0.82, 1.7, 7.5, 3.9, { backdrop: true });
  addCallout(s, "Winner shares", "GDP wins 31.7% of routes. SCI wins 24.7%. RM still wins 11.4%, mostly lower-observed-flow routes.", 8.75, 1.9, 3.45, 1.4, C.clay, C.claySoft);
  addCallout(s, "Interpretation", "The model extension is feature-dependent: social connectedness contains useful bilateral signal, but GDP captures a broader absorption gradient.", 8.75, 3.65, 3.45, 1.35, C.teal, C.tealSoft);
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "ABEL DIAGNOSTIC", 0.72, 0.42, C.red);
  addTitle(s, "Abel is conceptually appealing, but this specification zero-predicts too many real routes.", "The zero routes are not only tiny flows; several large Gulf/South Asia corridors are set to zero.");
  addMetric(s, "20.7%", "Abel zero-prediction share", 0.85, 1.95, 2.5, C.red);
  addMetric(s, "57,919", "observed migrants on zero routes", 3.55, 1.95, 2.8, C.red);
  addMetric(s, "7,424", "largest observed zero route", 6.55, 1.95, 2.7, C.red);
  const zeroRows = [
    [{ text: "Route", options: { bold: true, color: C.white } }, { text: "Observed", options: { bold: true, color: C.white, align: "center" } }, { text: "Prediction", options: { bold: true, color: C.white, align: "center" } }],
    ["Sri Lanka -> Qatar", "7,424", "0"],
    ["Qatar -> Sri Lanka", "6,239", "0"],
    ["Bahrain -> Philippines", "6,137", "0"],
    ["Sri Lanka -> Kuwait", "4,500", "0"],
    ["Nepal -> Cyprus", "2,815", "0"],
  ];
  s.addTable(zeroRows, { x: 0.95, y: 3.45, w: 6.2, h: 2.5, colW: [3.55, 1.3, 1.1], rowH: [0.38, 0.36, 0.36, 0.36, 0.36, 0.36], border: { type: "solid", color: C.rule, pt: 0.6 }, fontFace: "Aptos", fontSize: 11.2, margin: 0.06, color: C.ink, fit: "shrink" });
  s.addShape(pptx.ShapeType.rect, { x: 0.95, y: 3.45, w: 6.2, h: 0.38, fill: { color: C.ink }, line: { color: C.ink } });
  addCallout(s, "Takeaway", "Abel may still be useful, but not as currently normalized and simulated. The immediate evidence is instability, not improvement.", 7.75, 3.45, 4.55, 1.45, C.red, "F4DADA");
  addFooter(s, n++);

  s = pptx.addSlide(); addBg(s);
  addKicker(s, "CONCLUSION", 0.72, 0.42, C.teal);
  addTitle(s, "The relational model is viable, but not every relational feature deserves the same confidence.", "", { h: 0.95 });
  const conclusions = [
    "GDP is the strongest global feature because it captures destination absorption.",
    "SCI is the only relational feature with a meaningful positive out-of-sample signal.",
    "Common religion is marginal; Abel and diplo are weak in this specification.",
    "Normalization matters, but it does not create signal where the feature is misaligned.",
    "The natural next model is combined: destination attractiveness plus relational corridor structure.",
  ];
  addBullets(s, conclusions, 0.92, 2.0, 7.2, 3.6, { size: 17 });
  addCallout(s, "Presentation sentence", "I would not claim that relational FERM failed. I would claim that the current relational features are uneven, and SCI is the most promising one.", 8.45, 2.05, 3.8, 1.8, C.teal, C.tealSoft);
  addCallout(s, "Next step", "Estimate or tune a combined Sigma with destination and relational components, then validate out of sample.", 8.45, 4.35, 3.8, 1.35, C.clay, C.claySoft);
  addFooter(s, n++);

  await pptx.writeFile({ fileName: FINAL });
  console.log(FINAL);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
