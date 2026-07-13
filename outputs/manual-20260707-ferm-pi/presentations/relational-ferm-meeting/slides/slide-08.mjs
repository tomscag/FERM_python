import { canvas, kicker, title, footer, callout, equation, C } from "./shared.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 8);
  kicker(slide, ctx, "DESCRIPTION LENGTH", C.TEAL);
  title(slide, ctx, "Nel codice la description length è calcolata come codelength multinomiale più costo del modello.", 70, 33, 1110);

  equation(slide, ctx, "p_ij^M = E_ij^M / sum_k E_ik^M", 82, 178, 470, C.INK);
  equation(slide, ctx, "L(D | M) = - sum_i sum_j O_ij log2(p_ij^M)", 82, 276, 590, C.INK);
  equation(slide, ctx, "L(D, M) = L(D | M) + L(M)", 82, 374, 470, C.INK);

  callout(slide, ctx, "1 · Dati condizionati all'origine", "Il totale osservato in uscita da ogni origine è fissato. Il modello viene valutato su come redistribuisce quel totale sulle destinazioni.", 720, 176, 430, 116, C.BLUE);
  callout(slide, ctx, "2 · Probabilità dal modello", "Dalle predizioni attese E_ij ricavo p_ij normalizzando per origine. Questo rende RM e FERM confrontabili come modelli di scelta della destinazione.", 720, 318, 430, 132, C.TEAL);
  callout(slide, ctx, "3 · Costo del modello", "Nella versione strict includo scelta del modello, calibrazione di sigma sulla griglia, e costo della matrice Sigma: n² entries × b bits, con b=16 come assunzione di sensitività.", 720, 476, 430, 142, C.GOLD);

  ctx.addText(slide, {
    text: "Interpretazione: FERM è giustificato solo se la riduzione di L(D | M) supera il costo informativo aggiuntivo L(M).",
    x: 88, y: 586, width: 560, height: 44,
    fontSize: 19, bold: true, color: C.INK,
  });
  footer(slide, ctx);
  return slide;
}
