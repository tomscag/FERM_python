import { canvas, kicker, title, footer, image, callout, paths, C } from "./shared.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 7);
  kicker(slide, ctx, "NORMALIZATION BOTTLENECK", C.GOLD);
  title(slide, ctx, "La normalizzazione non è neutrale: cambia la forza effettiva della feature dentro Sigma.", 70, 34, 1080);
  await image(slide, ctx, paths.normalization, 60, 166, 560, 340, "contain");
  await image(slide, ctx, paths.sciDistribution, 650, 166, 550, 340, "contain");
  callout(slide, ctx, "Rischio di selezione", "Scegliere la normalizzazione che performa meglio rischia di diventare un livello nascosto di model selection.", 78, 516, 350, 116, C.RED);
  callout(slide, ctx, "Implicazione per GDP + SCI", "Sommo GDP e SCI solo se le loro scale sono comparabili. Altrimenti introduco pesi impliciti.", 462, 516, 350, 116, C.GOLD);
  callout(slide, ctx, "Regola da fissare", "Serve una scelta a priori: scala off-diagonal comune, oppure pesi espliciti alpha/beta.", 846, 516, 350, 116, C.TEAL);
  footer(slide, ctx);
  return slide;
}
