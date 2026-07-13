import { canvas, kicker, title, footer, image, callout, paths, C } from "./shared.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 3);
  kicker(slide, ctx, "CONTROLLED CHECK", C.BLUE);
  title(slide, ctx, "Nel toy model la vera matrice relazionale è nota, quindi possiamo testare FERM in modo pulito.", 70, 34, 1100);
  await image(slide, ctx, paths.toySetup, 60, 170, 760, 430, "contain");
  callout(slide, ctx, "Perché serve", "Nei dati reali non sappiamo qual è la vera Sigma. Qui invece la conosco, genero i flussi da FERM, e chiedo se il modello riesce a recuperare il segnale.", 860, 180, 340, 132, C.BLUE);
  callout(slide, ctx, "Cosa confronto", "RM contro FERM con feature corretta, feature rumorosa, feature shuffled, feature random e anti-feature.", 860, 335, 340, 118, C.TEAL);
  callout(slide, ctx, "Criterio", "Sigma viene calibrato su dati sintetici di calibration; la valutazione finale è su test indipendente.", 860, 476, 340, 104, C.GOLD);
  footer(slide, ctx);
  return slide;
}
