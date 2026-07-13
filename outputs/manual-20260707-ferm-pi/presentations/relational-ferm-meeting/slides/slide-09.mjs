import { canvas, kicker, title, footer, image, callout, equation, paths, C } from "./shared.mjs";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 9);
  kicker(slide, ctx, "MODEL COST", C.TEAL);
  title(slide, ctx, "Description length sposta la domanda: non solo “migliora?”, ma “vale il costo?”.", 70, 34, 1090);
  await image(slide, ctx, paths.mdl, 60, 170, 710, 382, "contain");
  equation(slide, ctx, "L(D, M) = L(D | M) + L(M)", 820, 188, 350, C.INK);
  callout(slide, ctx, "Perché conta", "RM è più semplice. FERM deve descrivere meglio i dati abbastanza da compensare Sigma, sigma e la scelta della feature.", 810, 270, 370, 126, C.TEAL);
  callout(slide, ctx, "Uso nel progetto", "Se il guadagno in bits supera il costo informativo, allora anche un miglioramento moderato è metodologicamente difendibile.", 810, 428, 370, 124, C.BLUE);
  footer(slide, ctx);
  return slide;
}
