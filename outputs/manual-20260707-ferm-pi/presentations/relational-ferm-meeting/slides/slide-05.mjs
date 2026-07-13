import { canvas, kicker, title, footer, image, callout, paths, C } from "./shared.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 5);
  kicker(slide, ctx, "CALIBRATION", C.BLUE);
  title(slide, ctx, "Nel toy model sigma viene scelto su calibration, poi il modello è valutato su test.", 70, 34, 1110);
  await image(slide, ctx, paths.toyCalibration, 58, 158, 1100, 285, "contain");
  callout(slide, ctx, "Criterio principale", "Scelgo sigma minimizzando la codelength multinomiale sul calibration set: meno bit per migrante, migliore descrizione probabilistica.", 58, 460, 352, 136, C.BLUE);
  callout(slide, ctx, "Controllo meccanistico", "La feature relazionale corretta mantiene il vantaggio; random, shuffled e anti-feature perdono informazione.", 448, 460, 352, 136, C.TEAL);
  callout(slide, ctx, "Separazione pulita", "Il test non entra nella scelta di sigma. Verifica solo se il segnale calibrato generalizza fuori campione.", 838, 460, 352, 136, C.GOLD);
  ctx.addText(slide, {
    text: "La calibrazione è quindi parte del modello, non un aggiustamento ex post: prima scelgo sigma, poi misuro la performance finale su dati non usati nella scelta.",
    x: 82, y: 626, width: 1000, height: 38,
    fontSize: 18, bold: true, color: C.INK,
  });
  footer(slide, ctx);
  return slide;
}
