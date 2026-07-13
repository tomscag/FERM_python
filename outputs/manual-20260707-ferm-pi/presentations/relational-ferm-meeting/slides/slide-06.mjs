import { canvas, kicker, title, footer, image, callout, paths, C } from "./shared.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 6);
  kicker(slide, ctx, "EMPIRICAL RESULTS", C.BLUE);
  title(slide, ctx, "Sui dati reali il miglioramento esiste, ma è più misto e dipende molto dalla feature.", 70, 34, 1080);
  await image(slide, ctx, paths.empiricalRank, 58, 168, 760, 402, "contain");
  callout(slide, ctx, "Risultato empirico", "GDP-FERM è il segnale più stabile. Le feature relazionali aiutano alcune rotte, ma non dominano RM con la stessa chiarezza del toy model.", 858, 178, 342, 146, C.BLUE);
  callout(slide, ctx, "Interpretazione", "Questo non falsifica FERM: sui dati reali la feature vera non è nota, e la scala di Sigma diventa parte del modello.", 858, 350, 342, 126, C.TEAL);
  callout(slide, ctx, "Rischio metodologico", "Se una feature migliora poco, può essere debole, rumorosa, normalizzata male, o combinata con pesi impliciti non controllati.", 858, 502, 342, 112, C.GOLD);
  footer(slide, ctx);
  return slide;
}
