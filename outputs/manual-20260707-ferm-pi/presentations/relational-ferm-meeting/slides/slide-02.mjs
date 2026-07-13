import { canvas, kicker, title, footer, railItem, C } from "./shared.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 2);
  kicker(slide, ctx, "NARRATIVE ARC", C.BLUE);
  title(slide, ctx, "La storia è in tre passaggi: prima il meccanismo, poi i dati reali, infine la normalizzazione.", 72, 36, 1080);
  railItem(slide, ctx, "1", "Toy model controllato", "Costruisco un mondo sintetico in cui conosco la vera matrice relazionale. Se FERM non funziona qui, il problema è strutturale.", 92, 208, C.BLUE);
  railItem(slide, ctx, "2", "Applicazione empirica", "Sui dati veri non conosco la feature giusta. Valuto se GDP, SCI e altre feature aggiungono informazione rispetto a RM.", 92, 342, C.TEAL);
  railItem(slide, ctx, "3", "Nodo metodologico", "La normalizzazione non è un dettaglio tecnico: determina la forza effettiva della feature dentro Sigma.", 92, 476, C.GOLD);
  ctx.addShape(slide, { x: 675, y: 210, width: 1, height: 350, fill: C.RULE });
  ctx.addText(slide, {
    text: "Tesi operativa",
    x: 735, y: 236, width: 360, height: 30,
    fontSize: 25, bold: true, color: C.INK,
  });
  ctx.addText(slide, {
    text: "Il toy model verifica che il meccanismo FERM può recuperare segnale relazionale quando la vera matrice Sigma è nota. Sui dati reali, invece, il risultato dipende soprattutto da quale informazione entra in Sigma e con quale scala.",
    x: 735, y: 286, width: 395, height: 170,
    fontSize: 20, color: C.MUTED,
  });
  footer(slide, ctx);
  return slide;
}
