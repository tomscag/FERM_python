import { canvas, kicker, title, subtitle, footer, equation, C } from "./shared.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 1);
  kicker(slide, ctx, "PROJECT STATUS", C.TEAL);
  title(slide, ctx, "Relational FERM: il meccanismo funziona, il nodo empirico è la normalizzazione.", 92, 42, 1080);
  subtitle(slide, ctx, "Sanity check controllato, risultati su dati reali, e decisione metodologica su scaling e combinazione delle feature.", 60, 230, 850);
  equation(slide, ctx, "L(D, M) = L(D | M) + L(M)", 730, 394, 430, C.INK);
  ctx.addText(slide, {
    text: "Obiettivo: non vendere FERM come black box predittiva, ma capire se aggiungere informazione interpretabile migliora RM abbastanza da giustificare il costo del modello.",
    x: 730, y: 470, width: 420, height: 100,
    fontSize: 18, color: C.MUTED,
  });
  ctx.addShape(slide, { x: 58, y: 586, width: 620, height: 1, fill: C.RULE });
  ctx.addText(slide, {
    text: "Tesi centrale: FERM passa un test controllato; sui dati reali il collo di bottiglia è come costruiamo, normalizziamo e combiniamo Sigma.",
    x: 58, y: 606, width: 800, height: 40,
    fontSize: 18, bold: true, color: C.INK,
  });
  footer(slide, ctx);
  return slide;
}
