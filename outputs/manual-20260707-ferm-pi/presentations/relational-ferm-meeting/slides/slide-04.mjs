import { canvas, kicker, title, footer, image, callout, paths, C } from "./shared.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 4);
  kicker(slide, ctx, "TOY MODEL RESULT", C.TEAL);
  title(slide, ctx, "Quando il segnale relazionale esiste davvero, FERM lo recupera e lo usa meglio di RM.", 70, 34, 1070);
  await image(slide, ctx, paths.toyTest, 58, 166, 770, 410, "contain");
  await image(slide, ctx, paths.toyRobustness, 860, 166, 335, 210, "contain");
  callout(slide, ctx, "Lettura", "La feature corretta vince; le versioni noisy degradano; shuffled/random quasi perdono il vantaggio; anti-feature peggiora.", 860, 402, 335, 130, C.TEAL);
  ctx.addText(slide, {
    text: "Conclusione: il meccanismo FERM passa un sanity check controllato. Se sui dati reali il segnale è debole, il limite può essere feature/scaling, non necessariamente la struttura del modello.",
    x: 80, y: 612, width: 1020, height: 44,
    fontSize: 18, bold: true, color: C.INK,
  });
  footer(slide, ctx);
  return slide;
}
