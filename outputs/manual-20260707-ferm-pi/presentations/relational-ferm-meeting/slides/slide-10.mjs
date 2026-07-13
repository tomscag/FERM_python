import { canvas, kicker, title, footer, callout, C } from "./shared.mjs";

export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  canvas(slide, ctx, 10);
  kicker(slide, ctx, "DISCUSSION", C.BLUE);
  title(slide, ctx, "Decisioni da prendere: normalizzazione, costo del modello, e prossimo esperimento controllato.", 70, 34, 1080);
  callout(slide, ctx, "1 · Fissare una normalizzazione principled", "Opzione A: tutte le matrici su scala off-diagonal comune. Opzione B: trasformazioni feature-specific ma pesi espliciti alpha/beta.", 80, 190, 520, 132, C.GOLD);
  callout(slide, ctx, "2 · Usare description length come criterio", "Valutare RM e FERM con L(D|M)+L(M), includendo costo di sigma, scelta feature e costo/assunzione sulla matrice Sigma.", 680, 190, 520, 132, C.TEAL);
  callout(slide, ctx, "3 · Rafforzare il toy model", "Variare intensità del segnale, rumore, sparsità, dimensione network e vedere quando FERM recupera/non recupera la feature.", 80, 370, 520, 132, C.BLUE);
  callout(slide, ctx, "4 · Tornare ai dati reali con una pipeline bloccata", "Una volta fissata la normalizzazione, rifare calibration/test e confrontare GDP, SCI, Abel e combinazioni senza cambiare regole a posteriori.", 680, 370, 520, 132, C.RED);
  ctx.addText(slide, {
    text: "Sintesi: il toy model mostra che il meccanismo può funzionare; i dati reali mostrano che il collo di bottiglia è la rappresentazione empirica di Sigma.",
    x: 110, y: 602, width: 980, height: 38,
    fontSize: 20, bold: true, color: C.INK, align: "center",
  });
  footer(slide, ctx);
  return slide;
}
