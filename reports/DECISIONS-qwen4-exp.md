# Decisioni sui punti aperti — percorso `qwen4_exp` (HF)

Data: 2026-11-06. Contesto: analisi del piano, T0 (baseline `3bbcdef`), T1 (inventario
`main@7ecef55a`, report `reports/T1-report-qwen4-exp.md`), T3 (fattibilità Job,
`reports/T3-report-job-feasibility.md`). Nessuna abilitazione del profilo: le decisioni
sono vincolanti per T2/T4/T5/T6 ma `architectures/qwen4_exp.json` resta `unsupported` fino a G2.

## D1 — Revisione del checkpoint

**Decisione:** si abbandona lo SHA `b89a7c4b…` (inesistente, assente da ogni ref) e si fissa
esplicitamente `7ecef55a2e549814c62143fcf8b5be457695e3b9` come revisione di riferimento,
ri-verificata alla sottomissione del Job: se il commit non risolve più, il Job fallisce
apertamente. Nessun fallback silenzioso a `main`.

**Rationale:** il vincolo "no silent `main`" resta; la ri-pinning è una decisione tracciata, non un fallback.

## D2 — Layout dei pesi su disco

**Decisione:** il layout fuso letto in T1 (`experts.gate_up_proj` [512,1280,2560],
`down_proj` [512,2560,640]) è il formato di riferimento; nessun layer di conversione
per-esperto. PLE resta sharded su disco (128 × [2.500.012,160], come sorgente); l'export usa
`save_original_format=True` per rimanere byte-compatibile con il layout upstream, verificato
in round-trip in T6. Vietato copiare l'indice sorgente dopo la riscrittura.

## D3 — `norm_topk_prob`

**Decisione:** il profilo dichiarativo fissa esplicitamente `norm_topk_prob: true` (valore
effettivo del default upstream), legato al commit del runtime fissato in T2; il preflight
rifiuta un checkpoint il cui `config.json` dichiari esplicitamente un valore diverso.

## D4 — Policy MTP

**Decisione (v1):** MTP è **disabilitato esplicitamente nell'artefatto**: chiavi `mtp.*`
escluse dal checkpoint con `mtp_num_hidden_layers=0` (o rimozione equivalente della config),
e il manifest elenca le chiavi escluse ricavate dall'indice sorgente.

**Rationale:** la classe upstream ignora `^mtp.*` al caricamento, quindi un MTP "conservato"
non sopravvive al round-trip con il loader nativo: conservarlo renderebbe il manifest
disonesto. La banca MoE indipendente scoperta in T1 (≈ 4,7 GiB) rende il pruning dedicato
*possibile* ma non gratuito: la selezione andrebbe profilata su un blocco che il loader non
ricarica. Il percorso "MTP preservato e potato" resta un lavoro post-G2 con loader patchato o
percorso dedicato (spec §5).

## D5 — Visione

**Decisione:** visione preservata intatta (0,84 GiB, 333 tensori) nel wrapper multimodale;
l'artefatto dichiara "visione preservata, non validata dalla prova testuale" (perimetro
della spec §5). Nessuna modifica ai pesi visuali.

## D6 — Runtime Transformers

**Decisione:** il runtime Qwen4 è fissato a un **commit esatto** di Transformers che
dichiari `qwen4_exp` (da produrre in T2), mai alla stringa `5.8.0.dev0`. Il vincolo globale
`4.55.4` resta invariato per OLMoE/Qwen2/Qwen3. Il controllo in `preflight.check_metadata`
diventa **per-profilo**: il runtime dell'ambiente deve corrispondere alla versione dichiarata
nel profilo (oggi il confronto è hard-coded su `4.55.4` e rifiuterebbe il nuovo profilo).

## D7 — Placement

**Decisione:** per il percorso qwen4, `load_model` usa `device_map='auto'` +
`low_cpu_mem_usage=True` (richiede `accelerate`) e **non** chiama `.to()`. È confermato che
il resto della pipeline (sonde, maschera, compattazione) è già compatibile con lo sharding.
Purge della cache sorgente dopo il load per il disco.

## D8 — Flavor, timeout, budget, sequenza smoke

**Decisione:**
- Smoke test (T7): flavor **h200x2**, timeout **12h**, budget atteso **25–40 $**;
  alternativa a parità di prezzo **a100x4**; opzione economica **l40sx4** (margine più stretto).
- Prima esecuzione a pagamento: **controllo identità B=512** (pipeline end-to-end senza
  rischio qualità; il costo è dominato da load/profilazione, non dalla compattazione).
- Poi pruning reali con audit separato sui budget **384/448/480/496** (+ 512 controllo),
  come da tabella frazioni della spec §6.

## D9 — Payload del Job

**Decisione:** `hf_jobs.SOURCES` e il bootstrap vengono estesi per il percorso qwen4:
aggiunta dei nuovi moduli (inspector, adapter) e di un requirements dedicato
(`requirements-qwen4.txt`, con `accelerate` e il pin Transformers di D6), selezionato dal
bootstrap in base al profilo della richiesta. I profili esistenti continuano a usare i file
attuali, invariati.

## D10 — Rollback

**Decisione:** l'abilitazione è **un solo commit** che modifica `architectures/qwen4_exp.json`
da `unsupported` a `qwen4_exp_fused_topk_v1`; il rollback è il revert di quel commit. Tutto
il lavoro pre-G2 è inerte di default (il profilo fallisce chiuso), quindi non esiste uno
stato "mezzo abilitato". I Job girano su commit fissati: la riproducibilità è per revisione,
non serve un meccanismo di abort oltre a cancel.

## Aperti di proposito (per G1/G2, non decisi ora)

- Il commit esatto del runtime (output di T2).
- Il confronto finale dei risultati su checkpoint reale (G2, con revisore indipendente).
- Eventuale percorso MTP preservato e studio GGUF (post-G2, perimetro Nico).
