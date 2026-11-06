# T1 — Snapshot e pesi: Qwen/Qwen3.8-Flash-Next

Data: 2026-11-06. Metodo: solo letture HTTP Range di `config.json`, indice safetensors e header
(`safetensors_metadata.inspect_hub_safetensors`); nessun payload scaricato, nessuna modifica al
profilo `qwen4_exp` (resta `unsupported`). Script: `t1_inventory.py`; dati grezzi:
`reports/qwen4-exp-t1-inventory.json` (1.658 tensori), riepilogo: `reports/qwen4-exp-t1-summary.json`.

## 1. Esito gate di accesso

| Voce | Esito |
| --- | --- |
| Revisione fissata nel profilo esperimento `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` | **NON risolve** (`RevisionNotFoundError`), assente da branch, tag e convert |
| Branch disponibili | solo `main` (nessun tag) |
| `main` al momento dell'ispezione | `7ecef55a2e549814c62143fcf8b5be457695e3b9` |
| Accesso | repo pubblico, non gated, leggibile senza token |
| `quantization_config` | assente (checkpoint BF16 non quantizzato) |

**Conseguenza per G1:** lo SHA del profilo esperimento è irreperibile; serve una decisione
esplicita per fissare una nuova revisione (candidata: `7ecef55a…`, da ripinare il giorno del Job).
Non è stato fatto alcun fallback silenzioso a `main` nel codice: l'inventario è etichettato
`main@7ecef55a` e va trattato come diagnostica, non come abilitazione.

## 2. Inventario pesi (main@7ecef55a)

| Categoria | Tensori | Forme | GiB |
| --- | ---: | --- | ---: |
| Router backbone (`layers.0-47.mlp.gate.weight`) | 48 | [512, 2560] BF16 | 0,12 |
| Esperti fusi backbone (`experts.gate_up_proj` / `down_proj`) | 96 | [512, 1280, 2560] / [512, 2560, 640] BF16 | 225,00 |
| Shared expert backbone (3 tensori × 48) + gate | 192 | [640,2560] ×2, [2560,640], [1,2560] | 0,45 |
| **Blocco MoE MTP** (`mtp.layers.0.mlp.*`) | 7 | identiche al backbone, banca indipendente | ≈ 4,69 |
| PLE (`layers.1.ple.*`) | 137 | vedi §3 | 95,43 |
| Embeddings (`embed_tokens`, `lm_head`) | 2 | [248320, 2560] BF16 | 2,36 |
| Visione (`model.visual.*`) | 333 | 13 forme distinte | 0,84 |
| Altro (GDN/QSA, hyper-connection, norm) | 819 | — | 6,23 |
| **Totale** | **1.658** | 131 shard, max 3,27 GiB/shard | **335,35** |

Census dtype: 1.655 BF16, 3 I64 (metadati PLE).

## 3. Confronto con il contratto dedotto dalla specifica

| Aspetto | Specifica (dedotta dal codice) | Lettura reale | Esito |
| --- | --- | --- | --- |
| Layout esperti su disco | fuso `[E,2I,H]`/`[E,H,I]` oppure per-esperto (vietato dedurre) | **fuso su disco** | confermato |
| Router | `gate.weight` [512,2560], 48 layer | 48 × [512,2560] | confermato |
| PLE | righe totali 320.001.536, dim 160 | 128 shard × [2.500.012, 160] = 320.001.536 | **esatto** |
| PLE layer | `ple_layer_ids [2]` (base 1) | tensori in `layers.1` (base 0) | confermato |
| PLE formato su disco | shard da concatenare a runtime | `ngram_embedding.shard_0…127` + 3 tensori I64 di metadati | confermato |
| `norm_topk_prob` | assente da config, default upstream True | **assente** → default da legare al runtime in T4 | aperto |
| MTP | classe upstream ignora `^mtp.*`; policy da decidere | **MTP ha un blocco MoE completo e indipendente** (esperti [512,…], QSA, hyper-connection) ≈ 4,7 GiB | decisione G1 richiesta |
| Visione | conservata, non validata | 333 tensori, 0,84 GiB, `deepstack_visual_indexes` vuoto | inventariata |
| `transformers_version` | `5.8.0.dev0` (non riproducibile) | `5.8.0.dev0` | aperto in T2 |

## 4. Implicazioni per i gate successivi

- **T4/adapter:** il layout fuso su disco semplifica il caricamento (nessuna conversione
  per-esperto); la maschera/compattazione sullo spazio 0…511 resta validata dall'esperimento
  isolato. Il default `norm_topk_prob=True` va risolto nel profilo legandolo al runtime fissato.
- **MTP (decisione G1):** la banca esperti MTP è separata dalla configurazione condivisa:
  prune indipendente, conservazione a 512, o disabilitazione esplicita con metadati coerenti.
- **T3/memoria:** il pavimento è invariato: esperti 225 + PLE 95,4 ≈ 320 GiB BF16, più
  backbone non-MoE ≈ 10 GiB e MTP ≈ 4,7 GiB. Budget pratici: B=384 → artefatto ≈ 279 GiB;
  B=496 → ≈ 328 GiB. Picco reale di compattazione da sommare (concat PLE, tensori nuovi,
  salvataggio). `l4x1` con `.to(cuda)` resta escluso.
- **T6/manifest:** il manifest deve elencare le 128 parti PLE, i 3 tensori I64 di metadati e
  il blocco MTP come voci esplicite (incluse eventuali chiavi escluse), e non copiare
  l'indice sorgente dopo la riscrittura dei pesi.

Nessuna abilitazione del profilo è stata eseguita: `architectures/qwen4_exp.json` è invariato.
