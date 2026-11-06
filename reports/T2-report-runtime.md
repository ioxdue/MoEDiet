# T2 — Runtime e wrapper: pin e verifica isolata

Data: 2026-11-06. Ambiente: `.venv-qwen4` (isolato, CPU), PyTorch 2.15.0+cpu, Transformers
installato dal commit git esatto `cdb75de4adf71059c01f58dbd2d24d85084a397d` (lo stesso fissato
dalla specifica §1). Il runtime globale 4.55.4 non è stato toccato. Script: `t2_runtime_check.py`;
esito grezzo: `reports/qwen4-exp-t2-runtime.json`.

## Risultati

| Controllo | Esito |
| --- | --- |
| Installazione dal commit git | ✓ riproducibile (content-hash), versione `5.23.0.dev0` |
| Import classi native | ✓ `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM`, `Qwen4ExpTextConfig` |
| `AutoConfig` sul checkpoint reale (`7ecef55a…`) | ✓ `Qwen4ExpConfig`, `model_type=qwen4_exp`, architettura `Qwen4ExpForConditionalGeneration`, 48 layer / 512 esperti testuali |
| `norm_topk_prob` effettivo | ✓ `True` (default della classe; assente nel JSON come da D3) |
| Fixture ridotta (4 layer: 3 GDN + QSA all'indice 3, PLE sul layer 2, MoE E=8/K=2, campi indexer espliciti) | ✓ costruzione e forward, logits [2,7,128] |
| Contratto router a tripla | ✓ forme `[T,E] [T,K] [T,K]` = `[3,8] [3,2] [3,2]` |
| Round-trip `save_pretrained`/`from_pretrained` | ✓ diff 0,0; nessuna chiave mancante o inattesa |
| `device_map='auto'` + `low_cpu_mem_usage` (accelerate) | ✓ caricamento e forward, diff 0,0 (decisione D7 verificata) |

Note: i kernel GDN ottimizzati (`causal_conv1d`, `flash-linear-attention`) non sono installati;
Transformers dichiara esplicitamente il fallback PyTorch "correct but much slower". Sul Job GPU
questo rallenta la profilazione: valutare in T5 l'installazione di `flash-linear-attention`
(`causal_conv1d` richiede build CUDA e non ha wheel Windows; il fallback resta la base di sicurezza).

Due vincoli scoperti sulla config, da riportare in T4: `eos_token_id` obbligatorio quando PLE è
attivo (validatore `validate_architecture`), e i campi QSA `indexer_*` non hanno default
(valori produzione: budget 2048, compress_ratio 4, head_dim 128, n_heads 4, kv_heads 1).

## Pin registrato (aggiornamento D6)

- Transformers: commit git `cdb75de4adf71059c01f58dbd2d24d85084a397d` (versione `5.23.0.dev0`).
- PyTorch: `2.15.0+cpu` per i test locali; sul Job l'immagine porta torch 2.8.0 — la matrice
  torch×transformers sul flavor scelto va verificata allo smoke (T7), con fallback al pin
  torch del Job se il fixture-scale passa ma il caricamento reale no.
- La stringa `5.8.0.dev0` del checkpoint è il metadato di produzione del modello, non una
  versione installabile: resta esclusa dal pin, come da decisione.

## Esito

Il runtime installabile esiste e supporta modello e wrapper. G1 ha ora i tre ingressi
(T1 pesi, T2 runtime, T3 infrastruttura) tutti positivi, condizionati alle decisioni D1–D10.
