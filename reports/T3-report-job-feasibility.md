# T3 — Fattibilità Job HF: flavor, memoria, placement

Data: 2026-11-06. Metodo: solo letture API pubbliche (`https://huggingface.co/api/jobs/hardware`,
catalogo completo in `reports/qwen4-exp-t3-flavors.json`) e analisi statica del codice di
caricamento. **Nessun Job è stato avviato o pagato.**

## 1. Vincoli dimensionali (da T1, main@7ecef55a)

| Voce | Valore |
| --- | ---: |
| Payload checkpoint sorgente | 335,35 GiB (1.658 tensori BF16, 131 shard) |
| Artefatto pruned B=384 / B=496 / B=512 | ≈ 279 / 328 / 335 GiB |
| RAM minima per `from_pretrained` senza `device_map` | ≈ 345–350 GiB (tutto in RAM) |
| VRAM per `.to('cuda')` globale su modello intero | ≈ 340+ GiB + attivazioni di profilazione |
| Disco di picco (cache sorgente + risultato stadiato + finale) | ≈ 900 GiB (B=384), ≈ 1,0 TiB (controllo B=512) |

Timeout: il default del profilo è 90 minuti — insufficiente (download + load + profilazione +
compattazione + upload realisticamente 2,5–4 h). Il formato config ammette `12h`.

## 2. Catalogo flavor e matrice di fattibilità

VRAM/RAM in GB decimali; "fit" = sorgente + compattazione + salvataggio sostenibili.

| Flavor | VRAM | RAM | Disco | $/min | Senza modifice (`.to('cuda')`) | Con `device_map=auto` + `low_cpu_mem_usage` |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| t4-small/medium, a10g-*, cpu-* | ≤ 96 | ≤ 256 | ≤ 1 TB | — | ✗ | ✗ (RAM/disk insufficienti) |
| l4x1 / l4x4 | 24 / 96 | 30 / 186 | 0,4 / 3,2 TB | 0,013–0,063 | ✗ | ✗ RAM |
| a100-large | 80 | 142 | 1 TB | 0,042 | ✗ | ✗ |
| **l40sx4** | 192 | 382 | 3,2 TB | 0,138 | ✗ | ✓ offload ~150 GiB in RAM |
| **h200** | 141 | 256 | 3 TB | 0,083 | ✗ | ✓ (marginale; offload ~200 GiB) |
| **h200x2** | 282 | 512 | 6 TB | 0,167 | ✗ | ✓ **consigliato** |
| **a100x4** | 320 | 568 | 4 TB | 0,167 | ✗ | ✓ **consigliato** (quasi tutto su GPU) |
| rtx-pro-6000x4 | 384 | 1024 | 1,9 TB | 0,183 | ✗ | ✓ (disco giusto) |
| h200x4 / a100x8 / l40sx8 / rtx-pro-6000x8 | 564–2048 | 1–2 TB | ≥ 3,8 TB | 0,33–0,67 | ✓ solo x8 | ✓ margine largo, costo alto |

Con il codice **attuale e invariato** (`.to('cuda')` globale su tutto il modello) il Job è
praticamente irrealizzabile: servirebbero ≥ 340 GiB di VRAM singola, cioè solo flavor x8
(da $0,33/min) e comunque al limite per `l40sx8` (384 GB). La modifica di placement non è
un optional ma un prerequisito.

## 3. Modifica di placement richiesta (per T4/T5)

Unica incompatibilità trovata: `model.to(device)` in `moediet.load_model`
(`moediet.py:83`). Il resto della pipeline è già compatibile con lo sharding:

- Input delle sonde seguono il device dell'embedding (`fidelity.py:19`, `routing.py:23`) — coerente con `device_map` di accelerate.
- Maschera e compattazione operano sul device di ciascun peso (`adapters.py:107,121-122`) — coerente con moduli distribuiti.
- Compattazione sequenziale per blocco: picco transitorio = un tensore vecchio + uno nuovo (max ~3,2 GiB per `gate_up_proj`), nessuna duplicazione dell'intero modello.

Modifica proposta, isolata al percorso qwen4: in `load_model`, per il nuovo profilo usare
`AutoModelForCausalLM.from_pretrained(..., device_map='auto', low_cpu_mem_usage=True)` e
**non** chiamare `.to()`; richiede `accelerate` nei requirements dedicati (T5). Note:
verificare che non esistano hook di dispatch da rispettare alla sostituzione dei parametri
(spec §5); prevedere purge della cache sorgente dopo il load per alleggerire il disco.

## 4. Stima costo per la decisione G1

Una build B=384 (download 335 GiB, load, profilazione, compattazione, salvataggio, upload):
2,5–4 h → **h200x2 o a100x4 ≈ 25–40 $**; l40sx4 ≈ 21–33 $. Il controllo B=512 raddoppia.
Timeout consigliato 8–12 h. Flavor consigliato per il primo smoke: **h200x2** (equilibrio
VRAM/RAM/disco) con **a100x4** come alternativa a parità di prezzo.

## 5. Esito

Fattibilità hardware: **sì, condizionata** a (a) modifica di placement con `device_map`,
(b) flavor ≥ h200x2/a100x4 o equivalente, (c) timeout e budget rivisti. Senza modifica del
loader, il percorso HF resta non sostenibile economicamente. Nessun Job avviato.
