# Raw data formats (`data/raw`)

| Path | Content |
|---|---|
| `MANIFEST.json` | generator version, seed, scale, file list |
| `fleet/*.csv` | aircraft, LRU catalogue, LRU installations (serial history) |
| `flights/flights.csv` | flight schedule, mission type, hours |
| `icd/parameter_dictionary.csv`, `icd/bus_mapping.csv` | parameters (units, ranges, rates) and the synthetic 1553 ICD (RT/SA/word/bits/scale/offset) |
| `bus1553/<aircraft>/<flight>.bin` / `.hex` | bus captures (below) |
| `lru_logs/<aircraft>/<flight>_<lru>.csv` | LRU internal logs: `lru_time` (LRU clock, epoch s, drifting/offset), `lru_serial`, parameter columns |
| `snags/snags.csv`, `maintenance/maintenance_records.csv` | free-text snags and maintenance actions |
| `fta/*.csv` | fault trees: `event_code, parent_event, gate (AND/OR/NOT), event_type, lru_type, indicating_parameters (;-separated), reference_document, corrective_action, ...` |
| `documents/*.txt|*.pdf` | synthetic FID/FIM/MM/test procedures/engineering notes with `Document ID/Version/Revision` headers |

## 1553 binary capture (`.bin`)

64-byte little-endian header followed by fixed 100-byte records.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `SYN1553\0` |
| 8 | 2 | format version (1) |
| 10 | 16 | aircraft id (NUL padded) |
| 26 | 24 | flight id (NUL padded) |
| 50 | 4 | record count |
| 54 | 8 | capture start (µs since epoch, signed) |

Record (`services/ingestion/app/decode.py: RECORD`): `message_id u64, timestamp_us i64, bus_id u8, rt u8, sa u8,
direction u8, command_word u16, status_word u16, word_count u8, message_validity u8, error_status u16,
message_sequence u32, parity u32, data u16[32]`. `parity` packs one odd-parity bit per data word (bit i = word i).
Messages with `message_validity=0`, a non-zero `error_status` or a parity mismatch are excluded from engineering values
and stored in `bus_message_errors` with the raw record hex.

## 1553 hex capture (`.hex`)

First line `# SYN1553 HEX v1 aircraft=<id> flight=<id> start_us=<µs> records=<n>`, then the same 100-byte records as
hex text (line breaks ignored).
