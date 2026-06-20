# Next-open execution note

The corrected next-open execution generator is implemented in:

- `scripts/generate_fullrisk_grid_300_next_open.py`

It fixes the previous optimistic close-to-close execution assumption. The intended execution model is:

- signal is generated after T close;
- buy executes at T+1 open;
- sell executes at T+1 open;
- switch executes at T+1 open;
- an existing holding still carries the T close to T+1 open overnight return before a sell or switch;
- a new holding carries only the T+1 open to T+1 close intraday return on the execution day;
- ETF transaction cost remains intentionally excluded.

Important: the existing production workflow still calls `scripts/generate_fullrisk_grid_300.py` until that command is switched to the next-open generator. The attempted workflow update was blocked by the tool safety layer, so the next required one-line change is:

```yaml
python scripts/generate_fullrisk_grid_300_next_open.py --promote
```

instead of:

```yaml
python scripts/generate_fullrisk_grid_300.py --promote
```
