# Verl
For more information about verl itself please read the official docs: https://verl.readthedocs.io/en/latest/index.html

# Preparing Dependencies
This project uses uv to manage dependencies. Installing extras before regular install may fail due to flash-attn requiring pytorch. Versioning may be finicky with the newest version of flash-attn.
```bash
uv sync
uv sync --no-build-isolation --all-extras
```

# Preparing Data
Generate the data with `process_data.py`. Actual code generating each dataset lives in data_gen

Usage:
```bash
uv run python process_data.py
```

### Explanations
`pir_grpo.parquet` comes from https://huggingface.co/datasets/jonluj/pir_full with extra formatting to adhere to Verl dataset format.

`pir_data_xml.parquet` comes from PIR but the content uses only user prompts with xml delimiters. This form the control prompt we use to compare with others.

`pir_grpo_oocr.parquet` comes from PIR but the main content is reformatted to utilize policies the model may have already trained with.

# Running Experiments
`./experiments` contains the bash files used to run experiements. They are loosely named by the model size, previous checkpoint, and dataset used.