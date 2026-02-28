nextflow.enable.dsl = 2

params.top_n           = 10
params.model           = "gemma2:9b"
params.context_chars   = 150
params.max_samples     = 20
params.max_occurrences = 15
params.seg_data_dir    = "${launchDir}/../seg_data"
params.text_data_dir   = "${launchDir}/../text_data"
params.senses_db       = "${launchDir}/../alfs_data/senses.db"
params.labeled_db      = "${launchDir}/../alfs_data/labeled.db"

process SELECT_TARGETS {
    input:  path("by_prefix")
    output: path "targets/*.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.select_targets \
        --seg-data-dir by_prefix --top-n ${params.top_n} \
        --senses-db ${params.senses_db} --labeled-db ${params.labeled_db} \
        --output-dir targets/
    """
}

process INDUCE_SENSES {
    input:  tuple path("target.json"), path("by_prefix"), path("docs.parquet")
    output: path "*_senses.json"
    script:
    """
    form=\$(python -c "import json, urllib.parse; print(urllib.parse.quote(json.load(open('target.json'))['form'], safe=''))")
    uv run --project ${launchDir} --no-sync python -m alfs.update.induction.induce_senses \
        --target target.json --seg-data-dir by_prefix --docs docs.parquet \
        --output \${form}_senses.json --model ${params.model} \
        --context-chars ${params.context_chars} --max-samples ${params.max_samples} \
        --senses-db ${params.senses_db} --labeled-db ${params.labeled_db}
    """
}

process UPDATE_INVENTORY {
    input:  path("senses_file.json")
    output: val "done"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.induction.update_inventory \
        --senses-file senses_file.json --senses-db ${params.senses_db}
    """
}

process LABEL_OCCURRENCES {
    input:  tuple path("target.json"), path("by_prefix"), path("docs.parquet"), val(_done)
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.label_occurrences \
        --target target.json --seg-data-dir by_prefix --docs docs.parquet \
        --senses-db ${params.senses_db} --labeled-db ${params.labeled_db} \
        --model ${params.model} --context-chars ${params.context_chars} \
        --max-occurrences ${params.max_occurrences}
    """
}

workflow {
    seg_dir = file("${params.seg_data_dir}/latest/by_prefix")
    docs    = file("${params.text_data_dir}/latest/docs.parquet")

    SELECT_TARGETS(Channel.value(seg_dir))
    targets_ch = SELECT_TARGETS.out.flatten()

    INDUCE_SENSES(
        targets_ch
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
    )

    UPDATE_INVENTORY(INDUCE_SENSES.out)

    // Collect all UPDATE_INVENTORY done signals before starting labeling
    // so senses.db is fully populated
    all_done = UPDATE_INVENTORY.out.collect()

    LABEL_OCCURRENCES(
        targets_ch
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
            .combine(all_done)
    )
}
