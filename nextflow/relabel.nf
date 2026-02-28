nextflow.enable.dsl = 2

params.model           = "llama3.1:8b"
params.context_chars   = 150
params.max_occurrences = 100  // TODO: artificially low for dev; raise once pipeline is stable
params.seg_data_dir    = "${launchDir}/../seg_data"
params.text_data_dir   = "${launchDir}/../text_data"
params.senses_db       = "${launchDir}/../alfs_data/senses.db"
params.labeled_db      = "${launchDir}/../alfs_data/labeled.db"

process GENERATE_TARGETS {
    output: path "targets/*.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.generate_relabel_targets \
        --senses-db ${params.senses_db} --output-dir targets/ --labeled-db ${params.labeled_db}
    """
}

process LABEL_OCCURRENCES {
    input:  tuple path("target.json"), path("by_prefix"), path("docs.parquet")
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

    GENERATE_TARGETS()
    targets_ch = GENERATE_TARGETS.out.flatten()

    LABEL_OCCURRENCES(
        targets_ch
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
    )
}
