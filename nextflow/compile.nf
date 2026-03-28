nextflow.enable.dsl = 2

params.senses_db           = "${launchDir}/../alfs_data/senses.db"
params.labeled_db          = "${launchDir}/../alfs_data/labeled.db"
params.text_data_dir       = "${launchDir}/../text_data"
params.seg_data_dir        = "${launchDir}/../seg_data"
params.viewer_data_dir     = "${launchDir}/../viewer_data"
params.num_compile_batches = 8

process COMPILE_CORPUS_COUNTS {
    output: path "corpus_counts.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile_corpus_counts \
        --senses-db ${params.senses_db} \
        --by-prefix-dir ${params.seg_data_dir}/by_prefix \
        --output corpus_counts.json
    """
}

process COMPILE_BATCH {
    input:
        val batch_idx
        path("docs.parquet")
        path("corpus_counts.json")
    output: path "entries_${batch_idx}.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile \
        --senses-db ${params.senses_db} --labeled-db ${params.labeled_db} \
        --docs docs.parquet \
        --corpus-counts corpus_counts.json \
        --batch-idx ${batch_idx} --num-batches ${params.num_compile_batches} \
        --output entries_${batch_idx}.json
    """
}

process COMPILE_MERGE {
    publishDir params.viewer_data_dir, mode: 'copy'
    input:
        path "entries_*.json"
        path "corpus_counts.json"
    output: path "data.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile_merge \
        --inputs entries_*.json \
        --corpus-counts corpus_counts.json \
        --output data.json
    """
}

process COMPILE_QC_STATS {
    publishDir params.viewer_data_dir, mode: 'copy'
    output: path "qc_stats.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile_qc \
        --mode stats --labeled-db ${params.labeled_db} --output qc_stats.json
    """
}

process COMPILE_QC_INSTANCES {
    publishDir params.viewer_data_dir, mode: 'copy'
    input:
        val rating
        path("docs.parquet")
    output: path "qc_${rating}.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile_qc \
        --mode instances --labeled-db ${params.labeled_db} \
        --senses-db ${params.senses_db} \
        --docs docs.parquet --rating ${rating} --output qc_${rating}.json
    """
}

workflow {
    docs_ch = Channel.value(file("${params.text_data_dir}/docs.parquet"))

    corpus_counts_ch = COMPILE_CORPUS_COUNTS()

    entries_ch = COMPILE_BATCH(
        Channel.of(0..<params.num_compile_batches),
        docs_ch,
        corpus_counts_ch,
    )
    COMPILE_MERGE(entries_ch.collect(), corpus_counts_ch)

    COMPILE_QC_STATS()
    COMPILE_QC_INSTANCES(Channel.of(0, 1), docs_ch)
}
