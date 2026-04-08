nextflow.enable.dsl = 2

params.senses_db     = "${launchDir}/../alfs_data/senses.db"
params.labeled_db    = "${launchDir}/../alfs_data/labeled.db"
params.seg_data_dir  = "${launchDir}/../seg_data"
params.text_data_dir = "${launchDir}/../text_data"
params.plots_dir     = "${launchDir}/plots"

process CORPUS_COUNTS {
    output:
        path "corpus_counts.json", emit: counts
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile_corpus_counts \
        --senses-db ${params.senses_db} \
        --by-prefix-dir ${params.seg_data_dir}/by_prefix \
        --output corpus_counts.json
    """
}

process PLOT_NSENSES_VS_FREQ {
    publishDir params.plots_dir, mode: 'copy'
    input:
        path "corpus_counts.json"
    output:
        path "nsenses_vs_freq.png"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.plots.nsenses_vs_freq \
        --senses-db ${params.senses_db} \
        --corpus-counts corpus_counts.json \
        --output nsenses_vs_freq.png
    """
}

process PLOT_SOURCE_SENSE_DIVERGENCE {
    publishDir params.plots_dir, mode: 'copy'
    output:
        path "source_sense_divergence.png"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.plots.source_sense_divergence \
        --senses-db ${params.senses_db} \
        --labeled-db ${params.labeled_db} \
        --docs ${params.text_data_dir}/docs.parquet \
        --output source_sense_divergence.png
    """
}

workflow {
    corpus_counts_ch = CORPUS_COUNTS().counts
    PLOT_NSENSES_VS_FREQ(corpus_counts_ch)
    PLOT_SOURCE_SENSE_DIVERGENCE()
}
