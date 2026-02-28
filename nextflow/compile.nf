nextflow.enable.dsl = 2

params.senses_db       = "${launchDir}/../alfs_data/senses.db"
params.labeled_db      = "${launchDir}/../alfs_data/labeled.db"
params.text_data_dir   = "${launchDir}/../text_data"
params.viewer_data_dir = "${launchDir}/../viewer_data"

process COMPILE {
    publishDir params.viewer_data_dir, mode: 'copy'
    input:  path("docs.parquet")
    output: path "data.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile \
        --senses-db ${params.senses_db} --labeled-db ${params.labeled_db} \
        --docs docs.parquet --output data.json
    """
}

workflow {
    docs = file("${params.text_data_dir}/latest/docs.parquet")
    COMPILE(Channel.value(docs))
}
