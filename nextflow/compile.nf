nextflow.enable.dsl = 2

params.alfs_data_dir   = "${launchDir}/../alfs_data"
params.text_data_dir   = "${launchDir}/../text_data"
params.viewer_data_dir = "${launchDir}/../viewer_data"

process COMPILE {
    publishDir params.viewer_data_dir, mode: 'copy'
    input:  tuple path("alfs.json"), path("labeled.parquet"), path("docs.parquet")
    output: path "data.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.viewer.compile \
        --alfs alfs.json --labeled labeled.parquet --docs docs.parquet \
        --output data.json
    """
}

workflow {
    alfs    = file("${params.alfs_data_dir}/alfs.json")
    labeled = file("${params.alfs_data_dir}/labeled.parquet")
    docs    = file("${params.text_data_dir}/latest/docs.parquet")
    COMPILE(Channel.value([alfs, labeled, docs]))
}
