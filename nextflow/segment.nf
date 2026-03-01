nextflow.enable.dsl = 2

params.seg_data_dir  = "${launchDir}/../seg_data"
params.text_data_dir = "${launchDir}/../text_data"
params.docs          = "${params.text_data_dir}/latest/docs.parquet"
params.out_date      = new Date().format('yyyy-MM-dd')
params.out_dir       = "${params.seg_data_dir}/${params.out_date}"
params.num_shards    = 4

process SEGMENT_DOCS_SHARD {
    input:  tuple path("docs.parquet"), val(shard_idx)
    output: path "occurrences_${shard_idx}.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.seg.segment_docs \
        --docs        docs.parquet \
        --shard-index ${shard_idx} \
        --num-shards  ${params.num_shards} \
        --output      occurrences_${shard_idx}.parquet
    """
}

process AGGREGATE_OCCURRENCES {
    publishDir "${params.out_dir}/by_prefix", mode: 'copy'
    input:  path "*.parquet"
    output: path "*"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.seg.aggregate_occurrences \
        --occurrences *.parquet \
        --output-dir  .
    """
}

workflow {
    docs_ch       = Channel.fromPath(params.docs)
    shard_indices = Channel.from(0..<params.num_shards)

    shards = SEGMENT_DOCS_SHARD(docs_ch.combine(shard_indices))
    AGGREGATE_OCCURRENCES(shards.collect())
}

workflow.onComplete {
    if (workflow.success) {
        def latestLink = file("${params.seg_data_dir}/latest")
        latestLink.delete()
        file(params.out_dir).mklink(latestLink)
    }
}
