nextflow.enable.dsl = 2

params.docs         = "${launchDir}/../text_data/latest/docs.parquet"
params.seg_data_dir = "${launchDir}/../seg_data/by_prefix"
params.shard_size   = 1000

process SPLIT_DOCS {
    output: path "shard_*.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.seg.split_docs \
        --docs ${params.docs} \
        --seg-data-dir ${params.seg_data_dir} \
        --shard-size ${params.shard_size}
    """
}

process SEGMENT_SHARD {
    input:  path shard_file
    output: path "raw_${shard_file.baseName}.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.seg.segment_docs \
        --docs ${shard_file} \
        --output raw_${shard_file.baseName}.parquet
    """
}

process AGGREGATE {
    input: path "raw_*.parquet"
    script:
    """
    files=(raw_*.parquet)
    uv run --project ${launchDir} --no-sync python -m alfs.seg.aggregate_occurrences \
        --occurrences "\${files[@]}" \
        --output-dir ${params.seg_data_dir} \
        --merge
    """
}

workflow {
    shards_ch = SPLIT_DOCS().flatten()
    raw_ch    = SEGMENT_SHARD(shards_ch)
    AGGREGATE(raw_ch.collect())
}
