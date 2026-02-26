nextflow.enable.dsl = 2

params.seg_data_dir  = "${projectDir}/../seg_data"
params.text_data_dir = "${projectDir}/../text_data"
params.docs          = "${params.text_data_dir}/latest/docs.parquet"
params.out_date      = new Date().format('yyyy-MM-dd')
params.out_dir       = "${params.seg_data_dir}/${params.out_date}"

process SEGMENT_DOCS {
    publishDir params.out_dir, mode: 'copy'
    input:  path "docs.parquet"
    output: path "raw_occurrences.parquet"
    script:
    """
    uv run --project ${projectDir} python -m alfs.etl.segment_docs \
        --docs   docs.parquet \
        --output raw_occurrences.parquet
    """
}

process AGGREGATE_OCCURRENCES {
    publishDir "${params.out_dir}/by_prefix", mode: 'copy'
    input:  path "raw_occurrences.parquet"
    output: path "*"
    script:
    """
    uv run --project ${projectDir} python -m alfs.etl.aggregate_occurrences \
        --occurrences raw_occurrences.parquet \
        --output-dir  .
    """
}

workflow {
    docs_ch = Channel.fromPath(params.docs)
    SEGMENT_DOCS(docs_ch)
    AGGREGATE_OCCURRENCES(SEGMENT_DOCS.out)
}
