nextflow.enable.dsl = 2

params.num_docs = 10
params.seed     = 42
params.out_date = new Date().format('yyyy-MM-dd')
params.out_dir  = "data/${params.out_date}"

process FETCH_TITLES {
    output: path "titles.txt"
    script:
    """
    uv run --project ${projectDir} python -m alfadict.etl.fetch_titles \
        --num-docs ${params.num_docs} \
        --seed     ${params.seed} \
        --output   titles.txt
    """
}

process DOWNLOAD_DOCS {
    publishDir params.out_dir, mode: 'copy'
    input:  path "titles.txt"
    output: path "docs.parquet"
    script:
    """
    uv run --project ${projectDir} python -m alfadict.etl.download \
        --titles titles.txt \
        --output docs.parquet
    """
}

workflow {
    FETCH_TITLES()
    DOWNLOAD_DOCS(FETCH_TITLES.out)
}
