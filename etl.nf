nextflow.enable.dsl = 2

params.num_docs  = 10
params.seed      = 42
params.out_date  = new Date().format('yyyy-MM-dd')
params.out_dir   = "data/${params.out_date}"
params.cache_dir = "data/cache"
params.dump_url  = "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"

process DOWNLOAD_DUMP {
    storeDir "${projectDir}/${params.cache_dir}"
    output: path "enwikibooks-latest-pages-articles.xml.bz2"
    script:
    """
    wget -q --show-progress -O enwikibooks-latest-pages-articles.xml.bz2 ${params.dump_url}
    """
}

process PARSE_DUMP {
    publishDir params.out_dir, mode: 'copy'
    input:  path "dump.xml.bz2"
    output: path "docs.parquet"
    script:
    """
    uv run --project ${projectDir} python -m alfadict.etl.parse_dump \
        --dump     dump.xml.bz2 \
        --num-docs ${params.num_docs} \
        --seed     ${params.seed} \
        --output   docs.parquet
    """
}

workflow {
    DOWNLOAD_DUMP()
    PARSE_DUMP(DOWNLOAD_DUMP.out)
}
