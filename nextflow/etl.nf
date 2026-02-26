nextflow.enable.dsl = 2

params.num_docs      = 10
params.seed          = 42
params.out_date      = new Date().format('yyyy-MM-dd')
params.text_data_dir = "${launchDir}/../text_data"
params.out_dir       = "${params.text_data_dir}/${params.out_date}"
params.cache_dir     = "${params.text_data_dir}/cache"
params.dump_url      = "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"

process DOWNLOAD_DUMP {
    storeDir params.cache_dir
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
    uv run --project ${launchDir} python -m alfs.etl.parse_dump \
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

workflow.onComplete {
    if (workflow.success) {
        def latestLink = file("${params.text_data_dir}/latest")
        latestLink.delete()
        file(params.out_dir).mklink(latestLink)
    }
}
