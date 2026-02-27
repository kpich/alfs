nextflow.enable.dsl = 2

params.num_docs           = 1000
params.seed               = 42
params.out_date           = new Date().format('yyyy-MM-dd')
params.text_data_dir      = "${launchDir}/../text_data"
params.out_dir            = "${params.text_data_dir}/${params.out_date}"
params.cache_dir          = "${params.text_data_dir}/cache"
params.wikibooks_dump_url = "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"
params.wikisource_dump_url = "https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2"

process DOWNLOAD_WIKIBOOKS_DUMP {
    storeDir params.cache_dir
    output: path "enwikibooks-latest-pages-articles.xml.bz2"
    script:
    """
    wget -q --show-progress -O enwikibooks-latest-pages-articles.xml.bz2 ${params.wikibooks_dump_url}
    """
}

process DOWNLOAD_WIKISOURCE_DUMP {
    storeDir params.cache_dir
    output: path "enwikisource-latest-pages-articles.xml.bz2"
    script:
    """
    wget -q --show-progress -O enwikisource-latest-pages-articles.xml.bz2 ${params.wikisource_dump_url}
    """
}

process PARSE_WIKIBOOKS_DUMP {
    input:  path "dump.xml.bz2"
    output: path "wikibooks.parquet"
    script:
    """
    uv run --project ${launchDir} python -m alfs.etl.parse_dump \
        --dump     dump.xml.bz2 \
        --num-docs ${(params.num_docs / 2).toInteger()} \
        --seed     ${params.seed} \
        --source   wikibooks \
        --output   wikibooks.parquet
    """
}

process PARSE_WIKISOURCE_DUMP {
    input:  path "dump.xml.bz2"
    output: path "wikisource.parquet"
    script:
    """
    uv run --project ${launchDir} python -m alfs.etl.parse_dump \
        --dump     dump.xml.bz2 \
        --num-docs ${(params.num_docs / 2).toInteger()} \
        --seed     ${params.seed} \
        --source   wikisource \
        --output   wikisource.parquet
    """
}

process MERGE_DOCS {
    publishDir params.out_dir, mode: 'copy'
    input:  tuple path("wikibooks.parquet"), path("wikisource.parquet")
    output: path "docs.parquet"
    script:
    """
    uv run --project ${launchDir} python -m alfs.etl.merge_docs \
        --inputs wikibooks.parquet wikisource.parquet \
        --output docs.parquet
    """
}

workflow {
    DOWNLOAD_WIKIBOOKS_DUMP()
    DOWNLOAD_WIKISOURCE_DUMP()
    PARSE_WIKIBOOKS_DUMP(DOWNLOAD_WIKIBOOKS_DUMP.out)
    PARSE_WIKISOURCE_DUMP(DOWNLOAD_WIKISOURCE_DUMP.out)
    MERGE_DOCS(PARSE_WIKIBOOKS_DUMP.out.combine(PARSE_WIKISOURCE_DUMP.out))
}

workflow.onComplete {
    if (workflow.success) {
        def latestLink = file("${params.text_data_dir}/latest")
        latestLink.delete()
        file(params.out_dir).mklink(latestLink)
    }
}
