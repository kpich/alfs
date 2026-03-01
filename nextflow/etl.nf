nextflow.enable.dsl = 2

params.num_shards         = 4
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

process STREAM_WIKIBOOKS_DUMP {
    input:  path "dump.xml.bz2"
    output: path "wikibooks_pages.jsonl"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.etl.stream_dump \
        --dump   dump.xml.bz2 \
        --source wikibooks \
        --output wikibooks_pages.jsonl
    """
}

process STREAM_WIKISOURCE_DUMP {
    input:  path "dump.xml.bz2"
    output: path "wikisource_pages.jsonl"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.etl.stream_dump \
        --dump   dump.xml.bz2 \
        --source wikisource \
        --output wikisource_pages.jsonl
    """
}

process PARSE_WIKIBOOKS_SHARD {
    input:  tuple path("pages.jsonl"), val(shard_idx)
    output: path "wikibooks_${shard_idx}.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.etl.parse_dump \
        --pages       pages.jsonl \
        --source      wikibooks \
        --shard-index ${shard_idx} \
        --num-shards  ${params.num_shards} \
        --output      wikibooks_${shard_idx}.parquet
    """
}

process PARSE_WIKISOURCE_SHARD {
    input:  tuple path("pages.jsonl"), val(shard_idx)
    output: path "wikisource_${shard_idx}.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.etl.parse_dump \
        --pages       pages.jsonl \
        --source      wikisource \
        --shard-index ${shard_idx} \
        --num-shards  ${params.num_shards} \
        --output      wikisource_${shard_idx}.parquet
    """
}

process MERGE_DOCS {
    publishDir params.out_dir, mode: 'copy'
    input:  path "*.parquet"
    output: path "docs.parquet"
    script:
    """
    uv run --project ${launchDir} python -m alfs.etl.merge_docs \
        --inputs *.parquet \
        --output docs.parquet
    """
}

workflow {
    DOWNLOAD_WIKIBOOKS_DUMP()
    DOWNLOAD_WIKISOURCE_DUMP()

    shard_indices = Channel.from(0..<params.num_shards)

    wb_jsonl = STREAM_WIKIBOOKS_DUMP(DOWNLOAD_WIKIBOOKS_DUMP.out)
    ws_jsonl = STREAM_WIKISOURCE_DUMP(DOWNLOAD_WIKISOURCE_DUMP.out)

    wb_shards = PARSE_WIKIBOOKS_SHARD(wb_jsonl.combine(shard_indices))
    ws_shards = PARSE_WIKISOURCE_SHARD(ws_jsonl.combine(shard_indices))

    MERGE_DOCS(wb_shards.mix(ws_shards).collect())
}

workflow.onComplete {
    if (workflow.success) {
        def latestLink = file("${params.text_data_dir}/latest")
        latestLink.delete()
        file(params.out_dir).mklink(latestLink)
    }
}
