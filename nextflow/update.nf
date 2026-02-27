nextflow.enable.dsl = 2

params.top_n           = 10
params.model           = "llama3.1:8b"
params.context_chars   = 150
params.max_samples     = 20
params.seg_data_dir    = "${launchDir}/../seg_data"
params.text_data_dir   = "${launchDir}/../text_data"
params.alfs_data_dir   = "${launchDir}/../alfs_data"
params.update_data_dir = "${launchDir}/../update_data"
params.out_date        = new Date().format('yyyy-MM-dd')
params.out_dir         = "${launchDir}/../update_data/${params.out_date}"

process SELECT_TARGETS {
    input:  tuple path("by_prefix"), path("update_data")
    output: path "targets/*.json"
    script:
    """
    uv run --project ${launchDir} python -m alfs.update.select_targets \
        --seg-data-dir by_prefix --top-n ${params.top_n} \
        --labeled-dir update_data --output-dir targets/
    """
}

process INDUCE_SENSES {
    input:  tuple path("target.json"), path("by_prefix"), path("docs.parquet")
    output: path "*_senses.json"
    script:
    """
    form=\$(python -c "import json, urllib.parse; print(urllib.parse.quote(json.load(open('target.json'))['form'], safe=''))")
    uv run --project ${launchDir} python -m alfs.update.induce_senses \
        --target target.json --seg-data-dir by_prefix --docs docs.parquet \
        --output \${form}_senses.json --model ${params.model} \
        --context-chars ${params.context_chars} --max-samples ${params.max_samples}
    """
}

process UPDATE_INVENTORY {
    publishDir params.alfs_data_dir, mode: 'copy'
    input:  tuple path("senses/"), path("alfs.json")
    output: path "alfs.json"
    script:
    """
    uv run --project ${launchDir} python -m alfs.update.update_inventory \
        --alfs-data alfs.json --senses-dir senses/ --output alfs.json
    """
}

process LABEL_OCCURRENCES {
    publishDir params.out_dir, mode: 'copy'
    input:  tuple path("target.json"), path("alfs.json"), path("by_prefix"), path("docs.parquet"), path("update_data")
    output: path "*_labeled.parquet"
    script:
    """
    form=\$(python -c "import json, urllib.parse; print(urllib.parse.quote(json.load(open('target.json'))['form'], safe=''))")
    uv run --project ${launchDir} python -m alfs.update.label_occurrences \
        --target target.json --seg-data-dir by_prefix --docs docs.parquet \
        --alfs alfs.json --output \${form}_labeled.parquet \
        --labeled-dir update_data \
        --model ${params.model} --context-chars ${params.context_chars}
    """
}

workflow {
    seg_dir     = file("${params.seg_data_dir}/latest/by_prefix")
    docs        = file("${params.text_data_dir}/latest/docs.parquet")
    alfs        = file("${params.alfs_data_dir}/alfs.json")
    update_data = file("${params.update_data_dir}")

    SELECT_TARGETS(Channel.value([seg_dir, update_data]))
    targets_ch = SELECT_TARGETS.out.flatten()

    INDUCE_SENSES(
        targets_ch
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
    )

    UPDATE_INVENTORY(
        INDUCE_SENSES.out.collect().map { senses -> [senses, alfs] }
    )

    LABEL_OCCURRENCES(
        targets_ch
            .combine(UPDATE_INVENTORY.out)
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
            .combine(Channel.value(update_data))
    )
}
