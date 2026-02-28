nextflow.enable.dsl = 2

params.model           = "llama3.1:8b"
params.context_chars   = 150
params.max_occurrences = 100  // TODO: artificially low for dev; raise once pipeline is stable
params.seg_data_dir    = "${launchDir}/../seg_data"
params.text_data_dir   = "${launchDir}/../text_data"
params.alfs_data_dir   = "${launchDir}/../alfs_data"
params.out_date        = new Date().format('yyyy-MM-dd')
params.out_dir         = "${launchDir}/../update_data/${params.out_date}"

process GENERATE_TARGETS {
    input:  path "alfs.json"
    output: path "targets/*.json"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.generate_relabel_targets \
        --alfs alfs.json --output-dir targets/
    """
}

process LABEL_OCCURRENCES {
    input:  tuple path("target.json"), path("alfs.json"), path("by_prefix"), path("docs.parquet")
    output: path "*_labeled.parquet"
    script:
    """
    form=\$(python -c "import json, urllib.parse; print(urllib.parse.quote(json.load(open('target.json'))['form'], safe=''))")
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.label_occurrences \
        --target target.json --seg-data-dir by_prefix --docs docs.parquet \
        --alfs alfs.json --output \${form}_labeled.parquet \
        --model ${params.model} --context-chars ${params.context_chars} \
        --max-occurrences ${params.max_occurrences}
    """
}

process UPDATE_LABELS {
    publishDir params.alfs_data_dir, mode: 'copy'
    input:  path "new_labeled/"
    output: path "labeled.parquet"
    script:
    """
    uv run --project ${launchDir} --no-sync python -m alfs.update.labeling.update_labels \
        --labeled-data labeled_empty.parquet --new-dir new_labeled/ \
        --output labeled.parquet
    """
}

workflow {
    seg_dir = file("${params.seg_data_dir}/latest/by_prefix")
    docs    = file("${params.text_data_dir}/latest/docs.parquet")
    alfs    = file("${params.alfs_data_dir}/alfs.json")

    GENERATE_TARGETS(alfs)
    targets_ch = GENERATE_TARGETS.out.flatten()

    LABEL_OCCURRENCES(
        targets_ch
            .combine(Channel.value(alfs))
            .combine(Channel.value(seg_dir))
            .combine(Channel.value(docs))
    )

    UPDATE_LABELS(
        LABEL_OCCURRENCES.out.collect()
    )
}
