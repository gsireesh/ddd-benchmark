from prompts.prompts import PromptBuilder, prompt_builders


def get_prompt_builders(model_name: str, dataset: str, modality: str, prompt_name_filter: str):
    selected_builders = [
        builder
        for builder in prompt_builders
        if any([model_name.startswith(m) for m in builder.model_prefix_filters])
        and dataset in builder.intended_datasets
        and modality in builder.intended_modalities
        and prompt_name_filter in builder.id
    ]

    return selected_builders


""
