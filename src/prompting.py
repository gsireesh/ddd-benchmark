def build_prompt(model, modal, author, properties, prompt_db, context=None):
    if context is None:
        context = ""
    template = prompt_db[(model, modal, author)]
    recipe_format = "{"
    for p in properties:
        recipe_format += f"\"{p}\": <value>, "
    recipe_format = recipe_format[:-2]
    recipe_format += "}"
    reagents = "Si, Ge, Al, OH, H20"
    makers = "OpenAI" if 'gpt' in model else "Anthropic"
    recipe_type = "zeolite synthesis"
    prompt = template.format(
        recipe_format=recipe_format,
        model_makers=makers,
        reagents=reagents,
        recipe_type=recipe_type,
        properties=" ".join(properties),
        context=context,
    )
    return prompt