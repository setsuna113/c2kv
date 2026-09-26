"""Explicit representation injection at the initial-allocation factory seam."""


def instantiate_initial(policy_type, tokenizer, *, initial_allocator_factory=None, **kwargs):
    factory = initial_allocator_factory or policy_type
    if initial_allocator_factory is None:
        return factory(tokenizer, **kwargs)
    return factory(policy_type, tokenizer, **kwargs)


def instantiate_composed_initial(composer, tokenizer, *, initial_allocator_factory=None, **kwargs):
    """Construct one policy root with explicitly constructed representation branches."""
    if initial_allocator_factory is None:
        return composer(tokenizer, **kwargs)
    return initial_allocator_factory.compose(composer, tokenizer, **kwargs)
