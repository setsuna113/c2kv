"""Explicit representation injection at the initial-allocation factory seam."""


def instantiate_initial(policy_type, tokenizer, *, initial_allocator_factory=None, **kwargs):
    factory = initial_allocator_factory or policy_type
    if initial_allocator_factory is None:
        return factory(tokenizer, **kwargs)
    return factory(policy_type, tokenizer, **kwargs)
