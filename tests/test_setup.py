def test_imports():
    import anthropic  # noqa: F401
    import chromadb  # noqa: F401
    import pypdf  # noqa: F401

    from citerag import config

    assert config.TOP_K > 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
