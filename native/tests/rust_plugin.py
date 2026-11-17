"""Run the existing suite with the sampler replaced, in this test process only."""


def pytest_configure(config):
    import ray360.cubemap
    import ray360.panorama
    import ray360.remap
    from ray360_native import sample

    for module in (ray360.remap, ray360.panorama, ray360.cubemap):
        module.sample = sample
