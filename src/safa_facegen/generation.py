"""Convenience sampling without changing the differentiable noise API."""
def generate_images(generator, num_images=1, seed=None):
    import torch
    if type(num_images) is not int or num_images < 1:
        raise ValueError('num_images must be a positive integer')
    parameter = next(generator.parameters())
    rng = torch.Generator(device=parameter.device)
    if seed is None:
        rng.seed()
    elif type(seed) is int:
        rng.manual_seed(seed)
    else:
        raise TypeError('seed must be an integer or None')
    shape = (num_images, *generator.noise_shape)
    noise = torch.randn(shape, device=parameter.device, dtype=torch.float32, generator=rng)
    extra = [torch.randn(shape, device=parameter.device, dtype=torch.float32, generator=rng)
             for _ in range(generator.step_noise_count)]
    return generator.sample(noise, step_noises=extra, grad_enabled=False)
