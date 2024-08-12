import comfy.samplers
from nodes import KSamplerAdvanced, VAEDecode, VAEEncode

class FlipFlopperSameArch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model1": ("MODEL",),
                    "model2": ("MODEL",),
                    "vae1": ("VAE",),
                    "vae2": ("VAE",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg1": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg2": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name1": (comfy.samplers.KSampler.SAMPLERS, ),
                    "sampler_name2": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler1": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "scheduler2": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive1": ("CONDITIONING", ),
                    "negative1": ("CONDITIONING", ),
                    "positive2": ("CONDITIONING", ),
                    "negative2": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "chunks": ("INT", {"default": 1, "min": 1, "max": 1000}),
                    "invert": (["false", "true"], ),
                     }
                }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("LATENT", "FINAL_VAE")
    FUNCTION = "sample"

    CATEGORY = "calbenodes"

    def sample(self, model1, model2, vae1, vae2, add_noise, noise_seed, steps, cfg1, cfg2, sampler_name1, sampler_name2, scheduler1, scheduler2, positive1, negative1, positive2, negative2, latent_image, denoise, chunks, invert):
        print(f"Initial latent shape: {latent_image['samples'].shape}")
        print(f"Initial latent stats: min={latent_image['samples'].min().item():.4f}, max={latent_image['samples'].max().item():.4f}, mean={latent_image['samples'].mean().item():.4f}")

        ksampler = KSamplerAdvanced()
        vae_decode = VAEDecode()
        vae_encode = VAEEncode()
        current_latent = latent_image

        # Apply invert if true
        if invert == "true":
            model1, model2 = model2, model1
            vae1, vae2 = vae2, vae1
            positive1, positive2 = positive2, positive1
            negative1, negative2 = negative2, negative1
            cfg1, cfg2 = cfg2, cfg1
            sampler_name1, sampler_name2 = sampler_name2, sampler_name1
            scheduler1, scheduler2 = scheduler2, scheduler1

        chunk_size = chunks
        num_iterations = steps // chunk_size
        if steps % chunk_size != 0:
            num_iterations += 1  # Add one more iteration to handle remaining steps

        for iteration in range(num_iterations):
            current_model = model1 if iteration % 2 == 0 else model2
            current_vae = vae1 if iteration % 2 == 0 else vae2
            positive = positive1 if iteration % 2 == 0 else positive2
            negative = negative1 if iteration % 2 == 0 else negative2
            cfg = cfg1 if iteration % 2 == 0 else cfg2
            sampler_name = sampler_name1 if iteration % 2 == 0 else sampler_name2
            scheduler = scheduler1 if iteration % 2 == 0 else scheduler2

            start_step = iteration * chunk_size
            end_step = min((iteration + 1) * chunk_size, steps)  # Ensure we don't exceed total steps

            print(f"\nIteration {iteration + 1}/{num_iterations}: Using model {'1' if iteration % 2 == 0 else '2'}")
            print(f"Current model id: {id(current_model)}")
            print(f"Sampler: {sampler_name}, Scheduler: {scheduler}, CFG: {cfg}")
            print(f"Processing steps {start_step} to {end_step}")

            add = "enable" if iteration == 0 else "disable"
            ret = "disable" if iteration == num_iterations - 1 else "enable"

            # log everything we're about to put into sample
            print(f"add_noise: {add}")
            print(f"noise_seed: {noise_seed}")
            print(f"steps: {steps}")
            print(f"cfg: {cfg}")
            print(f"sampler_name: {sampler_name}")
            print(f"scheduler: {scheduler}")
            print(f"start_at_step: {start_step}")
            print(f"end_at_step: {end_step}")
            print(f"return_with_leftover_noise: {ret}")

            current_latent = ksampler.sample(model=current_model, 
                                            add_noise=add, 
                                            noise_seed=noise_seed, 
                                            steps=steps, 
                                            cfg=cfg, 
                                            sampler_name=sampler_name, 
                                            scheduler=scheduler, 
                                            positive=positive, 
                                            negative=negative, 
                                            latent_image=current_latent, 
                                            start_at_step=start_step,
                                            end_at_step=end_step,
                                            return_with_leftover_noise=ret)[0]

            print(f"Latent stats after iteration {iteration + 1}: min={current_latent['samples'].min().item():.4f}, max={current_latent['samples'].max().item():.4f}, mean={current_latent['samples'].mean().item():.4f}")

            # Switch VAE at chunk boundaries if needed
            if iteration < num_iterations - 1:
                next_vae = vae2 if iteration % 2 == 0 else vae1
                # image = vae_decode.decode(current_vae, current_latent)[0]
                # current_latent = vae_encode.encode(next_vae, image)[0]
                print(f"VAE switch after iteration {iteration + 1}")
                print(f"Latent stats after VAE switch: min={current_latent['samples'].min().item():.4f}, max={current_latent['samples'].max().item():.4f}, mean={current_latent['samples'].mean().item():.4f}")

        # Determine the final VAE based on the total number of iterations
        final_vae = vae1 if (num_iterations - 1) % 2 == 0 else vae2

        print(f"\nFinal latent stats: min={current_latent['samples'].min().item():.4f}, max={current_latent['samples'].max().item():.4f}, mean={current_latent['samples'].mean().item():.4f}")

        return (current_latent, final_vae)