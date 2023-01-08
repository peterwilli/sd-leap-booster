# What is LEAP?

It's a research project where input images are being converted to a local minimum in latent space. Then, we feed the weights to Stable Diffusion's Textual Inversion model.

The benefits are huge, training takes easily under 5 minutes, with little quality difference from training for hours on the same hardware.

It is scalable enough to offer in a low-scale Discord bot like Thingy, where our goal is to introduce people to AI without it costing hundreds of dollars per month in GPU rent!

Love you all! Sorry for the fact this README is a little crunchy. It is because I'm so excited and jumpy, never thought it would work.

# How to use with Stable Diffusion

**Note** The author is used to Linux, while Windows should work, the author can't guarantee working README instructions.

- Run the following command: `pip install git+https://github.com/peterwilli/sd-leap-booster.git` (Installing the package automatically gets you the LEAP weights)
- Run `leap_textual_inversion` and set the parameters to what you wish (they are similar to the [official textual inversion script](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py))
- An example: `leap_textual_inversion --placeholder_token="<peter>" --train_data_dir=path/to/images --learning_rate=0.001`
