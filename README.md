# What is LEAP?

[Demo video](https://www.youtube.com/watch?v=iv_P6db88ts)

It's a research project where input images are being converted to a local minimum in latent space. Then, we feed the weights to Stable Diffusion's Textual Inversion model.

The benefits are huge, training takes easily under 5 minutes, with little quality difference from training for hours on the same hardware.

It is scalable enough to offer in a low-scale Discord bot like Thingy, where our goal is to introduce people to AI without it costing hundreds of dollars per month in GPU rent!

Love you all! Sorry for the fact this README is a little crunchy. It is because I'm so excited and jumpy, never thought it would work.

# How to use with Stable Diffusion

**Note** The author is used to Linux, while Windows should work, the author can't guarantee working README instructions.

- Run the following command: `pip install git+https://github.com/peterwilli/sd-leap-booster.git` (Installing the package automatically gets you the LEAP weights)
- Run `leap_textual_inversion` and set the parameters to what you wish (they are similar to the [official textual inversion script](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py))
- An example: `leap_textual_inversion --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base --placeholder_token="<peter>" --train_data_dir=path/to/images --learning_rate=0.001`

# Support, sponsorship and thanks

Are you looking to make a positive impact and get some awesome perks in the process? **[Join me on Patreon!](https://www.patreon.com/emerald_show)** For just $3 per month, you can join our Patreon community and help a creative mind in the Netherlands bring their ideas to life.

Not only will you get the satisfaction of supporting an individual's passions, but you'll also receive a 50% discount on any paid services that result from the projects you sponsor. Plus, as a Patreon member, you'll have exclusive voting rights on new features and the opportunity to shape the direction of future projects. Don't miss out on this chance to make a difference and get some amazing benefits in return.

One of the things we intend on doing, is trying to make LEAP with Lora!

**Special thanks to:**

 - LAION/Stability AI for providing training GPU's ~~And hopefully I get confident enough to soon try them.~~
 - Jina.ai for giving the computation power to run my bot!
 - You?