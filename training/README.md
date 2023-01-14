# Training LEAP

If you want to use LEAP on a different Stable Diffusion version, or simply learn how it works, then you find the instructions below very useful.

**Authors notes**: I'm using Linux (NixOS), while it could work on Windows, I'm not familiar with this operating system. Feel free to reach out for help. Commands are `written like this!`.

## Dataset creation

LEAP uses a synthetic dataset, we extract all words from Stable Diffusion and generate "samples" that allow us to associate its images with the weights used to make them.

- Clone this repository and `cd` to it.
- Install leap_sd: `pip install -e .`
- run `python training/dataset_creator/sd_extractor.py`
    - For using a custom Stable Diffusion, pass `--pretrained_model_name_or_path`, for example: `--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5`

## Train! 🐲

- `cd` to the training directory of this repo
- Check the size of your latent space! (for sd < 2.0 it's 768 and for >= 2.0 it's 1024)
- Run the following training command: `python training/train.py --batch_size=10 --gpus=1 --max_epochs=250 --latent_dim_size=1024`

### Examples

- Training for SD 1.5
    
    ```bash
    python training/dataset_creator/sd_extractor.py --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5
    python training/train.py --batch_size=10 --gpus=1 --max_epochs=250 --latent_dim_size=768
    ``` 

# Support, sponsorship and thanks

Are you looking to make a positive impact and get some awesome perks in the process? **[Join me on Patreon!](https://www.patreon.com/emerald_show)** For just $3 per month, you can join our Patreon community and help a creative mind in the Netherlands bring their ideas to life.

Not only will you get the satisfaction of supporting an individual's passions, but you'll also receive a 50% discount on any paid services that result from the projects you sponsor. Plus, as a Patreon member, you'll have exclusive voting rights on new features and the opportunity to shape the direction of future projects. Don't miss out on this chance to make a difference and get some amazing benefits in return.

One of the things we intend on doing, is trying to make LEAP with Lora!