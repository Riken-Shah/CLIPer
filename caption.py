from transformers import BlipProcessor, BlipForConditionalGeneration
# from textblob import TextBlob
# import ray
from clip_interrogator import Config, Interrogator
import PIL


class ImageCaption:
    def __init__(self, clip_model, clip_pre_process, cache_dir="", device="cpu"):
        # config = Config(clip_model_name="ViT-L-14/openai", cache_path=cache_dir)
        config = Config( cache_path=cache_dir, device=device)
        config.clip_model = clip_model
        config.clip_preprocess = clip_pre_process
        config.apply_low_vram_defaults()
        #

        config.blip_num_beams = 64
        config.chunk_size = 2048
        config.flavor_intermediate_count = 1024
        # self.ci_vitl = Interrogator(config)

    def get_caption_and_tags(self, image):
        # conditional image captioning
        text = "a photo of"
        # inputs = self.processor(image, text, return_tensors="pt")
        # out = self.model.generate(**inputs)
        # caption = self.processor.decode(out[0], skip_special_tokens=True)
        # blob = TextBlob(caption)
        # print(blob.noun_phrases)
        # print(caption)
        # caption = self.ci_vitl.interrogate_fast(image)
        caption = ""
        return caption, caption.split(",")

# print(ImageCaption("./.cache").get_caption_and_tags(PIL.Image.open("/Volumes/SSD/Backup 16-12-2023/Document/fabric database/12_orginal.jpeg")))