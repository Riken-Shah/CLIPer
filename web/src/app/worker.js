// import {
//   pipeline,
//   env,
//   RawImage,
//   CLIPTextModelWithProjection,
//   CLIPVisionModelWithProjection,
//   AutoTokenizer,
//   AutoProcessor,
// } from "@xenova/transformers";
// import axios from "axios";
//
// // Skip local model check
// env.allowLocalModels = false;
//
// // Use the Singleton pattern to enable lazy construction of the pipeline.
// class PipelineSingleton {
//   static model_id = "Xenova/clip-vit-base-patch32";
//   static tokenizer = null;
//   static text_model = null;
//   static image_model = null;
//   static processor = null;
//
//   static async getInstance(progress_callback = null) {
//     if (this.tokenizer === null) {
//       this.tokenizer = AutoTokenizer.from_pretrained(this.model_id);
//     }
//
//     if (this.processor === null) {
//       this.processor = AutoProcessor.from_pretrained(this.model_id);
//     }
//
//     if (this.image_model === null) {
//       this.image_model = CLIPVisionModelWithProjection.from_pretrained(
//         this.model_id,
//         {
//           quantized: false,
//           progress_callback,
//         }
//       );
//     }
//
//     if (this.text_model === null) {
//       this.text_model = CLIPTextModelWithProjection.from_pretrained(
//         this.model_id,
//         {
//           quantized: false,
//           progress_callback,
//         }
//       );
//     }
//
//     return Promise.all([
//       this.tokenizer,
//       this.processor,
//       this.text_model,
//       this.image_model,
//     ]);
//   }
// }
//
// // Listen for messages from the main thread
// self.addEventListener("message", async (event) => {
//   // Retrieve the classification pipeline. When called for the first time,
//   // this will load the pipeline and save it for future use.
//   let [tokenizer, processor, text_model, image_model] =
//     await PipelineSingleton.getInstance((x) => {
//       // We also add a progress callback to the pipeline so that we can
//       // track model loading.
//       self.postMessage(x);
//     });
//
//   // Actually perform the classification
//   // let output = await classifier(event.data.text);
//   let query_embedding = null;
//   if (event.data.text) {
//     // Run tokenization
//     let text_inputs = tokenizer(event.data.text, {
//       padding: true,
//       truncation: true,
//     });
//
//     // Compute embeddings
//     const { text_embeds } = await text_model(text_inputs);
//     query_embedding = text_embeds.tolist()[0];
//     console.log("txt", query_embedding);
//   } else {
//     const image = await RawImage.read(event.data.image);
//     const image_inputs = await processor(image);
//     const { image_embeds } = await image_model(image_inputs);
//     query_embedding = image_embeds.tolist()[0];
//     console.log("image", query_embedding);
//   }
//
//   // Your API endpoint and key
//   const PUBLIC_ENDPOINT =
//     "https://in03-4e728d395bbf9d3.api.gcp-us-west1.zillizcloud.com";
//   const API_KEY =
//     "88fd45721ec798ee735283cc5aff7f8e74c30562111d0c5aec3150d27bdc7f826d4126df09ce46602954b80d2bb9fecc7339939e";
//
//
//
//   // Make the POST request
//   const response = await axios.post(
//     `${PUBLIC_ENDPOINT}/v1/vector/search`,
//     {
//       collectionName: "dev_collection",
//       limit: 20,
//       vector: query_embedding,
//       outputFields: ["id", "image_path", "image_embedding"],
//     },
//     {
//       headers: {
//         Authorization: `Bearer ${API_KEY}`,
//         Accept: "application/json",
//         "Content-Type": "application/json",
//       },
//     }
//   );
//
//   console.log(response.data.data)
//
//   // Send the output back to the main thread
//   self.postMessage({
//     status: "complete",
//     output: response.data.data,
//   });
// });
