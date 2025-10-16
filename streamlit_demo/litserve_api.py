# server.py
import litserve as ls
from litserve.specs.openai import ChatCompletionRequest
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# Define your model constants
DEFAULT_MODEL = "ddvd233/QoQ-Med-VL-7B"
QWEN2_5_VL_MODELS = {"QoQ-Med-7B": "ddvd233/QoQ-Med-VL-7B",
                     "QoQ-Med-32B": "ddvd233/QoQ-Med-VL-32B"}


def process_vision_info(messages):
    """
    Process image and video inputs from messages.
    Returns empty lists for text-only messages.
    """
    image_inputs = []
    video_inputs = []

    for message in messages:
        content = message.get("content", "")

        # Handle content that's a list (text + images)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Handle image URLs
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            # Extract base64 data
                            import base64
                            from io import BytesIO

                            from PIL import Image

                            try:
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(BytesIO(image_bytes))
                                image_inputs.append(image)
                            except Exception as e:
                                print(f"Error processing image: {e}")

                    # Handle video inputs if needed
                    elif item.get("type") == "video_url":
                        # Process video (not implemented in this example)
                        pass

    return image_inputs, video_inputs


class Qwen25VLAPI(ls.LitAPI):
    def setup(self, device, model_id=DEFAULT_MODEL):
        if model_id not in QWEN2_5_VL_MODELS.values():
            model_id = DEFAULT_MODEL

        # Initialize vLLM model with vision support
        self.model = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=8192,
            limit_mm_per_prompt={"image": 10, "video": 10},  # Support multiple images/videos
            # Enable tensor parallelism if you have multiple GPUs
            tensor_parallel_size=2,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.model_id = model_id

    def decode_request(self, request: ChatCompletionRequest, context: dict):
        # Set the model if different from current
        requested_model = request.model
        model_path = QWEN2_5_VL_MODELS.get(requested_model, DEFAULT_MODEL)

        if model_path != self.model_id:
            self.setup(self.device, model_path)

        # Create SamplingParams for vLLM
        context["sampling_params"] = SamplingParams(
            max_tokens=request.max_tokens if request.max_tokens else 2048,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 0.9,
        )

        # Process messages
        try:
            # Convert the Pydantic model to a dictionary
            messages = [message.model_dump(exclude_none=True) for message in request.messages]

            # Pre-process messages to ensure correct format
            for message in messages:
                # Convert list content to proper format if needed
                if isinstance(message.get("content"), list):
                    # Ensure all text items are properly formatted
                    for i, item in enumerate(message["content"]):
                        if isinstance(item, str):
                            message["content"][i] = {"type": "text", "text": item}

            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare vLLM inputs
            # vLLM expects multi-modal data in a specific format
            multi_modal_data = {}
            if image_inputs:
                multi_modal_data["image"] = image_inputs
            if video_inputs:
                multi_modal_data["video"] = video_inputs

            # Store both prompt and multi-modal data
            return {
                "prompt": text,
                "multi_modal_data": multi_modal_data if multi_modal_data else None,
            }
        except Exception as e:
            # Log the error for debugging
            print(f"Error in decode_request: {e}")
            raise

    def predict(self, model_inputs, context: dict):
        # Extract prompt and multi-modal data
        prompt = model_inputs["prompt"]
        multi_modal_data = model_inputs.get("multi_modal_data")

        # Get sampling parameters
        sampling_params = context["sampling_params"]

        # Generate with vLLM
        # For multi-modal inputs, wrap prompt and data in dict format
        if multi_modal_data:
            vllm_inputs = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            }
            outputs = self.model.generate(vllm_inputs, sampling_params=sampling_params)
        else:
            # Text-only generation
            outputs = self.model.generate(prompt, sampling_params=sampling_params)

        # Extract the generated text from the output
        generated_text = outputs[0].outputs[0].text
        yield generated_text


# Start the server
if __name__ == "__main__":
    api = Qwen25VLAPI(spec=ls.OpenAISpec())
    server = ls.LitServer(api, accelerator="cuda")
    server.run(port=8000)
